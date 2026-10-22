"""The options strategy contract.

Every `tick_minutes` the engine calls `on_tick(ctx)`. A strategy answers with the
structures it wants open, keyed by a tag it chooses:

    {"spy-call": Structure((Leg("SPY261020C00770000", 1),), qty=3, note="ORB up")}

- a new tag is opened, a tag that disappears is closed, a changed qty is resized,
  changed legs mean close-then-open (a roll);
- returning None means "keep everything as it is".

Strategies are pure: everything they may know is in the context (chains are CBOE's
~15-minute-delayed IV repriced at the live underlying price). The engine computes
each structure's worst-case loss itself and refuses undefined risk, structures
bigger than the risk limits, and anything that would let the bot lose more than
its equity. Expiring contracts are closed before Alpaca's cutoff, and everything
is closed before Friday's scoring, whatever the strategy says.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

from autotrader.options.helpers import Chain


@dataclass(frozen=True)
class Leg:
    symbol: str  # OCC symbol
    ratio: int  # per unit of the structure: +1 long, -1 short, +2 ...


@dataclass(frozen=True)
class Structure:
    legs: tuple[Leg, ...]
    qty: int  # units of the whole structure (>= 1)
    note: str = ""  # why; shown in the trade feed

    def key(self) -> tuple:
        return tuple(sorted((leg.symbol, leg.ratio) for leg in self.legs))


@dataclass(frozen=True)
class OpenStructure:
    tag: str
    legs: tuple[Leg, ...]
    qty: int
    entry: float  # net premium per unit at entry, per share (+ debit paid, - credit received)
    mark: float  # net value per unit now, per share (same sign convention)
    pnl: float  # dollars, all units, after fees so far
    opened_at: pd.Timestamp
    note: str = ""

    @property
    def pnl_pct(self) -> float:
        """P&L relative to the premium put up (debit) or the credit received."""
        return (self.mark - self.entry) / abs(self.entry) if self.entry else 0.0


@dataclass(frozen=True)
class ClosedStructure:
    tag: str
    legs: tuple[Leg, ...]
    qty: int
    pnl: float
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp


@dataclass(frozen=True)
class OptionContext:
    now: pd.Timestamp  # UTC
    chains: Mapping[str, Chain]  # nowcast chains per underlying
    spot: Mapping[str, float]  # live underlying prices
    bars: Mapping[str, pd.DataFrame]  # 5-minute bars, several sessions, up to now
    daily: Mapping[str, pd.DataFrame]  # daily bars (last ~1 year), completed sessions only
    vix: float | None  # latest ^VIX
    vix1d: float | None  # latest ^VIX1D (expected move for one day)
    positions: Mapping[str, OpenStructure]
    closed_today: tuple[ClosedStructure, ...]
    equity: float
    cash: float
    week_start_equity: float
    session_open: pd.Timestamp
    session_close: pd.Timestamp
    week_last_session: bool  # today is the week's scoring day (Friday, or earlier on holidays)
    extras: Mapping[str, object] = field(default_factory=dict)

    @property
    def et(self) -> pd.Timestamp:
        return self.now.tz_convert("America/New_York")

    @property
    def minutes_since_open(self) -> float:
        return (self.now - self.session_open).total_seconds() / 60

    @property
    def minutes_to_close(self) -> float:
        return (self.session_close - self.now).total_seconds() / 60

    @property
    def week_pnl(self) -> float:
        return self.equity - self.week_start_equity

    def today_bars(self, symbol: str) -> pd.DataFrame:
        df = self.bars.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df.index >= self.session_open]


Targets = dict[str, Structure] | None


class OptionParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OptionStrategy(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    underlyings: ClassVar[tuple[str, ...]]
    bar_symbols: ClassVar[tuple[str, ...]] = ()  # builtins only: extra symbols whose 5m bars appear in ctx.bars
    description: ClassVar[str] = ""
    thesis: ClassVar[str] = ""  # why this should make money, in a few sentences
    Params: ClassVar[type[OptionParams]] = OptionParams

    def __init__(self, params: Mapping | OptionParams | None = None):
        if isinstance(params, OptionParams):
            self.params = params
        else:
            self.params = self.Params.model_validate(dict(params or {}))

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @abstractmethod
    def on_tick(self, ctx: OptionContext) -> Targets:
        """Structures to hold after this tick, by tag (None = no change)."""


def encode_targets(targets: Targets) -> dict | None:
    """Plain JSON for the sandbox protocol; validates shapes."""
    if targets is None:
        return None
    if not isinstance(targets, dict):
        raise TypeError("on_tick() must return a dict of tag -> Structure, or None")
    out = {}
    for tag, s in targets.items():
        if not isinstance(tag, str) or not tag or len(tag) > 40:
            raise ValueError(f"bad tag {tag!r} (non-empty str, <= 40 chars)")
        if not isinstance(s, Structure):
            raise TypeError(f"{tag}: expected a Structure, got {type(s).__name__}")
        qty = int(s.qty)
        if qty != s.qty or qty < 0:
            raise ValueError(f"{tag}: qty must be a non-negative whole number")
        legs = []
        for leg in s.legs:
            if not isinstance(leg, Leg) or int(leg.ratio) != leg.ratio or not leg.ratio:
                raise ValueError(f"{tag}: legs must be Leg(symbol, non-zero int ratio)")
            legs.append([str(leg.symbol), int(leg.ratio)])
        out[tag] = {"legs": legs, "qty": qty, "note": str(s.note)[:200]}
    return out


def decode_targets(data: dict | None) -> Targets:
    if data is None:
        return None
    return {tag: Structure(tuple(Leg(sym, int(r)) for sym, r in v["legs"]), int(v["qty"]),
                           v.get("note", "")) for tag, v in data.items()}
