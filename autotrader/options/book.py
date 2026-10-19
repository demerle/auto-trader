"""A bot's options book: cash plus open structures (by tag), rebuilt from fills.

Prices are per share; quantities are contracts (x100). Shared by the simulator and
the live ledger, so backtests and paper trading account for money the same way.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce

import pandas as pd

from autotrader.options.occ import MULTIPLIER
from autotrader.options.strategy import ClosedStructure, Leg, OpenStructure


@dataclass
class _Pos:
    legs: dict[str, float] = field(default_factory=dict)  # symbol -> signed contracts
    avg: dict[str, float] = field(default_factory=dict)  # symbol -> avg price per share
    realized: float = 0.0  # dollars realized so far in this structure's life (after fees)
    fees: float = 0.0
    opened_at: pd.Timestamp | None = None
    note: str = ""


class OptionBook:
    def __init__(self, cash: float):
        self.cash = cash
        self.open: dict[str, _Pos] = {}
        self.closed: list[ClosedStructure] = []
        self.trades = 0  # structures closed

    def fill(self, tag: str, symbol: str, qty: float, price: float, fees: float = 0.0,
             ts: pd.Timestamp | None = None, note: str = "") -> None:
        pos = self.open.setdefault(tag, _Pos(opened_at=ts, note=note))
        if note and not pos.note:
            pos.note = note
        self.cash -= qty * price * MULTIPLIER + fees
        pos.fees += fees
        pos.realized -= fees
        q, a = pos.legs.get(symbol, 0.0), pos.avg.get(symbol, 0.0)
        new_q = q + qty
        if q == 0 or math.copysign(1, q) == math.copysign(1, qty):
            pos.avg[symbol] = (q * a + qty * price) / new_q
        else:
            closed = min(abs(qty), abs(q))
            pos.realized += closed * (price - a) * math.copysign(1, q) * MULTIPLIER
            if new_q and math.copysign(1, new_q) != math.copysign(1, q):
                pos.avg[symbol] = price
        if abs(new_q) < 1e-9:
            pos.legs.pop(symbol, None)
            pos.avg.pop(symbol, None)
        else:
            pos.legs[symbol] = new_q
        if not pos.legs:
            self.closed.append(ClosedStructure(tag, (), 0, pos.realized, pos.opened_at, ts))
            self.trades += 1
            del self.open[tag]

    def cash_flow(self, amount: float) -> None:
        self.cash += amount

    # ----- views -----
    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for pos in self.open.values():
            for s, q in pos.legs.items():
                out[s] = out.get(s, 0.0) + q
        return {s: q for s, q in out.items() if abs(q) > 1e-9}

    def units(self, tag: str) -> int:
        qs = [int(round(abs(q))) for q in self.open[tag].legs.values()]
        return reduce(math.gcd, qs) if qs else 0

    def legs_of(self, tag: str) -> tuple[Leg, ...]:
        n = self.units(tag) or 1
        return tuple(sorted((Leg(s, int(round(q / n))) for s, q in self.open[tag].legs.items()),
                            key=lambda leg: leg.symbol))

    def value(self, marks: Mapping[str, float], tag: str | None = None) -> float:
        tags = [tag] if tag else list(self.open)
        return sum(q * marks[s] * MULTIPLIER for t in tags for s, q in self.open[t].legs.items())

    def equity(self, marks: Mapping[str, float]) -> float:
        return self.cash + self.value(marks)

    def snapshot(self, marks: Mapping[str, float]) -> dict[str, OpenStructure]:
        out = {}
        for tag, pos in self.open.items():
            n = self.units(tag) or 1
            entry = sum(q * pos.avg[s] for s, q in pos.legs.items()) / n
            mark = sum(q * marks[s] for s, q in pos.legs.items()) / n
            pnl = pos.realized + sum(q * (marks[s] - pos.avg[s]) * MULTIPLIER for s, q in pos.legs.items())
            out[tag] = OpenStructure(tag, self.legs_of(tag), n, entry, mark, pnl, pos.opened_at, pos.note)
        return out
