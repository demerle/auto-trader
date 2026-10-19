"""The strategy plugin contract.

A strategy is pure and deterministic: it sees only the closed bars handed to
it in a StrategyContext and returns target portfolio weights. The same code
runs unchanged in backtests, paper trading and live trading. Strategies must
never do I/O, touch the network or talk to the broker; see `lint.py`.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

from autotrader.config.universes import resolve
from autotrader.data.intervals import Interval

# Complete desired portfolio as {symbol: weight of bot equity}, weights in [-1, 1].
# Held symbols missing from the mapping are closed. None means "hold everything as is".
Targets = dict[str, float] | None


class StrategyParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class StrategyContext:
    now: pd.Timestamp
    """Start time of the latest closed bar."""
    bars: Mapping[str, pd.DataFrame]
    """Up to `lookback` most recent closed bars per symbol (only symbols with data)."""
    features: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    """Output of Strategy.features per symbol, sliced to the same rows as `bars`."""
    weights: Mapping[str, float] = field(default_factory=dict)
    """Current portfolio weights of this bot (signed, fraction of equity)."""
    equity: float = 0.0

    def weight(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.0)

    def is_fresh(self, symbol: str) -> bool:
        """True if `symbol` has a bar at `now` (i.e. it traded on the latest bar)."""
        df = self.bars.get(symbol)
        return df is not None and not df.empty and df.index[-1] == self.now


class Strategy(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    interval: ClassVar[Interval]
    universe: ClassVar[str | tuple[str, ...]]
    description: ClassVar[str] = ""
    Params: ClassVar[type[StrategyParams]] = StrategyParams

    def __init__(self, params: StrategyParams | Mapping | None = None):
        if isinstance(params, StrategyParams):
            self.params = params
        else:
            self.params = self.Params.model_validate(dict(params or {}))

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def symbols(self) -> tuple[str, ...]:
        return resolve(self.universe)

    @property
    def lookback(self) -> int:
        """How many bars of history per symbol `on_bar` needs. Override when params matter."""
        return 250

    def features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Optional per-symbol indicators, computed once over a whole bar series.

        Must be causal: row t may depend only on rows <= t. This is enforced by
        `autotrader.strategies.validation.check_causal` before a strategy is accepted.
        """
        return None

    @abstractmethod
    def on_bar(self, ctx: StrategyContext) -> Targets: ...
