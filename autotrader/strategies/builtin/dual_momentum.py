"""Antonacci Global Equities Momentum (dual momentum), rebalanced monthly.

If US equities beat T-bills over the lookback, hold whichever of US/international
equities is stronger; otherwise hold bonds.
"""

import pandas as pd

from autotrader.strategies.base import Strategy, StrategyContext, StrategyParams, Targets


class DualMomentumParams(StrategyParams):
    lookback_bars: int = 252
    us: str = "SPY"
    intl: str = "EFA"
    bonds: str = "AGG"
    cash: str = "BIL"


class DualMomentum(Strategy):
    name = "dual_momentum"
    version = "1"
    interval = "1d"
    universe = ("SPY", "EFA", "AGG", "BIL")
    description = "Antonacci dual momentum (SPY/EFA vs T-bills, else AGG), monthly."
    Params = DualMomentumParams

    @property
    def lookback(self) -> int:
        return self.params.lookback_bars + 5

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        n = self.params.lookback_bars
        return pd.DataFrame({"ret": df["close"] / df["close"].shift(n) - 1})

    def on_bar(self, ctx: StrategyContext) -> Targets:
        p = self.params
        us = ctx.bars.get(p.us)
        if us is None or len(us) < 2 or us.index[-1] != ctx.now:
            return None
        new_month = us.index[-1].month != us.index[-2].month
        if not new_month and ctx.weights:
            return None
        rets = {}
        for s in (p.us, p.intl, p.cash):
            f = ctx.features.get(s)
            if f is None or f.empty or f.ret.iloc[-1] != f.ret.iloc[-1]:
                return None
            rets[s] = float(f.ret.iloc[-1])
        if rets[p.us] > rets[p.cash]:
            pick = p.us if rets[p.us] >= rets[p.intl] else p.intl
        else:
            pick = p.bonds
        return {pick: 1.0}
