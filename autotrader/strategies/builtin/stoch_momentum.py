"""Fast/slow stochastic + momentum reversal: the project's original signal.

Buy when momentum is non-positive and both stochastics are oversold with
slow %K crossing above slow %D; sell when momentum is positive and both are
overbought with slow %K below slow %D. Long-only, equal weight.
"""

import pandas as pd

from autotrader.indicators import fast_stochastic, momentum_pct, slow_stochastic
from autotrader.strategies.base import Strategy, StrategyContext, StrategyParams, Targets


class StochMomentumParams(StrategyParams):
    k_period: int = 21
    d_period: int = 5
    slow_d_period: int = 5
    momentum_period: int = 21
    overbought: float = 80.0
    oversold: float = 20.0
    max_positions: int = 8


class StochMomentum(Strategy):
    name = "stoch_momentum"
    version = "1"
    interval = "1d"
    universe = "core"
    description = "Stochastic oversold/overbought reversal gated by momentum sign (original signal)."
    Params = StochMomentumParams

    @property
    def lookback(self) -> int:
        p = self.params
        return max(p.k_period + p.d_period + p.slow_d_period, p.momentum_period + 1) + 5

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return pd.concat([
            fast_stochastic(df, p.k_period, p.d_period),
            slow_stochastic(df, p.k_period, p.d_period, p.slow_d_period),
            momentum_pct(df["close"], p.momentum_period).rename("momentum_pct"),
        ], axis=1)

    def signal_from(self, row: pd.Series) -> str | None:
        p = self.params
        if row.isna().any():
            return None
        if row.momentum_pct > 0:
            if row.fast_k > p.overbought and row.slow_k > p.overbought and row.slow_k < row.slow_d:
                return "sell"
        elif row.fast_k < p.oversold and row.slow_k < p.oversold and row.slow_k > row.slow_d:
            return "buy"
        return None

    def signal(self, df: pd.DataFrame) -> str | None:
        return self.signal_from(self.features(df).iloc[-1])

    def on_bar(self, ctx: StrategyContext) -> Targets:
        held = {s for s, w in ctx.weights.items() if w > 0}
        changed = False
        for symbol, feats in ctx.features.items():
            if not ctx.is_fresh(symbol):
                continue
            sig = self.signal_from(feats.iloc[-1])
            if sig == "buy" and symbol not in held and len(held) < self.params.max_positions:
                held.add(symbol)
                changed = True
            elif sig == "sell" and symbol in held:
                held.discard(symbol)
                changed = True
        if not changed:
            return None
        # Held positions keep their current weight; only new entries are sized.
        weight = 1.0 / self.params.max_positions
        return {s: ctx.weight(s) or weight for s in sorted(held)}
