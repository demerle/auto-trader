"""Connors RSI(2) mean reversion: buy short-term oversold dips inside a long-term uptrend.

Entry: close > SMA(200) and RSI(2) < entry_rsi. Exit: close > SMA(5). Long-only, equal weight.
"""

import pandas as pd

from autotrader.indicators import rsi, sma
from autotrader.strategies.base import Strategy, StrategyContext, StrategyParams, Targets


class Rsi2Params(StrategyParams):
    rsi_period: int = 2
    entry_rsi: float = 10.0
    trend_period: int = 200
    exit_period: int = 5
    max_positions: int = 5


class Rsi2Reversion(Strategy):
    name = "rsi2_reversion"
    version = "1"
    interval = "1d"
    universe = "liquid_etfs"
    description = "Connors RSI(2) dip-buying above the 200-day SMA; exit above the 5-day SMA."
    Params = Rsi2Params

    @property
    def lookback(self) -> int:
        return self.params.trend_period + 5

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return pd.DataFrame({
            "close": df["close"],
            "rsi": rsi(df["close"], p.rsi_period),
            "trend": sma(df["close"], p.trend_period),
            "exit": sma(df["close"], p.exit_period),
        })

    def on_bar(self, ctx: StrategyContext) -> Targets:
        p = self.params
        held = {s for s, w in ctx.weights.items() if w > 0}
        before = set(held)
        for symbol, f in ctx.features.items():
            if not ctx.is_fresh(symbol):
                continue
            row = f.iloc[-1]
            if row.isna().any():
                continue
            if symbol in held and row.close > row.exit:
                held.discard(symbol)
            elif symbol not in held and row.close > row.trend and row.rsi < p.entry_rsi \
                    and len(held) < p.max_positions:
                held.add(symbol)
        if held == before:
            return None
        # Held positions keep their current weight; only new entries are sized.
        return {s: ctx.weight(s) or 1.0 / p.max_positions for s in sorted(held)}
