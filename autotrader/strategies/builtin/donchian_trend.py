"""Turtle-style Donchian breakout: buy a new N-day high, exit on an M-day low. Long-only."""

import pandas as pd

from autotrader.indicators import donchian
from autotrader.strategies.base import Strategy, StrategyContext, StrategyParams, Targets


class DonchianParams(StrategyParams):
    entry_period: int = 55
    exit_period: int = 20
    max_positions: int = 10


class DonchianTrend(Strategy):
    name = "donchian_trend"
    version = "1"
    interval = "1d"
    universe = "mega_caps"
    description = "Donchian channel breakout trend following on mega caps (55-day in, 20-day out)."
    Params = DonchianParams

    @property
    def lookback(self) -> int:
        return max(self.params.entry_period, self.params.exit_period) + 5

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return pd.DataFrame({
            "close": df["close"],
            "entry": donchian(df, p.entry_period)["upper"],
            "exit": donchian(df, p.exit_period)["lower"],
        })

    def on_bar(self, ctx: StrategyContext) -> Targets:
        p = self.params
        held = {s for s, w in ctx.weights.items() if w > 0}
        before = set(held)
        # Rank breakouts by strength so a crowded day fills the strongest first.
        candidates = []
        for symbol, f in ctx.features.items():
            if not ctx.is_fresh(symbol):
                continue
            row = f.iloc[-1]
            if row.isna().any():
                continue
            if symbol in held and row.close < row.exit:
                held.discard(symbol)
            elif symbol not in held and row.close > row.entry:
                candidates.append((row.close / row.entry, symbol))
        for _, symbol in sorted(candidates, reverse=True):
            if len(held) >= p.max_positions:
                break
            held.add(symbol)
        if held == before:
            return None
        # Held positions keep their current weight; only new entries are sized.
        return {s: ctx.weight(s) or 1.0 / p.max_positions for s in sorted(held)}
