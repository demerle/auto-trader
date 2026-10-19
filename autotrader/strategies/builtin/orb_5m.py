"""Opening-range breakout on 5-minute bars.

The opening range is the high/low of the first `range_bars` bars of the session.
The first close outside the range that day opens a position in that direction;
it is closed if price closes back through the opposite side of the range, and
always before the session ends. At most one entry per symbol per day.
"""

import pandas as pd

from autotrader.strategies.base import Strategy, StrategyContext, StrategyParams, Targets


class OrbParams(StrategyParams):
    range_bars: int = 6  # 30 minutes
    flat_after_minute: int = 15 * 60 + 45  # New York minutes-of-day; last entry/forced exit
    allow_short: bool = True
    weight: float = 0.125


class OpeningRangeBreakout(Strategy):
    name = "orb_5m"
    version = "1"
    interval = "5m"
    universe = "core"
    description = "30-minute opening-range breakout, long or short, flat by the close."
    Params = OrbParams

    @property
    def lookback(self) -> int:
        return 100

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        local = df.index.tz_convert("America/New_York")
        day = pd.Series(local.date, index=df.index)
        bar_no = day.groupby(day).cumcount()
        in_range = bar_no < p.range_bars
        # Only valid once the whole range has closed (bar_no >= range_bars - 1); no lookahead.
        or_high = df["high"].where(in_range).groupby(day).cummax().where(bar_no >= p.range_bars - 1)
        or_low = df["low"].where(in_range).groupby(day).cummin().where(bar_no >= p.range_bars - 1)
        or_high = or_high.groupby(day).ffill()
        or_low = or_low.groupby(day).ffill()
        after = bar_no >= p.range_bars
        up = after & (df["close"] > or_high)
        down = after & (df["close"] < or_low)
        return pd.DataFrame({
            "close": df["close"],
            "or_high": or_high,
            "or_low": or_low,
            "breaks_up": up.astype(int).groupby(day).cumsum(),
            "breaks_down": down.astype(int).groupby(day).cumsum(),
            "up": up,
            "down": down,
            "minute": pd.Series(local.hour * 60 + local.minute, index=df.index),
        })

    def on_bar(self, ctx: StrategyContext) -> Targets:
        p = self.params
        targets = {s: w for s, w in ctx.weights.items() if w != 0}
        before = dict(targets)
        for symbol, f in ctx.features.items():
            if not ctx.is_fresh(symbol):
                continue
            row = f.iloc[-1]
            pos = targets.get(symbol, 0.0)
            if row.minute >= p.flat_after_minute:
                targets.pop(symbol, None)
            elif pos > 0 and row.close < row.or_low or pos < 0 and row.close > row.or_high:
                targets.pop(symbol, None)
            elif pos == 0 and row.up and row.breaks_up == 1 and row.breaks_down == 0:
                targets[symbol] = p.weight
            elif pos == 0 and p.allow_short and row.down and row.breaks_down == 1 \
                    and row.breaks_up == 0:
                targets[symbol] = -p.weight
        if targets.keys() == before.keys() and all(
            (targets[s] > 0) == (before[s] > 0) for s in targets
        ):
            return None
        return {s: (p.weight if w > 0 else -p.weight) for s, w in targets.items()}
