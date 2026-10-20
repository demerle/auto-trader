"""Data-quality gate: a symbol with issues is not traded on that tick."""

from dataclasses import dataclass

import pandas as pd

from autotrader.data.calendar import bar_close_time
from autotrader.data.intervals import BAR_DURATION, Interval


@dataclass(frozen=True)
class QualityIssue:
    symbol: str
    kind: str
    detail: str


def check_bars(
    symbol: str,
    df: pd.DataFrame,
    interval: Interval,
    now: pd.Timestamp | None = None,
    max_jump: float = 0.4,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if df.empty:
        return [QualityIssue(symbol, "empty", "no bars")]

    last = df.iloc[-1]
    if (last[["open", "high", "low", "close"]] <= 0).any():
        issues.append(QualityIssue(symbol, "non_positive", f"non-positive price at {df.index[-1]}"))
    if last.high < max(last.open, last.close) or last.low > min(last.open, last.close):
        issues.append(QualityIssue(symbol, "ohlc_inconsistent", f"high/low inconsistent at {df.index[-1]}"))
    if last.volume == 0 and interval != "1d":
        issues.append(QualityIssue(symbol, "zero_volume", f"zero volume at {df.index[-1]}"))

    if len(df) >= 2:
        prev_close = df.close.iloc[-2]
        jump = abs(last.open / prev_close - 1) if prev_close > 0 else 0.0
        if jump > max_jump:
            issues.append(
                QualityIssue(symbol, "price_jump", f"{jump:.0%} gap; possible split or bad print")
            )

    if now is not None:
        # Stale: last bar closed more than ~3 bar durations (or 4 days for daily) ago.
        slack = pd.Timedelta(days=4) if interval == "1d" else BAR_DURATION[interval] * 3
        closed_at = bar_close_time(df.index[-1], interval)
        if now - closed_at > slack:
            issues.append(QualityIssue(symbol, "stale", f"last bar closed {closed_at}"))
    return issues
