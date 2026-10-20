from typing import Literal

import pandas as pd

Interval = Literal["1m", "5m", "15m", "1h", "1d"]
INTERVALS: tuple[Interval, ...] = ("1m", "5m", "15m", "1h", "1d")

BAR_DURATION: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}

# Regular-session bars per year, used to annualize metrics.
PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "1h": 252 * 7,
    "1d": 252,
}

# Yahoo uses "60m" for hourly bars.
YAHOO_INTERVAL: dict[str, str] = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}

# Approximate maximum lookback Yahoo serves per interval.
YAHOO_MAX_LOOKBACK: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(days=7),
    "5m": pd.Timedelta(days=59),
    "15m": pd.Timedelta(days=59),
    "1h": pd.Timedelta(days=729),
    "1d": pd.Timedelta(days=365 * 30),
}
