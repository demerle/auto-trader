"""The canonical bar frame used everywhere in autotrader.

A bar frame is a DataFrame indexed by a tz-aware UTC DatetimeIndex named "ts"
holding the bar *start* time, with float columns open/high/low/close/volume.
Daily bars are stamped at 00:00 UTC of the exchange-local session date.
"""

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}, index=idx)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical shape: sorted, deduped (last wins), float columns."""
    if df.empty:
        return empty_bars()
    out = df[COLUMNS].astype("float64")
    out.index = pd.DatetimeIndex(out.index, name="ts")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["open", "high", "low", "close"])


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return normalize(new)
    if new.empty:
        return normalize(old)
    return normalize(pd.concat([old, new]))
