"""NYSE session calendar helpers."""

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from autotrader.data.intervals import BAR_DURATION, Interval


@lru_cache
def _nyse():
    return mcal.get_calendar("XNYS")


@lru_cache(maxsize=64)
def _schedule(year: int) -> pd.DataFrame:
    sched = _nyse().schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
    sched.index = pd.DatetimeIndex(sched.index).tz_localize("UTC")
    return sched[["market_open", "market_close"]]


def schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Sessions between start and end; index is session date at 00:00 UTC."""
    frames = [_schedule(y) for y in range(start.year, end.year + 1)]
    sched = pd.concat(frames)
    return sched[(sched.index >= start.normalize()) & (sched.index <= end)]


def session_for(ts: pd.Timestamp) -> pd.Series | None:
    """The session (market_open/market_close) on ts's New York date, if any."""
    day = ts.tz_convert("America/New_York").normalize().tz_localize(None)
    sched = _schedule(day.year)
    key = pd.Timestamp(day).tz_localize("UTC")
    return sched.loc[key] if key in sched.index else None


def is_open(ts: pd.Timestamp) -> bool:
    session = session_for(ts)
    return session is not None and session.market_open <= ts < session.market_close


def bar_close_time(bar_start: pd.Timestamp, interval: Interval) -> pd.Timestamp:
    """When a bar that started at bar_start is complete."""
    if interval == "1d":
        session = session_for(bar_start + pd.Timedelta(hours=12))
        if session is None:
            return bar_start + BAR_DURATION["1d"]
        return session.market_close
    end = bar_start + BAR_DURATION[interval]
    session = session_for(bar_start)
    if session is not None:
        end = min(end, session.market_close)
    return end


def drop_incomplete(df: pd.DataFrame, interval: Interval, now: pd.Timestamp) -> pd.DataFrame:
    """Remove trailing bars that have not closed yet as of `now`."""
    while not df.empty and bar_close_time(df.index[-1], interval) > now:
        df = df.iloc[:-1]
    return df


def latest_bar_close(interval: Interval, now: pd.Timestamp, daily_at: str = "19:05"
                     ) -> pd.Timestamp | None:
    """The close time of the most recent bar that has closed by `now` in today's session.

    For daily bars this is the session close, but only once `now` is past `daily_at`
    (New York time), when the engine runs its daily tick.
    """
    session = session_for(now)
    if session is None:
        return None
    if interval == "1d":
        local = now.tz_convert("America/New_York")
        hh, mm = map(int, daily_at.split(":"))
        return session.market_close if (local.hour, local.minute) >= (hh, mm) else None
    step = BAR_DURATION[interval]
    if now < session.market_open + step:
        return None
    if now >= session.market_close:
        return session.market_close
    k = (now - session.market_open) // step
    return session.market_open + k * step
