"""Underlying bars and volatility indexes for option simulations (Yahoo, via the bar archive)."""

import logging

import pandas as pd

from autotrader.data.archive import BarArchive

log = logging.getLogger(__name__)
VIX, VIX1D = "^VIX", "^VIX1D"


def _get(archive: BarArchive, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
         refresh: bool) -> pd.DataFrame:
    try:
        return archive.get(symbol, interval, start=start, end=end, refresh=refresh)
    except Exception as exc:  # noqa: BLE001 - a missing symbol must not sink the run
        log.warning("no %s %s bars: %s", symbol, interval, exc)
        return pd.DataFrame()


def load_history(underlyings: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp,
                 archive: BarArchive | None = None, refresh: bool = True) -> dict:
    """5m bars from a few days before `start`, daily bars from a year before, VIX levels."""
    archive = archive or BarArchive()
    need = tuple(dict.fromkeys(("SPY",) + tuple(underlyings)))
    b5_start = start - pd.Timedelta(days=7)
    d_start = start - pd.Timedelta(days=400)
    bars5m = {u: _get(archive, u, "5m", b5_start, end, refresh) for u in underlyings}
    daily = {u: _get(archive, u, "1d", d_start, end, refresh) for u in need}

    def level(sym: str) -> pd.Series | None:
        intraday = _get(archive, sym, "5m", b5_start, end, refresh)
        d = _get(archive, sym, "1d", d_start, end, refresh)
        parts = []
        if not d.empty:  # daily closes, stamped at the close (16:00 ET) so they are known then
            idx = (d.index.tz_convert("UTC").normalize() + pd.Timedelta(hours=21))
            parts.append(pd.Series(d["close"].to_numpy(), index=idx))
        if not intraday.empty:
            parts.append((intraday["close"]).set_axis(intraday.index + pd.Timedelta(minutes=5)))
        if not parts:
            return None
        s = pd.concat(parts).sort_index()
        return s[~s.index.duplicated(keep="last")]

    return {"bars5m": {u: b for u, b in bars5m.items() if not b.empty},
            "daily": {u: d for u, d in daily.items() if not d.empty},
            "vix": level(VIX), "vix1d": level(VIX1D)}
