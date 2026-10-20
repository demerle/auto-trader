"""Local bar archive: one Parquet file per (interval, symbol).

Free intraday history is short (Yahoo 1m ~ 7 days), so every bar fetched is
kept forever. Backtests read from here.
"""

import logging
from pathlib import Path

import pandas as pd
from returns.result import Failure, Success

from autotrader.config import get_settings
from autotrader.data.bars import empty_bars, merge, normalize
from autotrader.data.calendar import drop_incomplete
from autotrader.data.intervals import BAR_DURATION, Interval
from autotrader.data.providers.base import BarProvider

log = logging.getLogger(__name__)


class BarArchive:
    def __init__(self, root: Path | None = None, providers: list[BarProvider] | None = None):
        self.root = Path(root or get_settings().bar_dir)
        if providers is None:
            from autotrader.data.providers.stooq import StooqProvider
            from autotrader.data.providers.yahoo import YahooProvider

            providers = [YahooProvider(), StooqProvider()]
        self.providers = providers

    def path(self, symbol: str, interval: Interval) -> Path:
        return self.root / interval / f"{symbol.upper()}.parquet"

    def load(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        p = self.path(symbol, interval)
        if not p.exists():
            return empty_bars()
        df = normalize(pd.read_parquet(p))
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    def save(self, symbol: str, interval: Interval, df: pd.DataFrame) -> pd.DataFrame:
        p = self.path(symbol, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        merged = merge(self.load(symbol, interval), df)
        tmp = p.with_suffix(".tmp")
        merged.to_parquet(tmp)
        tmp.replace(p)
        return merged

    def update(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp | None = None,
        now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch bars newer than what is archived (or from `start`) and store closed ones."""
        now = now or pd.Timestamp.now(tz="UTC")
        existing = self.load(symbol, interval)
        if existing.empty:
            return self._fetch_and_save(symbol, interval, start, now, now)
        if start is not None and start < existing.index[0] - BAR_DURATION[interval]:
            # Backfill history older than what is archived. Failure is not fatal: the
            # symbol may simply not have existed that far back.
            try:
                self._fetch_and_save(symbol, interval, start, existing.index[0], now)
            except RuntimeError as exc:
                log.info("no older history for %s %s: %s", symbol, interval, exc)
        # Re-fetch the last bar too, in case it was revised.
        return self._fetch_and_save(symbol, interval, existing.index[-1] - BAR_DURATION[interval],
                                    now, now)

    def _fetch_and_save(self, symbol: str, interval: Interval, start: pd.Timestamp | None,
                        end: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
        errors: list[Exception] = []
        for provider in self.providers:
            match provider.fetch_bars(symbol, interval, start=start, end=end):
                case Success(df):
                    return self.save(symbol, interval, drop_incomplete(df, interval, now))
                case Failure(exc):
                    errors.append(exc)
        raise RuntimeError(f"All providers failed for {symbol} {interval}: {errors}")

    def get(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        refresh: bool = True,
    ) -> pd.DataFrame:
        if refresh:
            self.update(symbol, interval, start=start)
        return self.load(symbol, interval, start=start, end=end)
