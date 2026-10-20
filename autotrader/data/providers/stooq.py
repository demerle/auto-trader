"""Stooq daily CSV provider, used as a fallback for daily bars."""

import io

import httpx
import pandas as pd
from returns.result import Failure, Result, Success

from autotrader.config import get_settings
from autotrader.data.bars import normalize
from autotrader.data.intervals import Interval


class StooqProvider:
    name = "stooq"

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=get_settings().http_timeout_sec)

    @staticmethod
    def _symbol(symbol: str) -> str:
        return symbol.lower().replace(".", "-") + ".us"

    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> Result[pd.DataFrame, Exception]:
        if interval != "1d":
            return Failure(ValueError("Stooq provider only serves daily bars"))
        try:
            resp = self._client.get(
                "https://stooq.com/q/d/l/", params={"s": self._symbol(symbol), "i": "d"}
            )
            resp.raise_for_status()
            if not resp.text.startswith("Date"):
                raise ValueError(f"Unexpected Stooq response for {symbol}: {resp.text[:80]!r}")
            raw = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
            raw.columns = [c.lower() for c in raw.columns]
            df = raw.set_index(pd.DatetimeIndex(raw["date"]).tz_localize("UTC"))
            if "volume" not in df:
                df["volume"] = 0.0
            df = normalize(df)
            if start is not None:
                df = df[df.index >= start.normalize()]
            if end is not None:
                df = df[df.index <= end]
            return Success(df)
        except Exception as exc:  # noqa: BLE001 - provider boundary
            return Failure(exc)
