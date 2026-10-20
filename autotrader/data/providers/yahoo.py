"""Yahoo Finance chart API provider (free, unofficial).

Ported from the original stochastics.get_data: same endpoints, query1 -> query2
fallback, and null-row cleaning, but returns a canonical bar frame and goes
through a shared rate limiter.
"""

import pandas as pd
import httpx
from returns.result import Failure, Result, Success

from autotrader.config import get_settings
from autotrader.data.bars import empty_bars, normalize
from autotrader.data.intervals import YAHOO_INTERVAL, YAHOO_MAX_LOOKBACK, Interval
from autotrader.data.ratelimit import TokenBucket

HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )
}


class YahooError(RuntimeError):
    pass


def parse_chart(payload: dict, interval: Interval) -> pd.DataFrame:
    """Turn a chart API JSON payload into a canonical bar frame."""
    chart = payload.get("chart") or {}
    if not chart.get("result"):
        err = (chart.get("error") or {}).get("description", "Unknown error")
        raise YahooError(f"No chart data returned: {err}")

    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return empty_bars()
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    n = len(timestamps)

    def col(name: str) -> list:
        values = quote.get(name) or []
        return list(values[:n]) + [None] * (n - len(values))

    df = pd.DataFrame(
        {
            "open": col("open"),
            "high": col("high"),
            "low": col("low"),
            "close": col("close"),
            "volume": col("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
        dtype="float64",
    )
    df["volume"] = df["volume"].fillna(0.0)
    # Drop rows with nulls in essential price fields, as the original get_data did.
    df = df.dropna(subset=["open", "high", "low", "close"])

    if interval == "1d":
        tz = (result.get("meta") or {}).get("exchangeTimezoneName", "America/New_York")
        local_dates = df.index.tz_convert(tz).normalize().tz_localize(None)
        df.index = pd.DatetimeIndex(local_dates).tz_localize("UTC")
    return normalize(df)


class YahooProvider:
    name = "yahoo"

    def __init__(self, client: httpx.Client | None = None, limiter: TokenBucket | None = None):
        settings = get_settings()
        self._client = client or httpx.Client(headers=HEADERS, timeout=settings.http_timeout_sec)
        self._limiter = limiter or TokenBucket(settings.yahoo_requests_per_sec)

    def _get(self, symbol: str, params: dict) -> dict:
        last_exc: Exception | None = None
        for host in HOSTS:
            self._limiter.acquire()
            try:
                resp = self._client.get(f"{host}/v8/finance/chart/{symbol}", params=params)
                if resp.status_code == 404:
                    return resp.json()
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
        raise YahooError(f"Failed to fetch '{symbol}' from Yahoo: {last_exc}") from last_exc

    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> Result[pd.DataFrame, Exception]:
        now = pd.Timestamp.now(tz="UTC")
        end = end or now
        earliest = now - YAHOO_MAX_LOOKBACK[interval]
        start = max(start, earliest) if start is not None else earliest
        if start >= end:
            return Success(empty_bars())  # nothing older than Yahoo's lookback limit
        params = {
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": YAHOO_INTERVAL[interval],
            "includePrePost": "false",
        }
        try:
            return Success(parse_chart(self._get(symbol, params), interval))
        except Exception as exc:  # noqa: BLE001 - provider boundary
            return Failure(exc)
