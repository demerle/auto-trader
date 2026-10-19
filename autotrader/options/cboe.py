"""CBOE's free delayed option chains (cdn.cboe.com), one JSON file per underlying.

Each contract comes with bid/ask, IV, greeks, open interest and volume computed by
CBOE. Quotes are ~15 minutes delayed; `quote_time` is the generation timestamp minus
that delay, and `spot` is the (equally delayed) underlying price the quotes go with.
"""

import logging
from dataclasses import dataclass
from datetime import date

import httpx
import numpy as np
import pandas as pd

from autotrader.config import get_settings
from autotrader.data.ratelimit import TokenBucket
from autotrader.options.occ import OCC_RE

log = logging.getLogger(__name__)

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/128.0 Safari/537.36", "Accept": "application/json"}
DELAY = pd.Timedelta(minutes=15)
COLUMNS = ["symbol", "underlying", "expiry", "right", "strike", "bid", "ask", "last", "iv",
           "delta", "gamma", "theta", "vega", "oi", "volume"]
_bucket = TokenBucket(2.0, capacity=4)


class ChainError(RuntimeError):
    pass


@dataclass
class ChainSnapshot:
    underlying: str
    fetched_at: pd.Timestamp  # CBOE's generation time (UTC)
    quote_time: pd.Timestamp  # what the quotes reflect (fetched_at - delay)
    spot: float  # underlying price at quote_time
    chain: pd.DataFrame  # COLUMNS


def parse(payload: dict, max_dte: int = 45, moneyness: float = 0.25,
          today: date | None = None) -> ChainSnapshot:
    data = payload["data"]
    fetched = pd.Timestamp(payload["timestamp"], tz="UTC")
    spot = float(data.get("current_price") or data.get("close") or 0.0)
    if spot <= 0:
        raise ChainError(f"{data.get('symbol')}: no underlying price in the chain")
    today = today or fetched.tz_convert("America/New_York").date()
    rows = []
    for o in data.get("options", []):
        m = OCC_RE.match(o.get("option", ""))
        if not m:
            continue
        root, yy, mm, dd, right, strike = m.groups()
        expiry = date(2000 + int(yy), int(mm), int(dd))
        k = int(strike) / 1000
        if (expiry - today).days > max_dte or expiry < today or abs(k / spot - 1) > moneyness:
            continue
        rows.append((o["option"], root, expiry, right, k, o.get("bid") or 0.0, o.get("ask") or 0.0,
                     o.get("last_trade_price") or 0.0, o.get("iv") or 0.0, o.get("delta") or 0.0,
                     o.get("gamma") or 0.0, o.get("theta") or 0.0, o.get("vega") or 0.0,
                     o.get("open_interest") or 0.0, o.get("volume") or 0.0))
    chain = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS[4:]:
        chain[c] = chain[c].astype(float)
    chain["iv"] = chain["iv"].where(chain["iv"] > 0, np.nan)
    chain = chain.sort_values(["expiry", "strike", "right"]).reset_index(drop=True)
    return ChainSnapshot(str(data.get("symbol", "")).upper(), fetched, fetched - DELAY, spot, chain)


def fetch_chain(underlying: str, max_dte: int = 45, moneyness: float = 0.25,
                client: httpx.Client | None = None) -> ChainSnapshot:
    _bucket.acquire()
    url = URL.format(symbol=underlying.upper())
    try:
        if client is not None:
            resp = client.get(url, headers=HEADERS, follow_redirects=True)
        else:
            resp = httpx.get(url, headers=HEADERS, follow_redirects=True,
                             timeout=get_settings().http_timeout_sec)
        resp.raise_for_status()
        return parse(resp.json(), max_dte, moneyness)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise ChainError(f"CBOE chain for {underlying}: {exc}") from exc
