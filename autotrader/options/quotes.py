"""Real-time option and stock quotes for the gauntlet. Alpaca only trades; it never prices.

Schwab's Trader API (Individual) serves real-time NBBO quotes and chains free with a Schwab account.
Its OAuth refresh token lasts 7 days: `autotrader options schwab-login` renews it. CBOE's free chain
(15 minutes delayed) is the fallback; while it is in use, `live` is False and the desk opens nothing.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from autotrader.data.ratelimit import TokenBucket
from autotrader.options import cboe
from autotrader.options.cboe import COLUMNS, ChainSnapshot
from autotrader.options.occ import OCC_RE

log = logging.getLogger(__name__)
TOKEN_DAYS = 7  # Schwab refresh tokens expire 7 days after the browser login


class QuoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    ts: pd.Timestamp  # when the quote was current (UTC)
    last: float = 0.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    def age(self, now: pd.Timestamp) -> float:
        return max((now - self.ts).total_seconds(), 0.0)

    def usable(self, now: pd.Timestamp, max_age: float) -> bool:
        """A two-sided (or zero-bid) quote, not crossed, at most `max_age` seconds old."""
        return self.ask > 0 and self.ask >= self.bid >= 0 and self.age(now) <= max_age

    def as_dict(self) -> dict:
        return {"bid": self.bid, "ask": self.ask, "bid_size": self.bid_size, "ask_size": self.ask_size,
                "ts": self.ts.isoformat()}


class QuoteSource(Protocol):
    live: bool  # real-time (False: delayed fallback; open nothing)

    def chain(self, underlying: str, max_dte: int, moneyness: float) -> ChainSnapshot: ...

    def quotes(self, symbols: list[str]) -> dict[str, Quote]: ...  # OCC option symbols

    def spot(self, symbols: list[str]) -> dict[str, Quote]: ...  # stocks/ETFs


# ----- Schwab -----

def schwab_symbol(occ: str) -> str:
    """SPY261027C00680000 -> 'SPY   261027C00680000' (Schwab pads the root to 6)."""
    m = OCC_RE.match(occ)
    if not m:
        raise ValueError(f"not an OCC option symbol: {occ!r}")
    return f"{m.group(1):<6}{occ[len(m.group(1)):]}"


def _num(x, default=np.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(v) or v <= -999 else v


def _ms(x) -> pd.Timestamp | None:
    v = _num(x, 0.0)
    return pd.Timestamp(int(v), unit="ms", tz="UTC") if v > 0 else None


def parse_schwab_chain(payload: dict, now: pd.Timestamp, max_dte: int = 45, moneyness: float = 0.25,
                       today: date | None = None) -> ChainSnapshot:
    """A /marketdata/v1/chains response -> ChainSnapshot (cboe.COLUMNS + bid_size, ask_size, quote_ts)."""
    und = payload.get("underlying") or {}
    spot = _num(und.get("last"), 0.0) or _num(und.get("mark"), 0.0) or _num(payload.get("underlyingPrice"), 0.0)
    symbol = str(payload.get("symbol", "")).upper()
    if payload.get("status") not in (None, "SUCCESS") or spot <= 0:
        raise QuoteError(f"{symbol}: Schwab chain status {payload.get('status')!r}, spot {spot}")
    today = today or now.tz_convert("America/New_York").date()
    delayed = bool(payload.get("isDelayed")) or bool(und.get("delayed"))
    rows = []
    for key in ("callExpDateMap", "putExpDateMap"):
        for strikes in (payload.get(key) or {}).values():
            for contracts in strikes.values():
                for o in contracts:
                    occ = str(o.get("symbol", "")).replace(" ", "")
                    m = OCC_RE.match(occ)
                    if not m:
                        continue
                    root, yy, mm, dd, right, strike = m.groups()
                    expiry = date(2000 + int(yy), int(mm), int(dd))
                    k = int(strike) / 1000
                    if (expiry - today).days > max_dte or expiry < today or abs(k / spot - 1) > moneyness:
                        continue
                    iv = _num(o.get("volatility"))
                    qt = _ms(o.get("quoteTimeInLong"))
                    rows.append((occ, root, expiry, right, k, _num(o.get("bid"), 0.0), _num(o.get("ask"), 0.0),
                                 _num(o.get("last"), 0.0), iv / 100 if iv > 0 else np.nan,
                                 _num(o.get("delta")), _num(o.get("gamma")), _num(o.get("theta")),
                                 _num(o.get("vega")), _num(o.get("openInterest"), 0.0),
                                 _num(o.get("totalVolume"), 0.0), _num(o.get("bidSize"), 0.0),
                                 _num(o.get("askSize"), 0.0), qt))
    chain = pd.DataFrame(rows, columns=COLUMNS + ["bid_size", "ask_size", "quote_ts"])
    for c in COLUMNS[4:] + ["bid_size", "ask_size"]:
        chain[c] = chain[c].astype(float)
    chain = chain.sort_values(["expiry", "strike", "right"]).reset_index(drop=True)
    # A quote that has not changed is still the current NBBO, so a real-time chain is as of the fetch.
    quote_time = now - cboe.DELAY if delayed else now
    return ChainSnapshot(symbol, now, quote_time, spot, chain, live=not delayed)


def _quote(q: dict, realtime: bool, now: pd.Timestamp) -> Quote:
    ts = _ms(q.get("quoteTime")) or now
    if not realtime:  # a delayed quote must never pass as current
        ts = min(ts, now - cboe.DELAY)
    return Quote(_num(q.get("bidPrice"), 0.0), _num(q.get("askPrice"), 0.0), _num(q.get("bidSize"), 0.0),
                 _num(q.get("askSize"), 0.0), ts, _num(q.get("lastPrice"), 0.0))


class SchwabQuotes:
    live = True

    def __init__(self, app_key: str, app_secret: str, token_path, strike_count: int = 80, client=None):
        self.app_key, self.app_secret = app_key, app_secret
        self.token_path = token_path
        self.strike_count = strike_count
        self._client = client
        self._bucket = TokenBucket(1.5, capacity=8)  # Schwab allows ~120 market data requests a minute

    @classmethod
    def from_settings(cls, strike_count: int = 80) -> "SchwabQuotes | None":
        from autotrader.config import get_settings

        s = get_settings()
        if not (s.schwab_app_key and s.schwab_app_secret):
            return None
        return cls(s.schwab_app_key, s.schwab_app_secret, s.schwab_token_path, strike_count)

    @property
    def client(self):
        if self._client is None:
            from schwab import auth

            try:
                self._client = auth.client_from_token_file(str(self.token_path), self.app_key, self.app_secret)
            except FileNotFoundError as exc:
                raise QuoteError(f"no Schwab token at {self.token_path}; run `autotrader options schwab-login`") \
                    from exc
        return self._client

    def token_expires(self) -> pd.Timestamp | None:
        """When the refresh token dies (7 days after the last browser login)."""
        try:
            created = json.loads(open(self.token_path).read())["creation_timestamp"]
        except (OSError, KeyError, ValueError):
            return None
        return pd.Timestamp(created, unit="s", tz="UTC") + pd.Timedelta(days=TOKEN_DAYS)

    def _get(self, call, *args, **kwargs) -> dict:
        self._bucket.acquire()
        try:
            resp = call(*args, **kwargs)
        except QuoteError:
            raise
        except Exception as exc:  # noqa: BLE001 - token refresh failures, network
            raise QuoteError(f"Schwab: {exc}") from exc
        if resp.status_code != 200:
            raise QuoteError(f"Schwab HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def chain(self, underlying: str, max_dte: int, moneyness: float) -> ChainSnapshot:
        now = pd.Timestamp.now(tz="UTC")
        today = now.tz_convert("America/New_York").date()
        payload = self._get(self.client.get_option_chain, underlying.upper(), strike_count=self.strike_count,
                            include_underlying_quote=True, from_date=today,
                            to_date=today + pd.Timedelta(days=max_dte))
        return parse_schwab_chain(payload, now, max_dte, moneyness, today)

    def _quotes(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(symbols), 200):
            out |= self._get(self.client.get_quotes, symbols[i:i + 200])
        return out

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        if not symbols:
            return {}
        now = pd.Timestamp.now(tz="UTC")
        raw = self._quotes([schwab_symbol(s) for s in symbols])
        out = {}
        for key, v in raw.items():
            occ = key.replace(" ", "")
            if isinstance(v, dict) and "quote" in v:
                out[occ] = _quote(v["quote"], bool(v.get("realtime", True)), now)
        return out

    def spot(self, symbols: list[str]) -> dict[str, Quote]:
        if not symbols:
            return {}
        now = pd.Timestamp.now(tz="UTC")
        raw = self._quotes([s.upper() for s in symbols])
        return {k: _quote(v["quote"], bool(v.get("realtime", True)), now)
                for k, v in raw.items() if isinstance(v, dict) and "quote" in v}


# ----- CBOE (delayed) and the failover between the two -----

class CboeQuotes:
    """CBOE's delayed chain. No live quotes: every open waits while this is the source."""
    live = False

    def chain(self, underlying: str, max_dte: int, moneyness: float) -> ChainSnapshot:
        return cboe.fetch_chain(underlying, max_dte, moneyness)

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {}

    def spot(self, symbols: list[str]) -> dict[str, Quote]:
        return {}


class FailoverQuotes:
    """Schwab first; CBOE when Schwab fails (one alert per outage, re-tried every `retry_sec`)."""

    def __init__(self, primary: QuoteSource, fallback: QuoteSource | None = None, retry_sec: float = 60.0,
                 alert=None):
        self.primary = primary
        self.fallback = fallback or CboeQuotes()
        self.retry_sec = retry_sec
        self._down_since: float | None = None
        self._alert = alert

    @property
    def live(self) -> bool:
        return self._down_since is None and self.primary.live

    def _primary_ok(self) -> bool:
        return self._down_since is None or time.monotonic() - self._down_since >= self.retry_sec

    def _failed(self, exc: Exception) -> None:
        first = self._down_since is None
        self._down_since = time.monotonic()
        log.warning("live quotes failed: %s", exc)
        if first and self._alert is not None:
            self._alert("Live option quotes are down",
                        f"{exc}. Pricing falls back to CBOE's 15-minute-delayed chain and no bot may open a "
                        "position until live quotes return. If the Schwab token expired, run "
                        "`autotrader options schwab-login`.")

    def _recovered(self) -> None:
        if self._down_since is not None:
            log.info("live quotes are back")
        self._down_since = None

    def chain(self, underlying: str, max_dte: int, moneyness: float) -> ChainSnapshot:
        if self._primary_ok():
            try:
                snap = self.primary.chain(underlying, max_dte, moneyness)
                self._recovered()
                return snap
            except Exception as exc:  # noqa: BLE001
                self._failed(exc)
        return self.fallback.chain(underlying, max_dte, moneyness)

    def _try(self, name: str, symbols: list[str]) -> dict[str, Quote]:
        if not symbols or not self._primary_ok():
            return {}
        try:
            out = getattr(self.primary, name)(symbols)
            self._recovered()
            return out
        except Exception as exc:  # noqa: BLE001
            self._failed(exc)
            return {}

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return self._try("quotes", symbols)

    def spot(self, symbols: list[str]) -> dict[str, Quote]:
        return self._try("spot", symbols)


def quote_source_from_settings(strike_count: int = 80, alert=None) -> QuoteSource:
    schwab = SchwabQuotes.from_settings(strike_count)
    if schwab is None:
        log.warning("AT_SCHWAB_APP_KEY/SECRET are not set: options pricing uses CBOE's delayed chain "
                    "and no bot can open positions")
        return CboeQuotes()
    return FailoverQuotes(schwab, CboeQuotes(), alert=alert)


def token_expires(source: QuoteSource | None) -> pd.Timestamp | None:
    """When the Schwab login behind `source` expires (None: no Schwab, or no token yet)."""
    source = getattr(source, "primary", source)
    return source.token_expires() if isinstance(source, SchwabQuotes) else None


def leg_quotes(source: QuoteSource | None, legs) -> dict[str, Quote]:
    """Live quotes for the option legs [(symbol, ratio), ...]; {} without a source or on failure."""
    if source is None:
        return {}
    syms = sorted({s for s, _ in legs if OCC_RE.match(s)})
    try:
        return source.quotes(syms)
    except Exception as exc:  # noqa: BLE001
        log.warning("quotes %s: %s", syms, exc)
        return {}

