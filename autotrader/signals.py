"""On-demand buy/sell scan using stoch_momentum (the original /signals behaviour)."""

import logging

import pandas as pd
from returns.result import Failure, Success

from autotrader.data.intervals import Interval
from autotrader.data.providers.yahoo import YahooProvider
from autotrader.indicators import fast_stochastic, momentum, momentum_pct, slow_stochastic
from autotrader.strategies.builtin.stoch_momentum import StochMomentum

log = logging.getLogger(__name__)

RANGE_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653}


def _entry(symbol: str, df: pd.DataFrame, s: StochMomentum) -> dict:
    p = s.params
    fast = fast_stochastic(df, p.k_period, p.d_period).iloc[-1]
    slow = slow_stochastic(df, p.k_period, p.d_period, p.slow_d_period).iloc[-1]
    return {
        "ticker": symbol,
        "current_price": float(df["close"].iloc[-1]),
        "fast_k": round(float(fast.fast_k), 2),
        "fast_d": round(float(fast.fast_d), 2),
        "slow_k": round(float(slow.slow_k), 2),
        "slow_d": round(float(slow.slow_d), 2),
        "momentum": round(float(momentum(df["close"], p.momentum_period).iloc[-1]), 2),
        "momentum_pct": round(float(momentum_pct(df["close"], p.momentum_period).iloc[-1]), 2),
    }


def get_buy_sell_signals(
    tickers: list[str], data_range: str = "1y", interval: Interval = "1d",
    provider: YahooProvider | None = None,
) -> dict:
    provider = provider or YahooProvider()
    strategy = StochMomentum()
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RANGE_DAYS.get(data_range, 366))
    out: dict[str, list] = {"buy": [], "sell": [], "errors": []}
    for symbol in tickers:
        match provider.fetch_bars(symbol, interval, start=start):
            case Success(df) if len(df) >= strategy.lookback:
                sig = strategy.signal(df)
                if sig:
                    out[sig].append(_entry(symbol, df, strategy))
            case Success(df):
                out["errors"].append({"ticker": symbol, "error": f"only {len(df)} bars"})
            case Failure(exc):
                log.warning("signal fetch failed for %s: %s", symbol, exc)
                out["errors"].append({"ticker": symbol, "error": str(exc)})
    return out
