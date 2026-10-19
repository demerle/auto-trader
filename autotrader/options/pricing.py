"""Black-Scholes prices and greeks (vectorized), an IV solver, and the nowcast.

CBOE's free chain is ~15 minutes old. The nowcast keeps each contract's delayed IV
and reprices it at the live underlying price and the current time, so marks and
limit prices aren't stale. The market's own pricing quirks are kept by adding the
snapshot's (observed mid - model mid) back on top.
"""

import numpy as np
import pandas as pd

RATE = 0.04  # risk-free rate; small effect on short-dated options
YEAR_SEC = 365.0 * 24 * 3600
MIN_T = 60.0 / YEAR_SEC  # one minute
SQRT_2PI = np.sqrt(2 * np.pi)


def ndtr(x):
    """Standard normal CDF (Abramowitz & Stegun 7.1.26 erf; abs error < 1.5e-7)."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / np.sqrt(2)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI


def _d1d2(s, k, t, iv, r):
    t = np.maximum(t, MIN_T)
    iv = np.maximum(iv, 1e-4)
    vt = iv * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * iv * iv) * t) / vt
    return d1, d1 - vt, t


def bs_price(s, k, t, iv, is_call, r: float = RATE):
    """Price per share. `is_call` is a bool array (or scalar)."""
    s, k, iv = (np.asarray(x, dtype=float) for x in (s, k, iv))
    d1, d2, t = _d1d2(s, k, np.asarray(t, dtype=float), iv, r)
    disc = np.exp(-r * t)
    call = s * ndtr(d1) - k * disc * ndtr(d2)
    put = k * disc * ndtr(-d2) - s * ndtr(-d1)
    return np.where(is_call, call, put)


def bs_greeks(s, k, t, iv, is_call, r: float = RATE) -> dict[str, np.ndarray]:
    """delta, gamma, theta (per calendar day), vega (per 1 vol point)."""
    s, k, iv = (np.asarray(x, dtype=float) for x in (s, k, iv))
    d1, d2, t = _d1d2(s, k, np.asarray(t, dtype=float), iv, r)
    sq = np.sqrt(t)
    pdf = _pdf(d1)
    disc = np.exp(-r * t)
    delta = np.where(is_call, ndtr(d1), ndtr(d1) - 1)
    gamma = pdf / (s * np.maximum(iv, 1e-4) * sq)
    common = -s * pdf * iv / (2 * sq)
    theta = np.where(is_call, common - r * k * disc * ndtr(d2), common + r * k * disc * ndtr(-d2)) / 365
    vega = s * pdf * sq / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def implied_vol(price, s, k, t, is_call, r: float = RATE, lo: float = 1e-3, hi: float = 5.0):
    """Vectorized bisection. NaN where the price is outside the no-arbitrage range."""
    price, s, k, t = (np.asarray(x, dtype=float) for x in (price, s, k, t))
    a, b = np.full(np.broadcast(price, s, k).shape, lo), np.full(np.broadcast(price, s, k).shape, hi)
    pa, pb = bs_price(s, k, t, a, is_call, r), bs_price(s, k, t, b, is_call, r)
    bad = (price < pa - 1e-9) | (price > pb + 1e-9)
    for _ in range(60):
        m = 0.5 * (a + b)
        pm = bs_price(s, k, t, m, is_call, r)
        high = pm > price
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    return np.where(bad, np.nan, 0.5 * (a + b))


def years_to_expiry(expiry_close: pd.Series | pd.Timestamp, now: pd.Timestamp):
    """Years from now to each expiry's 16:00 ET close (floored at one minute)."""
    secs = (pd.to_datetime(expiry_close) - now).dt.total_seconds() if isinstance(expiry_close, pd.Series) \
        else (expiry_close - now).total_seconds()
    return np.maximum(np.asarray(secs, dtype=float) / YEAR_SEC, MIN_T)


def expiry_close(expiry) -> pd.Series:
    """16:00 New York on each expiry date, in UTC."""
    d = pd.to_datetime(pd.Series(expiry))
    return (d + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def nowcast(chain: pd.DataFrame, spot_then: float, spot_now: float, then: pd.Timestamp,
            now: pd.Timestamp, r: float = RATE) -> pd.DataFrame:
    """Reprice a delayed chain at the live spot. Adds mark, est_bid, est_ask and fresh greeks."""
    df = chain.copy()
    if df.empty:
        for c in ("mark", "est_bid", "est_ask"):
            df[c] = pd.Series(dtype=float)
        return df
    is_call = (df["right"] == "C").to_numpy()
    k = df["strike"].to_numpy(float)
    closes = expiry_close(df["expiry"]).reset_index(drop=True)
    t_then = years_to_expiry(closes, then)
    t_now = years_to_expiry(closes, now)
    bid, ask = df["bid"].to_numpy(float), df["ask"].to_numpy(float)
    quoted = (bid > 0) & (ask >= bid)
    last = df["last"].to_numpy(float) if "last" in df else np.zeros(len(df))
    mid_then = np.where(quoted, 0.5 * (bid + ask), np.maximum(last, 0.0))
    iv = df["iv"].to_numpy(float)
    good_iv = np.isfinite(iv) & (iv > 0.01) & (iv < 5)
    iv_fill = np.where(good_iv, iv, 0.3)

    model_then = bs_price(spot_then, k, t_then, iv_fill, is_call, r)
    model_now = bs_price(spot_now, k, t_now, iv_fill, is_call, r)
    ds = spot_now - spot_then
    taylor = mid_then + df["delta"].fillna(0).to_numpy(float) * ds \
        + 0.5 * df["gamma"].fillna(0).to_numpy(float) * ds * ds
    mark = np.where(good_iv, model_now + (mid_then - model_then), taylor)
    intrinsic = np.where(is_call, np.maximum(spot_now - k, 0), np.maximum(k - spot_now, 0))
    mark = np.maximum(mark, intrinsic)
    mark = np.maximum(mark, 0.0)

    stale_min = max((now - then).total_seconds() / 60, 0.0)
    half = np.where(quoted, 0.5 * (ask - bid), np.maximum(0.05 * mark, 0.05))
    half = np.maximum(half, 0.005) * (1 + min(stale_min, 60) / 30)  # staler -> assume wider
    df["mark"] = mark
    df["est_bid"] = np.maximum(mark - half, 0.0)
    df["est_ask"] = mark + half
    g = bs_greeks(spot_now, k, t_now, iv_fill, is_call, r)
    for name, arr in g.items():
        df[name] = np.where(good_iv, arr, df[name].to_numpy(float) if name in df else np.nan)
    df["iv"] = np.where(good_iv, iv, np.nan)
    return df
