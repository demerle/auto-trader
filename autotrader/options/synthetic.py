"""Synthetic option chains for backtests (no free historical intraday options data exists).

Built from the real underlying price plus VIX / VIX1D, calibrated against real option
prices (Alpaca 0DTE prints, Feb 2024 - Oct 2026, and CBOE chains): same-day options
trade at an ATM IV (Black-Scholes, calendar time to the 16:00 close) of about 1.7x
VIX1D for SPY (QQQ ~2.2x), not 1x; later expiries sit near VIX. Both are scaled by the
underlying's realized vol relative to SPY's, and a smile lifts both wings (puts more).
Using VIX1D as-is priced 0DTE options at roughly half their real cost (OTM ones at a
fifth or less), which made option buyers look far better here than they are live.
Spreads are ~1.5-2x real ones and fills are at the far side (pessimistic). Good enough to prove a strategy trades and respects the limits; it
cannot prove a strategy profitable. Real recorded chains replace it where we have them.
"""

from datetime import date

import numpy as np
import pandas as pd

from autotrader.data.calendar import schedule
from autotrader.options.helpers import Chain
from autotrader.options.pricing import MIN_T, YEAR_SEC, bs_greeks, bs_price, expiry_close

DAILY_EXPIRIES = {"SPY", "QQQ", "IWM"}


def strike_step(underlying: str, spot: float) -> float:
    if underlying in DAILY_EXPIRIES:
        return 1.0
    return 0.5 if spot < 30 else 1.0 if spot < 150 else 2.5 if spot < 400 else 5.0


def expiries(underlying: str, today: date, max_dte: int = 45, limit: int = 8) -> list[date]:
    sessions = [d.date() for d in schedule(pd.Timestamp(today, tz="UTC"),
                                           pd.Timestamp(today, tz="UTC") + pd.Timedelta(days=max_dte)).index]
    if underlying in DAILY_EXPIRIES:
        out = sessions
    else:  # weeklies: the last session of each week
        out = [d for i, d in enumerate(sessions)
               if i + 1 == len(sessions) or sessions[i + 1].isocalendar()[1] != d.isocalendar()[1]]
    return out[:limit]


ZERO_DTE_PER_VIX1D = 1.7  # median real SPY 0DTE ATM IV / VIX1D, 2024-2026 (1.6-1.9 through the day)


def atm_vol(dte_years: np.ndarray, vix: float, vix1d: float | None, rel: float) -> np.ndarray:
    """ATM IV per contract: 0DTE from VIX1D, later expiries from VIX (single names on a damped rel)."""
    v30 = vix / 100
    v0 = ZERO_DTE_PER_VIX1D * (vix1d / 100 if vix1d else v30 * 0.75)
    same_day = np.asarray(dte_years) < 1 / 365
    return np.where(same_day, rel * v0, rel ** 0.8 * v30)


class SyntheticChains:
    """Per-day contract templates (symbols, strikes, expiries) cached; each tick only reprices."""

    def __init__(self, max_dte: int = 45, max_expiries: int = 6, moneyness: float = 0.10):
        self.max_dte, self.max_expiries, self.moneyness = max_dte, max_expiries, moneyness
        self._templates: dict[tuple[str, date], tuple[pd.DataFrame, np.ndarray, dict]] = {}

    def _template(self, underlying: str, spot: float, today: date):
        key = (underlying, today)
        if key not in self._templates:
            exps = expiries(underlying, today, self.max_dte, self.max_expiries)
            step = strike_step(underlying, spot)
            lo = np.floor(spot * (1 - self.moneyness) / step) * step
            hi = np.ceil(spot * (1 + self.moneyness) / step) * step
            strikes = np.arange(lo, hi + step / 2, step)
            rows = [(f"{underlying}{e:%y%m%d}{r}{round(k * 1000):08d}", e, r, k)
                    for e in exps for k in strikes for r in ("C", "P")]
            df = pd.DataFrame(rows, columns=["symbol", "expiry", "right", "strike"])
            df.insert(1, "underlying", underlying)
            closes = expiry_close(list(df["expiry"])) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
            close_sec = np.array([c.timestamp() for c in closes], dtype=float)
            self._templates[key] = (df, close_sec, {sym: i for i, sym in enumerate(df["symbol"])})
            if len(self._templates) > 64:
                self._templates.pop(next(iter(self._templates)))
        return self._templates[key]

    def __call__(self, underlying: str, spot: float, now: pd.Timestamp, vix: float,
                 vix1d: float | None, rel_vol: float = 1.0) -> Chain:
        today = now.tz_convert("America/New_York").date()
        tpl, close_sec, index = self._template(underlying, spot, today)
        if tpl.empty:
            return Chain(underlying, spot, now, pd.DataFrame(columns=_COLS), today)
        t = np.maximum((close_sec - now.timestamp()) / YEAR_SEC, MIN_T)
        k = tpl["strike"].to_numpy(float)
        is_call = (tpl["right"] == "C").to_numpy()
        atm = atm_vol(t, vix, vix1d, rel_vol)
        z = np.log(k / spot) / (atm * np.sqrt(np.maximum(t, 1 / 365)))
        z = np.clip(z, -4, 4)
        iv = atm * (1 - 0.06 * z + 0.06 * z * z)  # 0DTE smile: ~1.35x ATM 2 sd down, ~1.1x 2 sd up
        mark = np.maximum(bs_price(spot, k, t, iv, is_call), 0.0)
        g = bs_greeks(spot, k, t, iv, is_call)
        # About 1.5-2x the half-spreads measured on real CBOE chains (ETFs ~0.5-1c on cheap
        # contracts and ~1% above that; single names ~1-1.5%): pessimistic, not punitive.
        zero_dte = t < 1 / 365
        if underlying in DAILY_EXPIRIES:
            half = np.maximum(0.01, 0.015 * mark)
        else:
            half = np.maximum(0.015, 0.02 * mark)
        half = half + np.where(zero_dte, 0.005, 0.0)
        live = close_sec > now.timestamp()
        df = tpl.copy()
        bid = np.maximum(mark - half, 0.0)
        df["bid"], df["ask"], df["last"], df["iv"] = bid, mark + half, mark, iv
        for name in ("delta", "gamma", "theta", "vega"):
            df[name] = g[name]
        df["oi"], df["volume"] = np.nan, np.nan
        df["mark"], df["est_bid"], df["est_ask"] = mark, bid, mark + half
        if not live.all():
            df = df[live]
            return Chain(underlying, spot, now, df, today)
        return Chain(underlying, spot, now, df, today, index=index)


_default = SyntheticChains()


def synthetic_chain(underlying: str, spot: float, now: pd.Timestamp, vix: float,
                    vix1d: float | None, rel_vol: float = 1.0) -> Chain:
    return _default(underlying, spot, now, vix, vix1d, rel_vol)


_COLS = ["symbol", "underlying", "expiry", "right", "strike", "bid", "ask", "last", "iv", "delta",
         "gamma", "theta", "vega", "oi", "volume", "mark", "est_bid", "est_ask"]
