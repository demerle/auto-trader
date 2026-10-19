"""Synthetic option chains for backtests (no free historical intraday options data exists).

Built from the real underlying price plus VIX / VIX1D. The ATM IV is VIX1D for
same-day expiries blending into VIX at 30 days, scaled by the underlying's realized
vol relative to SPY's; skew lifts downside strikes. Spreads are ~1.5-2x real ones
and fills are at the far side (pessimistic). Good enough to prove a strategy trades and respects the limits; it
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


def atm_vol(dte_years: np.ndarray, vix: float, vix1d: float | None, rel: float) -> np.ndarray:
    v30 = vix / 100
    v0 = (vix1d / 100) if vix1d else v30
    w = np.clip(dte_years / (30 / 365), 0, 1)
    return rel * (v0 * (1 - w) + v30 * w)


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
        iv = np.maximum(atm * (1 - 0.12 * z + 0.03 * z * z), 0.4 * atm)
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
