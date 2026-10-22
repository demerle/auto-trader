"""Intraday features for the gold_scar direction model.

One code path for training and for live ticks: 5-minute bars -> per-session arrays (days x 78 RTH slots) ->
ridge features and transformer inputs. Prices are normalised by the day's expected move (VIX / sqrt(252)),
so one model serves calm and wild regimes.
"""

import numpy as np
import pandas as pd

ET = "America/New_York"
TARGETS = ("SPY", "QQQ", "IWM")  # traded
CROSS = ("DIA", "TLT", "HYG", "SMH", "XLF", "XLK", "GLD")  # read only: bonds, credit, semis, banks, tech, gold
SYMBOLS = TARGETS + CROSS
SLOTS = 78  # 5-minute bars in a regular session
VOL_DAYS = 3  # volume profile lookback (sessions; the live feed holds about a week of 5m bars)
LAGS = (1, 3, 6, 12)
OWN = ("r_open", "r_pc", "gap") + tuple(f"r_{n}" for n in LAGS) + (
    "vwapd", "pos", "rng", "rv_in", "up_frac", "relvol")


def ridge_names(u: str) -> tuple[str, ...]:
    return OWN + tuple(f"x_{s}" for s in SYMBOLS if s != u)


VIX = "^VIX"  # its prior close sets the day's expected move
RIDGE_DIM = len(ridge_names(TARGETS[0]))
BAR_DIM = 3 * len(SYMBOLS)  # per symbol: bar return, bar range, relative volume
STATIC_DIM = 3  # log VIX, weekday, monthly opex


def sessions(df: pd.DataFrame, days: pd.DatetimeIndex | None = None) -> tuple[pd.DatetimeIndex, dict]:
    """5m bars (index = bar start, UTC; open/high/low/close/volume) -> session dates and arrays (days x 78).
    Slots with no print carry the last price forward (first slots take the first print); missing days are NaN."""
    a = {k: np.full((0 if days is None else len(days), SLOTS), np.nan) for k in "ohlcv"}
    if df is None or df.empty:
        return (days if days is not None else pd.DatetimeIndex([])), a
    et = df.index.tz_convert(ET)
    m = et.hour * 60 + et.minute
    keep = (m >= 570) & (m < 960)
    df, et, m = df[keep], et[keep], m[keep]
    date = pd.DatetimeIndex(et.date)
    if days is None:
        days = date.unique().sort_values()
    row = days.get_indexer(date)
    ok = row >= 0
    slot = ((m - 570) // 5).to_numpy()[ok]
    row = row[ok]
    for k, col in zip("ohlcv", ("open", "high", "low", "close", "volume")):
        x = np.full((len(days), SLOTS), np.nan)
        x[row, slot] = df[col].to_numpy(float)[ok]
        a[k] = x
    for k in "ohlc":  # carry within each day only
        a[k] = pd.DataFrame(a[k]).T.ffill().bfill().T.to_numpy()
    a["v"] = np.where(np.isnan(a["c"]), np.nan, np.nan_to_num(a["v"]))
    return days, a


def _prev_close(a: dict) -> np.ndarray:
    return np.r_[np.nan, a["c"][:-1, -1]]


def _vol_avg(v: np.ndarray) -> np.ndarray:
    """Mean volume per slot over the previous VOL_DAYS sessions (needs at least 2)."""
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        past = v[max(i - VOL_DAYS, 0):i]
        past = past[~np.isnan(past).all(1)] if len(past) else past
        if len(past) >= 2:
            out[i] = past.mean(0)
    return out


def bar_raw(A: dict) -> np.ndarray:
    """days x 78 x BAR_DIM: per symbol, each bar's log return (the first vs yesterday's close), its log
    high-low range, and log volume vs the slot's recent average. Returns and ranges are scaled in `window`."""
    out = []
    for s in SYMBOLS:
        a = A[s]
        c, h, lo, v = a["c"], a["h"], a["l"], a["v"]
        prev = np.concatenate([_prev_close(a)[:, None], c[:, :-1]], 1)
        avg = _vol_avg(v)
        out += [np.log(c / prev), np.log(h / lo), np.log((np.nan_to_num(v) + 1) / (np.nan_to_num(avg) + 1))]
    return np.stack(out, -1)


_SCALED = np.array([i % 3 < 2 for i in range(BAR_DIM)])
_LO = np.where(_SCALED, np.tile([-5.0, 0.0, -4.0], len(SYMBOLS)), -4.0)
_HI = np.where(_SCALED, 5.0, 4.0)


def window(raw: np.ndarray, rows: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Transformer input for each day in `rows`: yesterday's and today's bars (156 steps), returns and ranges
    in units of that day's expected move."""
    rows = np.asarray(rows)
    x = np.concatenate([raw[rows - 1], raw[rows]], 1)  # n x 156 x BAR_DIM
    scale = np.where(_SCALED, 1.0, 0.0)[None, None, :] / sig[:, None, None] + np.where(_SCALED, 0.0, 1.0)
    return np.clip(np.nan_to_num(x * scale), _LO, _HI).astype(np.float32)


def prior_closes(df: pd.DataFrame | None, days: pd.DatetimeIndex) -> np.ndarray:
    """For each session date, the last close strictly before that date (ET); NaN if none."""
    if df is None or df.empty:
        return np.full(len(days), np.nan)
    d = pd.DatetimeIndex(df.index.tz_convert(ET).date)
    last = df["close"].groupby(d).last().sort_index()
    i = last.index.searchsorted(days, side="left") - 1
    return np.where(i >= 0, last.to_numpy(float)[np.maximum(i, 0)], np.nan)


def static_inputs(days: pd.DatetimeIndex, vix: np.ndarray) -> np.ndarray:
    """days x STATIC_DIM from VIX's prior close."""
    opex = np.array([d.weekday() == 4 and 15 <= d.day <= 21 for d in days], float)
    return np.nan_to_num(np.stack([np.log(vix / 18.0), days.dayofweek.to_numpy() / 4.0, opex], 1)).astype(
        np.float32)


def ridge_features(A: dict, u: str, k: int, sig: np.ndarray) -> np.ndarray:
    """days x RIDGE_DIM, as of the close of bar k (decision at 9:30 + 5(k+1) ET)."""
    a = A[u]
    c, o, h, lo, v = a["c"], a["o"], a["h"], a["l"], a["v"]
    pc = _prev_close(a)
    f = [np.log(c[:, k] / o[:, 0]), np.log(c[:, k] / pc), np.log(o[:, 0] / pc)]
    f += [np.log(c[:, k] / c[:, max(k - n, 0)]) for n in LAGS]
    hi, low = h[:, :k + 1].max(1), lo[:, :k + 1].min(1)
    vv = np.nan_to_num(v[:, :k + 1])
    tp = (h[:, :k + 1] + lo[:, :k + 1] + c[:, :k + 1]) / 3
    vwap = (vv * tp).sum(1) / np.maximum(vv.sum(1), 1)
    f += [np.log(c[:, k] / np.where(vv.sum(1) > 0, vwap, c[:, k]))]
    f = [x / sig for x in f]
    f += [(c[:, k] - low) / np.maximum(hi - low, 1e-9), np.log(hi / low) / sig]
    lr = np.diff(np.log(c[:, :k + 1]), axis=1)
    f += [np.sqrt((lr ** 2).sum(1) * SLOTS / max(k, 1)) / sig if k else np.zeros(len(c)),
          (lr > 0).mean(1) if k else np.full(len(c), 0.5)]
    avg = _vol_avg(v)
    f += [vv.sum(1) / np.maximum(np.nan_to_num(avg[:, :k + 1]).sum(1), 1)]
    for s in SYMBOLS:
        if s != u:
            f += [np.log(A[s]["c"][:, k] / A[s]["o"][:, 0]) / sig]
    return np.nan_to_num(np.stack(f, 1), posinf=0.0, neginf=0.0)


def expected_move(vix: float | np.ndarray | None) -> np.ndarray:
    """VIX -> one day's expected move (fraction); 18 when unknown."""
    v = np.asarray(18.0 if vix is None else vix, float)
    return np.where(np.isfinite(v) & (v > 0), v, 18.0) / 100 / np.sqrt(252)
