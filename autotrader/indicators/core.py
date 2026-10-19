import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Close_t - Close_{t-period}."""
    return close - close.shift(period)


def momentum_pct(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change in percent; 0 where the base close is 0."""
    base = close.shift(period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (close - base) / base * 100
    return pct.mask(base == 0, 0.0)


def fast_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Fast %K and %D. A flat channel (high == low) gives %K = 50."""
    hh = df["high"].rolling(k_period, min_periods=k_period).max()
    ll = df["low"].rolling(k_period, min_periods=k_period).min()
    denom = hh - ll
    k = ((df["close"] - ll) / denom.replace(0, np.nan) * 100).mask(denom == 0, 50.0)
    return pd.DataFrame({"fast_k": k, "fast_d": sma(k, d_period)})


def slow_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slow_d_period: int = 3
) -> pd.DataFrame:
    """Slow %K = Fast %D; Slow %D = SMA(Slow %K)."""
    slow_k = fast_stochastic(df, k_period, d_period)["fast_d"]
    return pd.DataFrame({"slow_k": slow_k, "slow_d": sma(slow_k, slow_d_period)})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.mask((loss == 0) & gain.notna(), 100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1, skipna=True)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd,
                         "width": (2 * k * sd) / mid})


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Channel of the *previous* n bars (excludes the current bar, for breakout tests)."""
    return pd.DataFrame({
        "upper": df["high"].shift(1).rolling(n, min_periods=n).max(),
        "lower": df["low"].shift(1).rolling(n, min_periods=n).min(),
    })


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, reset each New York trading date."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.tz_convert("America/New_York").date
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return (pv / vol.replace(0, np.nan)).fillna(typical)


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - mean) / sd.replace(0, np.nan)
