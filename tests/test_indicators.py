import numpy as np
import pandas as pd
import pytest

from autotrader import indicators as ind
from tests.reference import legacy_stochastics as legacy


def as_legacy(df: pd.DataFrame) -> dict:
    return {
        "meta": {"gmtoffset": 0},
        "timestamp": [int(t.timestamp()) for t in df.index],
        **{c: df[c].tolist() for c in ["open", "high", "low", "close", "volume"]},
    }


@pytest.mark.parametrize("cut", [60, 150, 399])
def test_stochastics_and_momentum_match_original(bars, cut):
    df = bars.iloc[: cut + 1]
    data = as_legacy(df)
    fast_old = legacy.get_fast_stochastic(data, k_period=21, d_period=5)
    slow_old = legacy.get_slow_stochastic(data, k_period=21, d_period=5, slow_d_period=5)
    mom_old = legacy.get_momentum(data, period=21)

    fast = ind.fast_stochastic(df, 21, 5).iloc[-1]
    slow = ind.slow_stochastic(df, 21, 5, 5).iloc[-1]
    assert fast.fast_k == pytest.approx(fast_old["fast_k"])
    assert fast.fast_d == pytest.approx(fast_old["fast_d"])
    assert slow.slow_k == pytest.approx(slow_old["slow_k"])
    assert slow.slow_d == pytest.approx(slow_old["slow_d"])
    assert ind.momentum(df.close, 21).iloc[-1] == pytest.approx(mom_old["momentum"])
    assert ind.momentum_pct(df.close, 21).iloc[-1] == pytest.approx(mom_old["momentum_pct"])


def test_flat_channel_gives_50():
    df = pd.DataFrame({c: [10.0] * 30 for c in ["open", "high", "low", "close", "volume"]})
    assert ind.fast_stochastic(df, 14, 3).fast_k.iloc[-1] == 50.0


def test_rsi_bounds_and_extremes(bars):
    r = ind.rsi(bars.close, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    rising = pd.Series(np.arange(1.0, 40.0))
    assert ind.rsi(rising, 14).iloc[-1] == 100.0


def test_indicators_are_causal(bars):
    """Values at bar t must not change when later bars are appended."""
    short, full = bars.iloc[:200], bars
    for fn in (lambda d: ind.atr(d, 14), lambda d: ind.rsi(d.close, 14),
               lambda d: ind.bollinger(d.close).upper, lambda d: ind.donchian(d).upper,
               lambda d: ind.zscore(d.close), lambda d: ind.vwap(d)):
        pd.testing.assert_series_equal(fn(short), fn(full).iloc[:200], check_names=False)
