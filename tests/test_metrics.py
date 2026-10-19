import numpy as np
import pandas as pd

from autotrader.backtest import metrics as m


def series(mu, sd=0.01, n=1000, seed=1):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n))


def test_psr_orders_by_skill():
    assert m.probabilistic_sharpe(series(0.002)) > 0.99
    assert m.probabilistic_sharpe(series(-0.002)) < 0.01


def test_deflation_penalizes_many_trials():
    r = series(0.0008)
    one = m.deflated_sharpe(r, n_trials=1, sr_variance=0.002)
    many = m.deflated_sharpe(r, n_trials=500, sr_variance=0.002)
    assert many < one


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 65.0])
    assert m.max_drawdown(eq) == -0.5


def test_trade_stats():
    s = m.trade_stats([10, -5, 20, -5])
    assert s["win_rate"] == 0.5 and s["profit_factor"] == 3.0
