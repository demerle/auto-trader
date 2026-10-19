"""Rolling walk-forward: tune params on each train window, score only on the
following unseen test window. The stitched test-window equity is the
out-of-sample record that gates promotion to paper trading."""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.backtest.costs import CostModel
from autotrader.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from autotrader.backtest.metrics import compute_metrics
from autotrader.data.intervals import PERIODS_PER_YEAR
from autotrader.strategies.base import Strategy


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_score: float
    test: BacktestResult


@dataclass
class WalkForwardResult:
    strategy: str
    folds: list[Fold]
    oos_equity: pd.Series
    n_trials: int
    trial_sharpes: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def param_grid(grid: Mapping[str, Sequence]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def walk_forward(
    strategy_cls: type[Strategy],
    bars: Mapping[str, pd.DataFrame],
    grid: Mapping[str, Sequence] | None = None,
    train_bars: int = 504,
    test_bars: int = 126,
    base_params: Mapping | None = None,
    config: BacktestConfig = BacktestConfig(),
    costs: CostModel = CostModel(),
    benchmark_returns: pd.Series | None = None,
    prior_trial_sharpes: Sequence[float] = (),
) -> WalkForwardResult:
    """`prior_trial_sharpes`: per-period Sharpes of variants tried before this run (e.g. by
    an agent), so the deflated Sharpe accounts for the whole search, not just this grid."""
    timeline = pd.DatetimeIndex(sorted(set().union(*(df.index for df in bars.values()))))
    combos = [dict(base_params or {}) | c for c in param_grid(grid or {})]
    ppy = PERIODS_PER_YEAR[strategy_cls.interval]
    folds: list[Fold] = []
    trial_sharpes: list[float] = []
    pieces: list[pd.Series] = []
    level = config.initial_cash

    start = 0
    while start + train_bars + 1 < len(timeline):
        tr0, tr1 = timeline[start], timeline[start + train_bars - 1]
        te0 = timeline[start + train_bars]
        te1 = timeline[min(start + train_bars + test_bars, len(timeline)) - 1]

        best, best_score = combos[0], -np.inf
        for params in combos:
            res = run_backtest(strategy_cls(params), bars, config, costs, start=tr0, end=tr1)
            score = res.metrics()["sharpe"]
            trial_sharpes.append(score / np.sqrt(ppy))  # per-period SR for deflation
            if score > best_score:
                best, best_score = params, score

        test = run_backtest(strategy_cls(best), bars, config, costs, start=te0, end=te1)
        folds.append(Fold(tr0, tr1, te0, te1, best, float(best_score), test))
        if len(test.equity):
            scaled = test.equity / test.equity.iloc[0] * level
            pieces.append(scaled if not pieces else scaled.iloc[1:])
            level = float(scaled.iloc[-1])
        start += test_bars

    oos = pd.concat(pieces) if pieces else pd.Series(dtype="float64", name="equity")
    realized = [t.realized_pnl for f in folds for t in f.test.trades if t.realized_pnl is not None]
    searched = trial_sharpes + [float(x) for x in prior_trial_sharpes if np.isfinite(x)]
    n_trials = len(combos) + len(prior_trial_sharpes)
    sr_var = float(np.var(searched, ddof=1)) if len(searched) > 1 else 0.0
    metrics = compute_metrics(oos, ppy, realized, 0.0, benchmark_returns, n_trials, sr_var) \
        if len(oos) > 1 else {}
    return WalkForwardResult(strategy_cls.name, folds, oos, n_trials, trial_sharpes, metrics)
