"""The walk-forward gate a strategy must pass before it gets a paper slot.

The benchmark is equal-weight buy-and-hold of the strategy's *own* universe over
the same out-of-sample days. A hand-picked universe (e.g. today's mega caps)
flatters any long strategy through survivorship bias; comparing against holding
that same universe cancels most of that out, leaving the strategy's timing skill.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from autotrader.backtest.engine import run_backtest
from autotrader.backtest.metrics import sharpe
from autotrader.backtest.walkforward import walk_forward
from autotrader.data.intervals import PERIODS_PER_YEAR
from autotrader.evaluator.policy import GateRules
from autotrader.strategies.base import Strategy
from autotrader.strategies.validation import validate


@dataclass
class GateResult:
    strategy: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    holdout: dict[str, float] = field(default_factory=dict)


def universe_benchmark(bars: Mapping[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """Daily-rebalanced equal-weight returns of the universe, aligned to `index`."""
    rets = pd.concat({s: df["close"].pct_change() for s, df in bars.items()}, axis=1)
    return rets.mean(axis=1).reindex(index).fillna(0.0)


def _daily(series: pd.Series) -> pd.Series:
    """Collapse an intraday equity curve to session-date closes."""
    days = series.index.tz_convert("America/New_York").normalize()
    return series.groupby(days).last()


def split_holdout(bars: Mapping[str, pd.DataFrame], holdout_start: pd.Timestamp | None
                  ) -> dict[str, pd.DataFrame]:
    """The part of history an agent may see: everything before the sealed holdout."""
    if holdout_start is None:
        return dict(bars)
    return {s: df[df.index < holdout_start] for s, df in bars.items()}


def run_gate(strategy_cls: type[Strategy], bars: Mapping[str, pd.DataFrame], rules: GateRules,
             grid: Mapping[str, Sequence] | None = None, params: Mapping | None = None,
             holdout_start: pd.Timestamp | None = None,
             prior_trial_sharpes: Sequence[float] = ()) -> GateResult:
    """Walk-forward on the history before `holdout_start`, then (if given) one plain
    backtest with the last fold's params over the sealed holdout."""
    key = f"{strategy_cls.name}@{strategy_cls.version}"
    reasons: list[str] = []

    problems = validate(strategy_cls(params), bars)
    if problems:
        return GateResult(key, False, problems)

    seen = split_holdout(bars, holdout_start)
    n = len(set().union(*(df.index for df in seen.values())))
    train, test = max(int(n * 0.4), 50), max(int(n * 0.15), 20)
    wf = walk_forward(strategy_cls, seen, grid, train_bars=train, test_bars=test, base_params=params,
                      prior_trial_sharpes=prior_trial_sharpes)
    if len(wf.oos_equity) < 3:
        return GateResult(key, False, ["not enough history for a walk-forward"])

    m = dict(wf.metrics)
    oos = wf.oos_equity
    strat_rets = oos.pct_change().dropna()
    bench = universe_benchmark(bars, strat_rets.index)
    ppy = PERIODS_PER_YEAR[strategy_cls.interval]
    m["benchmark_sharpe"] = sharpe(bench, ppy)
    m["excess_sharpe"] = m["sharpe"] - m["benchmark_sharpe"]
    daily = _daily(oos).pct_change().dropna()
    m["daily_mean"] = float(daily.mean()) if len(daily) else 0.0
    m["daily_std"] = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    m["folds"] = float(len(wf.folds))
    m["n_trials"] = float(wf.n_trials)

    if m["sharpe"] < rules.min_oos_sharpe:
        reasons.append(f"OOS Sharpe {m['sharpe']:.2f} < {rules.min_oos_sharpe}")
    if m["dsr"] < rules.min_dsr:
        reasons.append(f"deflated Sharpe {m['dsr']:.2f} < {rules.min_dsr}")
    if m["max_drawdown"] < -rules.max_drawdown:
        reasons.append(f"max drawdown {m['max_drawdown']:.1%} beyond -{rules.max_drawdown:.0%}")
    min_trades = rules.trades_for(strategy_cls.interval)
    if m["closing_trades"] < min_trades:
        reasons.append(f"only {m['closing_trades']:.0f} OOS trades < {min_trades}")
    if m["excess_sharpe"] < rules.min_excess_sharpe:
        reasons.append(f"Sharpe {m['sharpe']:.2f} doesn't beat holding its universe "
                       f"({m['benchmark_sharpe']:.2f})")

    holdout: dict[str, float] = {}
    if holdout_start is not None:
        holdout = _holdout(strategy_cls, bars, holdout_start, wf.folds[-1].best_params, ppy)
        if holdout["n_periods"] < 5:
            reasons.append("holdout too short to judge")
        elif holdout["sharpe"] <= rules.holdout_min_sharpe:
            reasons.append(f"holdout Sharpe {holdout['sharpe']:.2f} <= {rules.holdout_min_sharpe}")
        elif holdout["excess_sharpe"] < rules.holdout_min_excess_sharpe:
            reasons.append(f"holdout Sharpe {holdout['sharpe']:.2f} trails holding its universe "
                           f"({holdout['benchmark_sharpe']:.2f})")
    return GateResult(key, not reasons, reasons, m, holdout)


def _holdout(strategy_cls: type[Strategy], bars: Mapping[str, pd.DataFrame],
             holdout_start: pd.Timestamp, params: Mapping, ppy: float) -> dict[str, float]:
    res = run_backtest(strategy_cls(params), bars, start=holdout_start)
    h = res.metrics()
    rets = res.equity.pct_change().dropna()
    h["benchmark_sharpe"] = sharpe(universe_benchmark(bars, rets.index), ppy)
    h["excess_sharpe"] = h["sharpe"] - h["benchmark_sharpe"]
    h["start"] = float(holdout_start.timestamp())
    return h
