"""Acceptance checks every strategy must pass before it may trade."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.strategies.base import Strategy


def check_causal_features(strategy: Strategy, bars: Mapping[str, pd.DataFrame], cuts: int = 5,
                          seed: int = 0) -> list[str]:
    """Features at row t must be identical whether or not rows after t exist."""
    problems = []
    rng = np.random.default_rng(seed)
    for symbol, df in bars.items():
        full = strategy.features(df)
        if full is None:
            return []
        if not full.index.equals(df.index):
            problems.append(f"{symbol}: features index must equal the bar index")
            continue
        for cut in sorted(rng.integers(len(df) // 4, len(df), cuts)):
            part = strategy.features(df.iloc[: cut + 1])
            a, b = full.iloc[: cut + 1], part
            if not np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True,
                               rtol=1e-9, atol=1e-12):
                problems.append(f"{symbol}: features at/before {df.index[cut]} change when "
                                "later bars are added (lookahead)")
                break
    return problems


def check_no_lookahead(strategy: Strategy, bars: Mapping[str, pd.DataFrame],
                       cut_fraction: float = 0.6) -> list[str]:
    """A backtest truncated at T must match the full backtest up to T."""
    timeline = sorted(set().union(*(df.index for df in bars.values())))
    cut = timeline[int(len(timeline) * cut_fraction)]
    cfg = BacktestConfig(fractional=True)
    full = run_backtest(strategy, bars, cfg)
    part = run_backtest(strategy, {s: df[df.index <= cut] for s, df in bars.items()}, cfg)
    head = full.equity[full.equity.index <= cut]
    if len(head) != len(part.equity) or not np.allclose(head.to_numpy(), part.equity.to_numpy()):
        return [f"equity before {cut} depends on bars after it (lookahead)"]
    if [t for t in full.trades if t.ts <= cut] != part.trades:
        return [f"trades before {cut} depend on bars after it (lookahead)"]
    return []


def validate(strategy: Strategy, bars: Mapping[str, pd.DataFrame]) -> list[str]:
    return check_causal_features(strategy, bars) + check_no_lookahead(strategy, bars)
