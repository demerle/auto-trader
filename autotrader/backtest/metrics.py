"""Performance metrics, including Probabilistic and Deflated Sharpe ratios
(Bailey & Lopez de Prado), which correct for luck when many variants are tried."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    sd = returns.std(ddof=1)
    if len(returns) < 2 or not sd or math.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float) -> float:
    downside = returns.clip(upper=0)
    dd = math.sqrt((downside**2).mean()) if len(returns) else 0.0
    if not dd:
        return 0.0
    return float(returns.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def _per_period_sr(returns: pd.Series) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd) if len(returns) > 1 and sd else 0.0


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark) given sample length, skew and kurtosis."""
    n = len(returns)
    if n < 3:
        return 0.0
    sr = _per_period_sr(returns)
    skew = float(returns.skew()) if n > 2 else 0.0
    kurt = float(returns.kurt()) + 3 if n > 3 else 3.0
    skew = 0.0 if math.isnan(skew) else skew
    kurt = 3.0 if math.isnan(kurt) else kurt
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return 0.0
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less strategies."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_variance: float) -> float:
    """PSR against the Sharpe you'd expect from the best of n_trials lucky draws."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_variance))


def trade_stats(realized: list[float]) -> dict[str, float]:
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    gross_loss = -sum(losses)
    return {
        "closing_trades": float(len(realized)),
        "win_rate": len(wins) / len(realized) if realized else 0.0,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss else (math.inf if wins else 0.0),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }


def alpha_beta(returns: pd.Series, benchmark: pd.Series, periods_per_year: float) -> tuple[float, float]:
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 3 or joined.iloc[:, 1].var() == 0:
        return 0.0, 0.0
    r, b = joined.iloc[:, 0], joined.iloc[:, 1]
    beta = float(r.cov(b) / b.var())
    alpha = float((r.mean() - beta * b.mean()) * periods_per_year)
    return alpha, beta


def compute_metrics(
    equity: pd.Series,
    periods_per_year: float,
    realized_pnl: list[float] | None = None,
    turnover: float = 0.0,
    benchmark_returns: pd.Series | None = None,
    n_trials: int = 1,
    sr_variance: float = 0.0,
) -> dict[str, float]:
    rets = returns_from_equity(equity)
    years = len(rets) / periods_per_year if periods_per_year else 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else 0.0
    mdd = max_drawdown(equity)
    out = {
        "total_return": total,
        "cagr": float(cagr),
        "volatility": float(rets.std(ddof=1) * math.sqrt(periods_per_year)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe(rets, periods_per_year),
        "sortino": sortino(rets, periods_per_year),
        "max_drawdown": mdd,
        "calmar": float(cagr / -mdd) if mdd < 0 else 0.0,
        "psr": probabilistic_sharpe(rets),
        "dsr": deflated_sharpe(rets, n_trials, sr_variance),
        "n_periods": float(len(rets)),
        "turnover": turnover,
    }
    out |= trade_stats(realized_pnl or [])
    if benchmark_returns is not None:
        out["alpha"], out["beta"] = alpha_beta(rets, benchmark_returns, periods_per_year)
    return out
