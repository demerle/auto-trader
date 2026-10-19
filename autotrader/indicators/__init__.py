"""Vectorized technical indicators. Every function returns a Series aligned to its input."""

from autotrader.indicators.core import (
    atr,
    bollinger,
    donchian,
    ema,
    fast_stochastic,
    momentum,
    momentum_pct,
    rsi,
    slow_stochastic,
    sma,
    true_range,
    vwap,
    zscore,
)

__all__ = [
    "atr", "bollinger", "donchian", "ema", "fast_stochastic", "momentum", "momentum_pct",
    "rsi", "slow_stochastic", "sma", "true_range", "vwap", "zscore",
]
