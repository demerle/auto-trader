"""Turn target weights into share quantities. Shared by the backtester and the live engine."""

import math
from collections.abc import Mapping


def clean_targets(targets: Mapping[str, float], held: set[str], max_gross: float,
                  tradable: set[str]) -> dict[str, float]:
    """Drop unknown/NaN symbols, scale gross exposure down to max_gross, zero out dropped holdings."""
    out = {s: float(w) for s, w in targets.items() if s in tradable and w == w}
    gross = sum(abs(w) for w in out.values())
    if gross > max_gross:
        out = {s: w * max_gross / gross for s, w in out.items()}
    for s in held:
        out.setdefault(s, 0.0)
    return out


def target_quantities(weights: Mapping[str, float], equity: float, prices: Mapping[str, float],
                      fractional: bool = False) -> dict[str, float]:
    out = {}
    for s, w in weights.items():
        q = w * equity / prices[s]
        # The epsilon keeps an unchanged weight from losing a share to float error.
        out[s] = q if fractional else float(math.trunc(q + math.copysign(1e-6, q)))
    return out


def order_deltas(target_qty: Mapping[str, float], held_qty: Mapping[str, float],
                 prices: Mapping[str, float], min_trade_value: float = 1.0) -> dict[str, float]:
    """Signed share deltas needed to reach target_qty, skipping dust."""
    out = {}
    for s in set(target_qty) | set(held_qty):
        delta = target_qty.get(s, 0.0) - held_qty.get(s, 0.0)
        if abs(delta * prices[s]) >= min_trade_value:
            out[s] = delta
    return out
