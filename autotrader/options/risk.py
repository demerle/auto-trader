"""Worst-case analysis of option structures (defined risk only).

The value of a structure at its earliest expiry is piecewise linear in the
underlying price, so its minimum over all prices is found at S=0, at each strike,
or as S -> infinity (unbounded if the net call position is short). Legs expiring
later are valued at intrinsic, a lower bound for long legs; a short leg that
outlives the long legs could lose more, so such structures are rejected.
"""

import math

from autotrader.options.occ import MULTIPLIER, is_option, parse


class RiskError(ValueError):
    pass


def payoff(legs: list[tuple[str, float]], s: float) -> float:
    """Dollar value at the earliest expiry if the underlying is at s (legs: (symbol, contracts))."""
    total = 0.0
    for sym, q in legs:
        c = parse(sym)
        intrinsic = max(s - c.strike, 0.0) if c.right == "C" else max(c.strike - s, 0.0)
        total += q * intrinsic * MULTIPLIER
    return total


def worst_value(legs: list[tuple[str, float]]) -> float:
    """Minimum dollar value at expiry over every underlying price (-inf if undefined risk)."""
    legs = [(s, q) for s, q in legs if abs(q) > 1e-9]
    if not legs:
        return 0.0
    contracts = [parse(s) for s, _ in legs]
    if not all(is_option(s) for s, _ in legs):
        raise RiskError("only option legs are allowed")
    if len({c.underlying for c in contracts}) > 1:
        raise RiskError("all legs of a structure must be on one underlying")
    first = min(c.expiry for c in contracts)
    longest_long = max((c.expiry for c, (_, q) in zip(contracts, legs) if q > 0), default=None)
    for c, (_, q) in zip(contracts, legs):
        if q < 0 and c.expiry > first and (longest_long is None or c.expiry > longest_long):
            raise RiskError("a short leg may not expire after every long leg")
    net_calls = sum(q for c, (_, q) in zip(contracts, legs) if c.right == "C")
    if net_calls < -1e-9:
        return -math.inf  # short calls not covered: unlimited loss as the price rises
    points = [0.0] + sorted({c.strike for c in contracts})
    return min(payoff(legs, p) for p in points)


def structure_check(legs: list[tuple[str, int]], qty: int, net_price: float, equity: float,
                    max_risk_pct: float, max_legs: int, max_contracts: int,
                    universe: tuple[str, ...]) -> float:
    """Validate a new structure; returns its max loss in dollars. Raises RiskError."""
    if not legs:
        raise RiskError("a structure needs at least one leg")
    if len(legs) > max_legs:
        raise RiskError(f"{len(legs)} legs > {max_legs}")
    if qty <= 0:
        raise RiskError("qty must be >= 1")
    if len({s for s, _ in legs}) != len(legs):
        raise RiskError("duplicate legs")
    for s, r in legs:
        if not is_option(s):
            raise RiskError(f"{s} is not an OCC option symbol")
        if parse(s).underlying not in universe:
            raise RiskError(f"{parse(s).underlying} is not in the options universe")
        if abs(r) * qty > max_contracts:
            raise RiskError(f"{s}: {abs(r) * qty} contracts > {max_contracts} per order")
    worst = worst_value([(s, r * qty) for s, r in legs])
    if worst == -math.inf:
        raise RiskError("undefined risk (uncovered short calls)")
    cost = net_price * qty * MULTIPLIER  # + paid, - received
    max_loss = cost - worst
    if max_loss > max_risk_pct * equity + 1e-6:
        raise RiskError(f"max loss ${max_loss:,.0f} > {max_risk_pct:.0%} of equity ${equity:,.0f}")
    return max(max_loss, 0.0)


def worst_equity(cash: float, structures: list[list[tuple[str, float]]]) -> float:
    """Cash plus every structure at its own worst case (conservative: worst cases don't offset)."""
    return cash + sum(worst_value(legs) for legs in structures)
