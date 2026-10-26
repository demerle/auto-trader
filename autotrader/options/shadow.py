"""The shadow ledger: what each options fill would have cost with real money.

Alpaca's paper engine fills at real prices, but optimistically: always in full, never behind
anyone in the queue, and multi-leg orders at a net price no exchange guaranteed. For every
paper fill the desk records the real-time NBBO (Schwab) and a pessimistic price:

- buys at the ask, sells at the bid (never better than the paper fill);
- each displayed size's worth beyond the first costs one more tick;
- multi-leg orders are legged in, each leg at its own touch.

The quote is the one at submit time when the fill came within `fill_latency_sec` of the
order (it was marketable), else the first one seen after the fill (it rested).
A bot's shadow return is its paper return minus the extra this cost; going live is judged on it.
"""

import math
from collections.abc import Mapping

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autotrader.db.models import Fill
from autotrader.options.occ import MULTIPLIER


def tick(price: float, symbol: str) -> float:
    """Option price increment: pennies below $3 (and always for SPY/QQQ/IWM), nickels above."""
    return 0.05 if abs(price) >= 3 and not symbol.startswith(("SPY", "QQQ", "IWM")) else 0.01


def touch_price(symbol: str, qty: float, quote: Mapping) -> float:
    """The real-money price of trading `qty` contracts (signed) against `quote`."""
    buy = qty > 0
    px = float(quote["ask"] if buy else quote["bid"])
    size = float(quote.get("ask_size" if buy else "bid_size") or 0)
    lots = math.ceil(abs(qty) / size) if size > 0 else 1
    worse = tick(px, symbol) * (lots - 1)
    return round(px + worse, 4) if buy else round(max(px - worse, 0.0), 4)


def shadow_prices(legs: list[tuple[str, float]], prices: Mapping[str, float],
                  quotes: Mapping[str, Mapping]) -> dict[str, float] | None:
    """Per-leg real-money prices for one fill increment, or None if a leg has no quote.

    legs: (symbol, signed contracts filled). The paper fill stands when it is worse than
    legging in at the touch (whole order compared, so a net mleg price is never split wrongly).
    """
    if not legs or any(s not in quotes for s, _ in legs):
        return None
    touch = {s: touch_price(s, q, quotes[s]) for s, q in legs}
    cost_touch = sum(q * touch[s] for s, q in legs)
    cost_paper = sum(q * prices[s] for s, q in legs)
    return touch if cost_touch > cost_paper + 1e-9 else {s: float(prices[s]) for s, _ in legs}


def shadow_cost(session: Session, bot_id: int, since=None, until=None) -> float:
    """Dollars real money would have paid beyond the paper fills (>= 0 by construction)."""
    q = select(func.coalesce(func.sum((Fill.shadow_price - Fill.price) * Fill.qty), 0.0)) \
        .where(Fill.bot_id == bot_id, Fill.shadow_price.is_not(None))
    if since is not None:
        q = q.where(Fill.filled_at >= pd.Timestamp(since).to_pydatetime())
    if until is not None:
        q = q.where(Fill.filled_at < pd.Timestamp(until).to_pydatetime())
    return float(session.scalar(q) or 0.0) * MULTIPLIER


def coverage(session: Session, bot_id: int, since=None) -> tuple[int, int]:
    """(fills with a shadow price, all option fills) since `since`: unpriced fills weaken the gate."""
    base = select(func.count()).select_from(Fill).where(Fill.bot_id == bot_id, Fill.internal.is_(False),
                                                        Fill.order_id.is_not(None))
    if since is not None:
        base = base.where(Fill.filled_at >= pd.Timestamp(since).to_pydatetime())
    total = session.scalar(base) or 0
    priced = session.scalar(base.where(Fill.shadow_price.is_not(None))) or 0
    return int(priced), int(total)
