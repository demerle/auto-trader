"""The order desk: one Alpaca account, many bots.

Alpaca rejects opposite-side orders working at the same time in one symbol
("potential wash trade") and won't flip long<->short in one order. So the desk
nets all bot intents per symbol into a single account order, crosses opposing
intents internally, never sends an order that crosses the account's net position
through zero (the remainder is retried next tick from the bots' target_qty), and
attributes broker fills back to bots pro rata.
"""

import logging
import math
import secrets
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.backtest.costs import CostModel
from autotrader.broker.base import TERMINAL, Broker, OrderType, TimeInForce
from autotrader.db.models import Fill, Order, OrderAllocation, utcnow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Intent:
    bot_id: int
    symbol: str
    qty: float  # signed whole shares
    reason: dict = field(default_factory=dict, compare=False)


@dataclass
class ExternalOrder:
    symbol: str
    qty: float
    allocations: dict[int, float]  # bot_id -> signed qty


@dataclass
class DeskPlan:
    external: list[ExternalOrder] = field(default_factory=list)
    internal: list[tuple[int, str, float]] = field(default_factory=list)  # (bot, symbol, qty)
    deferred: list[tuple[int, str, float]] = field(default_factory=list)  # retried next tick


def apportion(total: float, weights: Mapping[int, float]) -> dict[int, float]:
    """Split whole-share `total` (>= 0) across keys in proportion to weights (largest remainder)."""
    wsum = sum(weights.values())
    if total <= 0 or wsum <= 0:
        return {k: 0.0 for k in weights}
    raw = {k: total * w / wsum for k, w in weights.items()}
    out = {k: float(math.floor(v + 1e-9)) for k, v in raw.items()}
    leftover = int(round(total - sum(out.values())))
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:leftover]:
        out[k] += 1
    return out


def net_intents(intents: Iterable[Intent], account_position: Mapping[str, float]) -> DeskPlan:
    plan = DeskPlan()
    by_symbol: dict[str, list[Intent]] = defaultdict(list)
    for it in intents:
        if it.qty:
            by_symbol[it.symbol].append(it)

    for symbol, items in sorted(by_symbol.items()):
        buys = {it.bot_id: it.qty for it in items if it.qty > 0}
        sells = {it.bot_id: -it.qty for it in items if it.qty < 0}
        crossed = min(sum(buys.values()), sum(sells.values()))
        remaining: dict[int, float] = defaultdict(float)
        for it in items:
            remaining[it.bot_id] += it.qty
        if crossed > 0:
            for bot, q in apportion(crossed, buys).items():
                if q:
                    plan.internal.append((bot, symbol, q))
                    remaining[bot] -= q
            for bot, q in apportion(crossed, sells).items():
                if q:
                    plan.internal.append((bot, symbol, -q))
                    remaining[bot] += q

        net = sum(remaining.values())
        if not net:
            continue
        held = account_position.get(symbol, 0.0)
        send = net
        if held and math.copysign(1, held + net) != math.copysign(1, held) and held + net != 0:
            send = -held  # close to flat now; the opening remainder is retried next tick
        side_bots = {b: abs(q) for b, q in remaining.items() if q and math.copysign(1, q) == math.copysign(1, net)}
        alloc = {b: math.copysign(q, net) for b, q in apportion(abs(send), side_bots).items() if q}
        for b, q in remaining.items():
            if q and q - alloc.get(b, 0.0):
                plan.deferred.append((b, symbol, q - alloc.get(b, 0.0)))
        plan.external.append(ExternalOrder(symbol, send, alloc))
    return plan


def new_client_order_id(interval: str, symbol: str) -> str:
    return f"at-{interval}-{symbol}-{pd.Timestamp.now(tz='UTC'):%y%m%d%H%M%S}-{secrets.token_hex(3)}"


class Desk:
    def __init__(self, broker: Broker, costs: CostModel = CostModel()):
        self.broker = broker
        self.costs = costs

    def working_symbols(self, session: Session) -> set[str]:
        rows = session.scalars(select(Order.symbol).where(Order.status.not_in(TERMINAL))).all()
        return set(rows)

    def execute(
        self,
        session: Session,
        plan: DeskPlan,
        interval: str,
        prices: Mapping[str, float],
        order_type: OrderType,
        tif: TimeInForce,
        collar_bps: float = 15.0,
        reasons: Mapping[tuple[int, str], dict] | None = None,
    ) -> list[Order]:
        reasons = reasons or {}
        for bot, symbol, qty in plan.internal:
            session.add(Fill(bot_id=bot, symbol=symbol, qty=qty, price=prices[symbol],
                             fees=0.0, internal=True))
        busy = self.working_symbols(session)
        placed = []
        for ext in plan.external:
            if ext.symbol in busy:
                log.info("skip %s: another order is still working", ext.symbol)
                continue
            ref = prices[ext.symbol]
            limit = None
            if order_type == "limit":
                limit = ref * (1 + collar_bps / 1e4) if ext.qty > 0 else ref * (1 - collar_bps / 1e4)
            order = Order(
                client_order_id=new_client_order_id(interval, ext.symbol),
                account=self.broker.account, interval=interval, symbol=ext.symbol, qty=ext.qty,
                order_type=order_type, time_in_force=tif, limit_price=limit, ref_price=ref,
                status="new",
            )
            order.allocations = [
                OrderAllocation(bot_id=b, qty=q, reason=reasons.get((b, ext.symbol), {}))
                for b, q in ext.allocations.items()
            ]
            session.add(order)
            session.flush()
            try:
                bo = self.broker.submit(order.client_order_id, ext.symbol, ext.qty, order_type,
                                        tif, limit)
                order.broker_order_id, order.status = bo.broker_order_id, bo.status
            except Exception as exc:  # noqa: BLE001 - record and continue with other symbols
                log.error("order %s %s %+g rejected: %s", order.client_order_id, ext.symbol,
                          ext.qty, exc)
                order.status, order.done_at = "rejected", utcnow()
            placed.append(order)
        return placed

    def sync(self, session: Session) -> int:
        """Pull fills for every working order and attribute them to bots. Returns new fills."""
        new_fills = 0
        for order in session.scalars(select(Order).where(Order.status.not_in(TERMINAL))).all():
            if order.broker_order_id is None:
                continue
            bo = self.broker.get_order(order.client_order_id)
            dq = bo.filled_qty - order.filled_qty
            if dq > 1e-9 and bo.filled_avg_price is not None:
                notional = bo.filled_avg_price * bo.filled_qty
                price = (notional - order.filled_notional) / dq
                sign = 1.0 if order.qty > 0 else -1.0
                open_allocs = {a.id: abs(a.qty - a.filled_qty) for a in order.allocations}
                shares = apportion(dq, open_allocs)
                for a in order.allocations:
                    q = shares.get(a.id, 0.0) * sign
                    if not q:
                        continue
                    a.filled_qty += q
                    session.add(Fill(order_id=order.id, bot_id=a.bot_id, symbol=order.symbol,
                                     qty=q, price=price, fees=self.costs.fees(q, price)))
                    new_fills += 1
                order.filled_qty, order.filled_notional = bo.filled_qty, notional
            order.status = bo.status
            if bo.terminal and order.done_at is None:
                order.done_at = utcnow()
        return new_fills

    def cancel_working(self, session: Session, interval: str | None = None,
                       wait_sec: float = 10.0) -> int:
        """Cancel working orders and wait (briefly) until the broker confirms them terminal."""
        q = select(Order).where(Order.status.not_in(TERMINAL))
        if interval is not None:
            q = q.where(Order.interval == interval)
        orders = [o for o in session.scalars(q).all() if o.broker_order_id is not None]
        for order in orders:
            self.broker.cancel(order.client_order_id)
        deadline = time.monotonic() + wait_sec
        while True:
            self.sync(session)
            if all(o.status in TERMINAL for o in orders) or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        still = [o.client_order_id for o in orders if o.status not in TERMINAL]
        if still:
            log.warning("orders not confirmed canceled yet: %s", still)
        return len(orders)
