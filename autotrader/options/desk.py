"""The options desk: one order per structure change, priced at the real-time NBBO mid and chased.

Every order is re-quoted live (Schwab) just before it is sent: the limit starts at the NBBO mid and
moves toward the far side each chase step. With no fresh real-time quote an opening order waits
(execution.require_live_quotes); a closing one is priced off the chain instead. Each fill records the
NBBO and its real-money (shadow) price; see shadow.py.

Rules that keep the shared Alpaca account valid:
- no two bots (or two tags) may hold opposite positions in the same contract, so every
  account-level order is the same open/close as the bot's own (Alpaca rejects uncovered
  shorts, and its coverage check is per order);
- never a buy and a sell working on the same contract at once (Alpaca rejects potential wash trades);
- multi-leg orders use ratios in lowest terms, as Alpaca requires.
"""

import logging
import math
import secrets
from dataclasses import dataclass
from functools import reduce

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.broker.base import TERMINAL
from autotrader.db.models import Bot, Fill, Order, OrderAllocation, utcnow
from autotrader.options.config import GauntletConfig
from autotrader.options.helpers import Chain
from autotrader.options.ledger import account_positions
from autotrader.options.occ import parse
from autotrader.options.planner import Change
from autotrader.options.quotes import Quote, QuoteSource, leg_quotes
from autotrader.options.shadow import shadow_prices

log = logging.getLogger(__name__)
INTERVAL = "opt"


class Deferred(Exception):
    """Can't place this now (busy contract); the planner will ask again next tick."""


class Conflict(Exception):
    """Opening this would put the bot opposite another holder of the same contract."""


@dataclass
class Placement:
    order: Order | None
    status: str  # placed / deferred / conflict / rejected
    detail: str = ""


def _coid(tag: str) -> str:
    safe = "".join(ch for ch in tag if ch.isalnum() or ch == "-")[:16]
    return f"op-{safe}-{pd.Timestamp.now(tz='UTC'):%y%m%d%H%M%S}-{secrets.token_hex(3)}"


def normalize(legs: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
    """(symbol, contracts) -> (symbol, ratio in lowest terms), units."""
    g = reduce(math.gcd, [abs(int(q)) for _, q in legs])
    return [(s, int(q) // g) for s, q in legs], g


class OptionsDesk:
    def __init__(self, broker, cfg: GauntletConfig, quotes: QuoteSource | None = None):
        self.broker = broker
        self.cfg = cfg
        self.quotes = quotes

    @property
    def account(self) -> str:
        return self.broker.account

    def working(self, session: Session, bot_id: int | None = None) -> list[Order]:
        q = select(Order).where(Order.account == self.account, Order.status.not_in(TERMINAL))
        orders = session.scalars(q).all()
        if bot_id is not None:
            orders = [o for o in orders if any(a.bot_id == bot_id for a in o.allocations)]
        return orders

    def busy_sides(self, session: Session) -> dict[str, set[int]]:
        """Contract -> the sides (+1 buy, -1 sell) of orders still working on it."""
        out: dict[str, set[int]] = {}
        for o in self.working(session):
            for leg in o.legs or []:
                out.setdefault(leg["symbol"], set()).add(1 if leg["ratio"] > 0 else -1)
        return out

    def busy_tags(self, session: Session, bot_id: int) -> set[str]:
        return {o.structure for o in self.working(session, bot_id) if o.structure}

    # ----- pricing -----
    def limit_for(self, legs: list[tuple[str, int]], chains: dict[str, Chain], chase: int) -> tuple[float, float] | None:
        """(limit, mid) per unit: start at the mid, move toward the far side each chase step."""
        mid = far = 0.0
        for sym, ratio in legs:
            chain = chains.get(parse(sym).underlying)
            row = chain.row(sym) if chain is not None else None
            if row is None:
                return None
            mid += ratio * float(row["mark"])
            far += ratio * float(row["est_ask"] if ratio > 0 else row["est_bid"])
        step = min(chase * self.cfg.execution.chase_frac, 1.0)
        return mid + (far - mid) * step, mid

    def limit_from_quotes(self, legs: list[tuple[str, int]], quotes: dict[str, Quote],
                          chase: int) -> tuple[float, float]:
        """(limit, mid) per unit from live NBBO quotes."""
        mid = sum(r * quotes[s].mid for s, r in legs)
        far = sum(r * (quotes[s].ask if r > 0 else quotes[s].bid) for s, r in legs)
        step = min(chase * self.cfg.execution.chase_frac, 1.0)
        return mid + (far - mid) * step, mid

    def live_quotes(self, legs: list[tuple[str, int]]) -> tuple[dict[str, Quote], list[str]]:
        """(fresh quotes for every leg, or {}; the legs without one)."""
        got = leg_quotes(self.quotes, legs)
        now = pd.Timestamp.now(tz="UTC")
        max_age = self.cfg.execution.quote_max_age_sec
        missing = [s for s, _ in legs if s not in got or not got[s].usable(now, max_age)]
        return ({} if missing else {s: got[s] for s, _ in legs}), missing

    # ----- orders -----
    def place(self, session: Session, bot: Bot, change: Change, chains: dict[str, Chain],
              chase: int = 0, market: bool = False) -> Placement:
        opening = change.kind == "open" or (change.kind == "resize" and change.units > 0)
        busy = self.busy_sides(session)  # Alpaca rejects opposite sides working at once (wash trades)
        if any(-(1 if q > 0 else -1) in busy.get(s, ()) for s, q in change.legs):
            return Placement(None, "deferred", "an opposite order is working on a leg")
        if opening:
            held = account_positions(session, self.account)
            mine = self._tag_position(session, bot.id, change.tag)
            for sym, q in change.legs:
                others = held.get(sym, 0.0) - mine.get(sym, 0.0)
                if others and math.copysign(1, others) != math.copysign(1, q):
                    return Placement(None, "conflict",
                                     f"{sym}: another bot holds the opposite side; use another strike")
        ratios, units = normalize(change.legs)
        live: dict[str, Quote] = {}
        if not market and self.quotes is not None:
            live, missing = self.live_quotes(ratios)
            if missing and opening and self.cfg.execution.require_live_quotes:
                return Placement(None, "deferred", f"no fresh real-time quote for {', '.join(missing)}")
        if market:
            priced = None
        elif live:
            priced = self.limit_from_quotes(ratios, live, chase)
        else:
            priced = self.limit_for(ratios, chains, chase)
            if priced is None:
                return Placement(None, "deferred", "a leg is not in the current chain")
        limit, mid = priced if priced else (None, 0.0)
        und = parse(change.legs[0][0]).underlying
        order = Order(
            client_order_id=_coid(change.tag), account=self.account, interval=INTERVAL, symbol=und,
            qty=float(units if opening else -units), order_type="market" if market else "limit",
            time_in_force="day", limit_price=limit, ref_price=mid, status="new",
            legs=[{"symbol": s, "ratio": r} for s, r in ratios], structure=change.tag, chase=chase)
        marks = {}
        for sym, _ in ratios:
            chain = chains.get(parse(sym).underlying)
            m = live[sym].mid if sym in live else (chain.mark(sym) if chain is not None else None)
            if m is not None:
                marks[sym] = round(m, 4)
        order.allocations = [OrderAllocation(bot_id=bot.id, qty=float(units), reason={
            "kind": change.kind, "reason": change.reason, "note": change.note, "tag": change.tag,
            "intent": "open" if opening else "close", "leg_marks": marks,
            "priced": "market" if market else ("live" if live else "chain"),
            "nbbo": {s: q.as_dict() for s, q in live.items()}})]
        session.add(order)
        session.flush()
        try:
            bo = self.broker.submit_option(order.client_order_id,
                                           [(s, r, "open" if opening else "close") for s, r in ratios],
                                           units, limit)
            order.broker_order_id, order.status = bo.broker_order_id, bo.status
        except Exception as exc:  # noqa: BLE001 - recorded; the planner retries next tick
            log.error("option order %s rejected: %s", order.client_order_id, exc)
            order.status, order.done_at = "rejected", utcnow()
            order.allocations[0].reason = dict(order.allocations[0].reason, error=str(exc)[:300])
            return Placement(order, "rejected", str(exc)[:300])
        return Placement(order, "placed")

    def _tag_position(self, session: Session, bot_id: int, tag: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for s, q in session.execute(select(Fill.symbol, Fill.qty)
                                    .where(Fill.bot_id == bot_id, Fill.structure == tag)).all():
            out[s] = out.get(s, 0.0) + q
        return out

    def sync(self, session: Session) -> int:
        """Pull fills for working orders; one Fill per leg per increment."""
        fee = self.cfg.execution.fee_per_contract
        new = 0
        for order in self.working(session):
            if order.broker_order_id is None:
                continue
            bo = self.broker.get_order(order.client_order_id)
            legs = order.legs or []
            single = len(legs) == 1
            # Single-leg orders report contracts; mleg orders report structure units.
            filled_units = bo.filled_qty / abs(legs[0]["ratio"]) if single else bo.filled_qty
            prev_units = order.filled_qty / abs(legs[0]["ratio"]) if single else order.filled_qty
            du = filled_units - prev_units
            if du > 1e-9:
                alloc = order.allocations[0]
                prices = self._leg_prices(order, bo)
                inc = [(leg["symbol"], leg["ratio"] * du) for leg in legs]
                quotes, basis = self._fill_quotes(order, bo, inc)
                shadow = shadow_prices(inc, prices, quotes) if quotes else None
                for sym, q in inc:
                    quote = dict(quotes[sym], basis=basis) if sym in quotes else None
                    session.add(Fill(order_id=order.id, bot_id=alloc.bot_id, symbol=sym, qty=q,
                                     price=prices[sym], fees=fee * abs(q), structure=order.structure,
                                     shadow_price=shadow[sym] if shadow else None, quote=quote))
                    new += 1
                alloc.filled_qty += du
                order.filled_notional = (bo.filled_avg_price or 0.0) * bo.filled_qty
            order.filled_qty = bo.filled_qty
            order.status = bo.status
            if bo.terminal and order.done_at is None:
                order.done_at = utcnow()
        return new

    def _fill_quotes(self, order: Order, bo, legs: list[tuple[str, float]]) -> tuple[dict[str, dict], str]:
        """The NBBO a fill is judged against: at submit if it filled within fill_latency_sec (it was
        marketable), else the live quote now (it rested; the first quote seen after the fill)."""
        submitted = (order.allocations[0].reason or {}).get("nbbo") or {} if order.allocations else {}
        created, filled = order.created_at, bo.filled_at
        if submitted and created is not None and filled is not None:
            c, f = pd.Timestamp(created), pd.Timestamp(filled)
            c = c.tz_localize("UTC") if c.tzinfo is None else c
            f = f.tz_localize("UTC") if f.tzinfo is None else f
            if (f - c).total_seconds() <= self.cfg.execution.fill_latency_sec \
                    and all(s in submitted for s, _ in legs):
                return submitted, "submit"
        live, missing = self.live_quotes([(s, 1) for s, _ in legs])
        if live:
            return {s: q.as_dict() for s, q in live.items()}, "observed"
        return {}, "none"

    def _leg_prices(self, order: Order, bo) -> dict[str, float]:
        legs = order.legs or []
        if len(legs) == 1:
            return {legs[0]["symbol"]: float(bo.filled_avg_price or order.limit_price or 0.0)}
        by_leg = {leg["symbol"]: leg.get("filled_avg_price") for leg in (bo.legs or [])}
        if all(by_leg.get(leg["symbol"]) for leg in legs):
            return {s: float(p) for s, p in by_leg.items() if s in {leg["symbol"] for leg in legs}}
        # No per-leg prices: spread the net fill over the legs in proportion to their size.
        net = abs(float(bo.filled_avg_price or order.limit_price or 0.0))
        ref = order.ref_price or 0.0
        sign = 1 if ref >= 0 else -1
        refs = order.allocations[0].reason.get("leg_marks") or {}
        base = {leg["symbol"]: float(refs.get(leg["symbol"], 0.0)) for leg in legs}
        diff = sign * net - sum(leg["ratio"] * base[leg["symbol"]] for leg in legs)
        weight = sum(abs(leg["ratio"]) for leg in legs)
        return {leg["symbol"]: max(base[leg["symbol"]] + diff * (1 if leg["ratio"] > 0 else -1) / weight, 0.0)
                for leg in legs}

    def chase(self, session: Session, now: pd.Timestamp, chains: dict[str, Chain]) -> int:
        """Cancel orders older than chase_sec; re-place the unfilled part one step closer."""
        ex = self.cfg.execution
        n = 0
        for order in self.working(session):
            created = pd.Timestamp(order.created_at)
            created = created.tz_localize("UTC") if created.tzinfo is None else created
            if order.broker_order_id and order.order_type == "limit" and \
                    (now - created).total_seconds() >= ex.chase_sec:
                self.broker.cancel(order.client_order_id)
        self.sync(session)
        recent = now - pd.Timedelta(minutes=10)
        done = session.scalars(select(Order).where(
            Order.account == self.account, Order.status == "canceled", Order.created_at >= recent.to_pydatetime(),
            Order.order_type == "limit")).all()
        for order in done:
            alloc = order.allocations[0]
            if alloc.reason.get("rechased") or order.chase >= ex.chase_steps:
                continue
            remaining = int(round(alloc.qty - alloc.filled_qty))
            alloc.reason = dict(alloc.reason, rechased=True)
            if remaining <= 0:
                continue
            bot = session.get(Bot, alloc.bot_id)
            opening = alloc.reason.get("intent") == "open"
            legs = [(leg["symbol"], leg["ratio"] * remaining) for leg in order.legs]
            ch = Change(order.structure, "open" if opening else "close", legs,
                        remaining if opening else -remaining, alloc.reason.get("reason", ""),
                        alloc.reason.get("note", ""))
            p = self.place(session, bot, ch, chains, chase=order.chase + 1)
            n += p.status == "placed"
        return n
