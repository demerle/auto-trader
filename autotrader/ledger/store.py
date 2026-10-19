"""Bot ledgers are rebuilt from the fills table, so there is no mutable state to drift."""

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import Bot, CashFlow, Fill, Order, OrderAllocation
from autotrader.broker.base import TERMINAL
from autotrader.ledger.book import Book


def load_book(session: Session, bot: Bot) -> Book:
    book = Book(bot.starting_capital)
    fills = session.scalars(select(Fill).where(Fill.bot_id == bot.id).order_by(Fill.id)).all()
    for f in fills:
        book.fill(f.symbol, f.qty, f.price, f.fees)
    # Allocator flows only move cash, so their order relative to fills doesn't matter.
    book.cash += sum(session.scalars(select(CashFlow.amount).where(CashFlow.bot_id == bot.id)).all())
    return book


def working_qty(session: Session, bot: Bot) -> dict[str, float]:
    """Signed shares this bot still has working at the broker, per symbol."""
    rows = session.execute(
        select(Order.symbol, OrderAllocation.qty, OrderAllocation.filled_qty)
        .join(Order, Order.id == OrderAllocation.order_id)
        .where(OrderAllocation.bot_id == bot.id, Order.status.not_in(TERMINAL))
    ).all()
    out: dict[str, float] = defaultdict(float)
    for symbol, qty, filled in rows:
        out[symbol] += qty - filled
    return dict(out)


def ledger_positions(session: Session, account: str | None = None) -> dict[str, float]:
    """Sum of all bots' positions per symbol (what the broker account should hold)."""
    out: dict[str, float] = defaultdict(float)
    q = select(Fill.symbol, Fill.qty)
    if account is not None:
        q = q.join(Bot, Bot.id == Fill.bot_id).where(Bot.account == account)
    for symbol, qty in session.execute(q).all():
        out[symbol] += qty
    return {s: q for s, q in out.items() if abs(q) > 1e-9}
