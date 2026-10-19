"""Options bots' books, rebuilt from the fills table (plus weekly-reset cash flows)."""

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import Bot, CashFlow, Fill
from autotrader.options.book import OptionBook


def load_option_book(session: Session, bot: Bot) -> OptionBook:
    book = OptionBook(bot.starting_capital)
    events = [(f.filled_at, 0, f) for f in session.scalars(select(Fill).where(Fill.bot_id == bot.id))]
    events += [(c.ts, 1, c) for c in session.scalars(select(CashFlow).where(CashFlow.bot_id == bot.id))]
    for ts, kind, e in sorted(events, key=lambda x: (x[0], x[1], x[2].id)):
        if kind == 0:
            book.fill(e.structure or e.symbol, e.symbol, e.qty, e.price, e.fees, _ts(e.filled_at))
        else:
            book.cash_flow(e.amount)
    return book


def account_positions(session: Session, account: str) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    rows = session.execute(select(Fill.symbol, Fill.qty).join(Bot, Bot.id == Fill.bot_id)
                           .where(Bot.account == account)).all()
    for s, q in rows:
        out[s] += q
    return {s: q for s, q in out.items() if abs(q) > 1e-9}


def _ts(dt):
    import pandas as pd

    t = pd.Timestamp(dt)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
