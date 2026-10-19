"""A bot's paper track record: daily time-weighted returns (capital moves by the
allocator are not gains), realized trades, and scores against SPY and its backtest."""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.backtest.metrics import compute_metrics
from autotrader.db.models import Bot, CashFlow, EquitySnapshot, Fill
from autotrader.ledger.book import Book


@dataclass
class PaperRecord:
    bot_id: int
    returns: pd.Series  # daily TWR, indexed by session date at 00:00 UTC
    realized: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def days(self) -> int:
        return len(self.returns)

    @property
    def trades(self) -> int:
        return len(self.realized)


def _day(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return pd.Timestamp(t.tz_convert("America/New_York").date()).tz_localize("UTC")


def daily_returns(starting_capital: float, snaps: list[tuple], flows: list[tuple]) -> pd.Series:
    """snaps: [(ts, equity)], flows: [(ts, amount)] -> daily time-weighted returns.

    A flow counts toward the first snapshot taken after it, and is treated as
    arriving at the start of that day.
    """
    if not snaps:
        return pd.Series(dtype="float64")
    snap_df = pd.DataFrame(snaps, columns=["ts", "equity"])
    snap_df["ts"] = pd.to_datetime(snap_df["ts"], utc=True)
    snap_df = snap_df.sort_values("ts")
    snap_df["day"] = snap_df["ts"].map(_day)
    eod = snap_df.groupby("day")["equity"].last()

    flow_by_day: dict[pd.Timestamp, float] = {}
    for ts, amount in flows:
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        after = snap_df[snap_df["ts"] > ts]
        if not after.empty:
            d = after["day"].iloc[0]
            flow_by_day[d] = flow_by_day.get(d, 0.0) + amount

    prev = starting_capital
    out = {}
    for day, eq in eod.items():
        base = prev + flow_by_day.get(day, 0.0)
        out[day] = eq / base - 1 if base > 0 else 0.0
        prev = eq
    return pd.Series(out, dtype="float64").sort_index()


def build_record(session: Session, bot: Bot, spy_returns: pd.Series | None = None) -> PaperRecord:
    snaps = session.execute(select(EquitySnapshot.ts, EquitySnapshot.equity)
                            .where(EquitySnapshot.bot_id == bot.id)).all()
    flows = session.execute(select(CashFlow.ts, CashFlow.amount)
                            .where(CashFlow.bot_id == bot.id)).all()
    rets = daily_returns(bot.starting_capital, [tuple(r) for r in snaps], [tuple(r) for r in flows])

    book = Book(bot.starting_capital)
    realized = []
    for f in session.scalars(select(Fill).where(Fill.bot_id == bot.id).order_by(Fill.id)).all():
        r = book.fill(f.symbol, f.qty, f.price, f.fees)
        if r is not None:
            realized.append(r)

    rec = PaperRecord(bot.id, rets, realized)
    if len(rets) >= 2:
        equity = (1 + rets).cumprod()
        equity = pd.concat([pd.Series([1.0], index=[rets.index[0] - pd.Timedelta(days=1)]), equity])
        m = compute_metrics(equity, 252, realized, benchmark_returns=spy_returns)
        if spy_returns is not None:
            spy = spy_returns.reindex(rets.index).fillna(0.0)
            m["spy_return"] = float((1 + spy).prod() - 1)
            m["excess_return"] = m["total_return"] - m["spy_return"]
        m["backtest_z"] = backtest_z(rets, bot.strategy.backtest_metrics or {})
        rec.metrics = m
    return rec


def backtest_z(paper: pd.Series, bt: dict) -> float:
    """How many standard errors the paper mean daily return sits from the backtest's."""
    mean, std = bt.get("daily_mean"), bt.get("daily_std")
    if mean is None or not std or len(paper) < 2:
        return float("nan")
    return float((paper.mean() - mean) / (std / math.sqrt(len(paper))))


def shrunk_sharpe(returns: pd.Series, prior_days: int) -> float:
    """Annualized Sharpe shrunk toward 0 by n / (n + prior_days)."""
    n = len(returns)
    if n < 2 or not returns.std(ddof=1):
        return 0.0
    sr = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    return float(sr * n / (n + prior_days))
