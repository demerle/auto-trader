"""Scheduled evaluator work: daily verdicts after the close, weekly capital tilt."""

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from autotrader.data.calendar import schedule, session_for
from autotrader.evaluator.allocator import apply_allocation, plan_allocation
from autotrader.evaluator.policy import Policy
from autotrader.evaluator.run import evaluate_all


def spy_daily_returns(archive, now: pd.Timestamp) -> pd.Series | None:
    try:
        df = archive.get("SPY", "1d", start=now - pd.Timedelta(days=800))
    except Exception:  # noqa: BLE001 - evaluation can proceed without a benchmark
        return None
    return df["close"].pct_change().dropna()


def last_session_of_week(now: pd.Timestamp) -> bool:
    today = session_for(now)
    if today is None:
        return False
    nxt = schedule(now.normalize() + pd.Timedelta(days=1), now.normalize() + pd.Timedelta(days=10))
    return len(nxt) == 0 or nxt.index[0].isocalendar().week != now.normalize().isocalendar().week


def run_daily(sessions: sessionmaker[Session], archive, policy: Policy,
              now: pd.Timestamp | None = None, allocate: bool | None = None,
              apply: bool = True) -> list[str]:
    now = now or pd.Timestamp.now(tz="UTC")
    if allocate is None:
        allocate = last_session_of_week(now)
    spy = spy_daily_returns(archive, now)
    lines = []
    with sessions() as session:
        for o in evaluate_all(session, policy, spy, apply=apply):
            lines.append(f"{o.bot.slug:18} {o.verdict.action:10} {o.verdict.rationale}")
        if allocate:
            moves = plan_allocation(session, policy)
            applied = apply_allocation(session, policy, moves) if apply else moves
            for mv in applied:
                lines.append(f"{mv.bot.slug:18} capital ${mv.current:,.0f} -> ${mv.target:,.0f} "
                             f"(shrunk Sharpe {mv.score:+.2f})")
        if apply:
            session.commit()
        else:
            session.rollback()
    return lines
