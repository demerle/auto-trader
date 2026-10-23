"""Read models shared by the CLI, the dashboard and the daily report, so they agree.

Everything returns plain dicts and lists (JSON-safe: NaN becomes None).
"""

import inspect
import math
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autotrader.db.models import (
    ARENA, AgentJob, AgentNote, Alert, Bot, BotStage, CashFlow, EquitySnapshot, Evaluation, Fill,
    Report, Submission,
)
from autotrader.engine.state import KILL_SWITCH, get_state
from autotrader.evaluator.record import build_record
from autotrader.evaluator.run import EVALUATED_STAGES

ET = "America/New_York"
LEAGUE_KEYS = ("total_return", "excess_return", "spy_return", "sharpe", "psr", "max_drawdown",
               "alpha", "backtest_z")
HELD_STAGES = (BotStage.PAPER_PROBATION, BotStage.PAPER_PROVEN, BotStage.LIVE, BotStage.PAUSED,
               BotStage.DEMOTED)


def num(x):
    """JSON-safe float."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def et_day_bounds(day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC start/end of an ET calendar date."""
    start = pd.Timestamp(day).tz_localize(ET)
    return start.tz_convert("UTC"), (start + pd.Timedelta(days=1)).tz_convert("UTC")


def today_et(now: pd.Timestamp | None = None) -> str:
    now = now or pd.Timestamp.now(tz="UTC")
    return now.tz_convert(ET).date().isoformat()


# ----- bots -----

def latest_snapshot(session: Session, bot_id: int, before=None) -> EquitySnapshot | None:
    q = select(EquitySnapshot).where(EquitySnapshot.bot_id == bot_id)
    if before is not None:
        q = q.where(EquitySnapshot.ts < before)
    return session.scalars(q.order_by(EquitySnapshot.ts.desc()).limit(1)).first()


def league(session: Session, spy_returns: pd.Series | None) -> list[dict]:
    """Bots under evaluation ranked by paper Sharpe (best first)."""
    rows = []
    for bot in session.scalars(select(Bot).where(Bot.stage.in_(EVALUATED_STAGES), Bot.group == ARENA).order_by(Bot.id)):
        rec = build_record(session, bot, spy_returns)
        m = rec.metrics
        rows.append({
            "slug": bot.slug, "strategy": f"{bot.strategy.name}@{bot.strategy.version}",
            "interval": bot.strategy.interval, "stage": bot.stage.value, "author": bot.strategy.author,
            "days": rec.days, "trades": rec.trades, "capital": bot.capital,
            "live_eligible": bot.stage == BotStage.PAPER_PROVEN,
            **{k: num(m.get(k)) for k in LEAGUE_KEYS},
        })
    rows.sort(key=lambda r: r["sharpe"] if r["sharpe"] is not None else 0.0, reverse=True)
    return rows


def roster(session: Session) -> list[dict]:
    """Every bot with its last equity snapshot."""
    out = []
    for bot in session.scalars(select(Bot).where(Bot.group == ARENA).order_by(Bot.id)):
        snap = latest_snapshot(session, bot.id)
        out.append({"slug": bot.slug, "strategy": f"{bot.strategy.name}@{bot.strategy.version}",
                    "interval": bot.strategy.interval, "stage": bot.stage.value,
                    "capital": bot.capital, "equity": snap.equity if snap else None,
                    "positions": dict(snap.positions) if snap else {},
                    "paused_until": bot.paused_until.isoformat() if bot.paused_until else None})
    return out


def recent_fills(session: Session, limit: int = 40) -> list[dict]:
    """The arena's latest fills (newest first), with the order behind each."""
    rows = session.execute(select(Fill, Bot.slug).join(Bot, Bot.id == Fill.bot_id)
                           .where(Bot.group == ARENA).order_by(Fill.id.desc()).limit(limit)).all()
    out = []
    for f, slug in rows:
        out.append({"ts": f.filled_at.isoformat() if f.filled_at else None, "bot": slug, "symbol": f.symbol,
                    "qty": f.qty, "price": f.price, "value": f.qty * f.price, "internal": f.internal})
    return out


def strategy_source(sv) -> str:
    from autotrader.strategies import registry

    try:
        cls = registry.get(sv.name, sv.version)
    except KeyError:
        return "(source not found)"
    path = getattr(cls, "source_path", None) or inspect.getfile(cls)
    try:
        return Path(path).read_text()
    except OSError:
        return "(source not readable)"


def equity_series(session: Session, bot_id: int, max_points: int = 1500) -> list[tuple[str, float]]:
    """Equity over time; one point per ET day when there are too many snapshots."""
    rows = session.execute(select(EquitySnapshot.ts, EquitySnapshot.equity)
                           .where(EquitySnapshot.bot_id == bot_id)
                           .order_by(EquitySnapshot.ts)).all()
    if len(rows) > max_points:
        by_day: dict[str, tuple] = {}
        for ts, eq in rows:
            by_day[_utc(ts).tz_convert(ET).date().isoformat()] = (ts, eq)
        rows = list(by_day.values())
    return [(_utc(ts).isoformat(), float(eq)) for ts, eq in rows]


def bot_detail(session: Session, slug: str, spy_returns: pd.Series | None = None) -> dict | None:
    bot = session.scalars(select(Bot).where(Bot.slug == slug)).first()
    if bot is None:
        return None
    sv = bot.strategy
    rec = build_record(session, bot, spy_returns)
    snap = latest_snapshot(session, bot.id)
    fills = session.scalars(select(Fill).where(Fill.bot_id == bot.id)
                            .order_by(Fill.id.desc()).limit(50)).all()
    evals = session.scalars(select(Evaluation).where(Evaluation.bot_id == bot.id)
                            .order_by(Evaluation.id.desc()).limit(20)).all()
    flows = session.scalars(select(CashFlow).where(CashFlow.bot_id == bot.id)
                            .order_by(CashFlow.id.desc()).limit(20)).all()
    notes = session.scalars(select(AgentNote).where(AgentNote.bot_id == bot.id)
                            .order_by(AgentNote.id.desc()).limit(20)).all()
    sub = session.scalars(select(Submission).where(Submission.bot_id == bot.id)).first()
    equity = equity_series(session, bot.id)
    spy_line = []
    if spy_returns is not None and equity:
        start = pd.Timestamp(equity[0][0])
        path = (1 + spy_returns[spy_returns.index > start.normalize()]).cumprod() * bot.starting_capital
        spy_line = [(ts.isoformat(), float(v)) for ts, v in path.items()]
    return {
        "slug": bot.slug, "stage": bot.stage.value, "account": bot.account,
        "strategy": f"{sv.name}@{sv.version}", "interval": sv.interval, "author": sv.author,
        "description": sv.description, "params": bot.params, "code_sha": sv.code_sha,
        "starting_capital": bot.starting_capital, "capital": bot.capital, "live_cap": bot.live_cap,
        "created_at": bot.created_at.isoformat(), "stage_since": bot.stage_since.isoformat(),
        "paused_until": bot.paused_until.isoformat() if bot.paused_until else None,
        "equity": snap.equity if snap else None, "cash": snap.cash if snap else None,
        "positions": dict(snap.positions) if snap else {},
        "days": rec.days, "trades": rec.trades,
        "metrics": {k: num(v) for k, v in rec.metrics.items()},
        "backtest": {k: num(v) for k, v in (sv.backtest_metrics or {}).items()},
        "equity_series": equity, "spy_series": spy_line,
        "fills": [{"ts": f.filled_at.isoformat(), "symbol": f.symbol, "qty": f.qty, "price": f.price,
                   "fees": f.fees, "internal": f.internal} for f in fills],
        "evaluations": [{"ts": e.ts.isoformat(), "decision": e.decision, "rationale": e.rationale}
                        for e in evals],
        "cash_flows": [{"ts": c.ts.isoformat(), "amount": c.amount, "reason": c.reason} for c in flows],
        "notes": bot.notes,
        "agent_notes": [{"ts": n.created_at.isoformat(), "topic": n.topic, "body": n.body} for n in notes],
        "submission_id": sub.id if sub else None,
        "source": strategy_source(sv),
    }


def graveyard(session: Session, limit: int = 100) -> list[dict]:
    bots = session.scalars(select(Bot).where(Bot.stage == BotStage.ERADICATED, Bot.group == ARENA)
                           .order_by(Bot.stage_since.desc()).limit(limit)).all()
    return [{"slug": b.slug, "strategy": f"{b.strategy.name}@{b.strategy.version}",
             "interval": b.strategy.interval, "died": b.stage_since.isoformat(),
             "autopsy": b.notes} for b in bots]


# ----- account -----

def account_overview(session: Session, now: pd.Timestamp | None = None) -> dict:
    """Arena-wide equity, today's P&L (vs the last snapshot before today, ET), kill switch."""
    now = now or pd.Timestamp.now(tz="UTC")
    day_start, _ = et_day_bounds(today_et(now))
    equity = prev = capital = 0.0
    stages: dict[str, int] = {}
    for bot in session.scalars(select(Bot).where(Bot.group == ARENA)):
        stages[bot.stage.value] = stages.get(bot.stage.value, 0) + 1
        if bot.stage not in HELD_STAGES:
            continue
        capital += bot.capital
        snap = latest_snapshot(session, bot.id)
        before = latest_snapshot(session, bot.id, before=day_start.to_pydatetime())
        eq = snap.equity if snap else bot.capital
        equity += eq
        prev += before.equity if before else eq
    flows = session.scalar(select(func.coalesce(func.sum(CashFlow.amount), 0.0))
                           .where(CashFlow.ts >= day_start.to_pydatetime())) or 0.0
    day_pnl = equity - prev - flows
    kill = get_state(session, KILL_SWITCH)
    return {"equity": equity, "capital": capital, "day_pnl": day_pnl,
            "day_return": day_pnl / prev if prev else 0.0, "stages": stages,
            "killed": bool(kill.get("on")), "kill_reason": kill.get("reason"), "kill_at": kill.get("at")}


def recent_alerts(session: Session, limit: int = 20, since=None, until=None) -> list[dict]:
    q = select(Alert)
    if since is not None:
        q = q.where(Alert.ts >= since)
    if until is not None:
        q = q.where(Alert.ts < until)
    return [{"ts": _utc(a.ts).isoformat(), "title": a.title, "message": a.message, "priority": a.priority}
            for a in session.scalars(q.order_by(Alert.id.desc()).limit(limit))]


def latest_report(session: Session) -> Report | None:
    return session.scalars(select(Report).order_by(Report.date.desc()).limit(1)).first()


# ----- agents -----

def job_row(j: AgentJob) -> dict:
    dur = None
    if j.started_at and j.finished_at:
        dur = int((_utc(j.finished_at) - _utc(j.started_at)).total_seconds())
    tokens = j.tokens or {}
    return {"id": j.id, "agent": j.agent, "status": j.status, "model": j.model or "",
            "cost_usd": j.cost_usd or 0.0, "est_cost_usd": j.est_cost_usd or 0.0,
            "tokens": int(sum(v for v in tokens.values() if isinstance(v, (int, float)))),
            "attempts": j.attempts, "target": j.target or {}, "bot_id": j.bot_id,
            "submission_id": j.submission_id,
            "created_at": j.created_at.isoformat(),
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            "duration_sec": dur, "error": (j.result or {}).get("error"),
            "transcript_path": j.transcript_path}


def submission_row(x: Submission) -> dict:
    return {"id": x.id, "job_id": x.job_id, "name": x.name, "version": x.version,
            "interval": x.interval, "status": x.status, "reasons": list(x.reasons or []),
            "gate": {k: num(v) for k, v in (x.gate_metrics or {}).items()},
            "holdout": {k: num(v) for k, v in (x.holdout_metrics or {}).items()},
            "n_trials": x.n_trials, "audit": x.audit or {}, "bot_id": x.bot_id,
            "rationale": x.rationale, "created_at": x.created_at.isoformat(),
            "challenger": x.parent_strategy_id is not None}


def agents_overview(session: Session, cfg, policy, now: pd.Timestamp | None = None,
                    limit: int = 50) -> dict:
    from autotrader.orchestrator import budget, slots
    from autotrader.orchestrator.loop import PAUSED, QUOTA_HOLD

    chk = budget.check(session, "researcher", cfg, now)
    jobs = session.scalars(select(AgentJob).order_by(AgentJob.id.desc()).limit(limit)).all()
    subs = session.scalars(select(Submission).order_by(Submission.id.desc()).limit(30)).all()
    notes = session.scalars(select(AgentNote).order_by(AgentNote.id.desc()).limit(20)).all()
    counts = session.execute(select(AgentJob.agent, AgentJob.status, func.count())
                             .group_by(AgentJob.agent, AgentJob.status)).all()
    return {
        "model": cfg.model,
        "paused": get_state(session, PAUSED) or None,
        "quota_hold": get_state(session, QUOTA_HOLD) or None,
        "jobs_today": budget.jobs_started(session, now or pd.Timestamp.now(tz="UTC")),
        "slots": {"total": policy.slots, "bots": len(slots.active_bots(session)),
                  "in_flight": slots.in_flight(session), "open": slots.open_slots(session, policy)},
        "budget": {"spend": chk.spend, "limits": chk.limits, "estimate": chk.estimate,
                   "ok": chk.ok, "reason": chk.reason, "text": str(chk)},
        "counts": [{"agent": a, "status": s, "n": n} for a, s, n in sorted(counts)],
        "jobs": [job_row(j) for j in jobs],
        "submissions": [submission_row(x) for x in subs],
        "notes": [{"ts": n.created_at.isoformat(), "job_id": n.job_id, "topic": n.topic,
                   "body": n.body} for n in notes],
    }
