"""Human controls over the agents, shared by the CLI and the dashboard."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import AgentJob, Bot, SystemState, utcnow
from autotrader.engine.state import set_state
from autotrader.orchestrator.slots import research_prompt

PAUSED = "agents_paused"  # set on provider auth errors or by hand; cleared by resume()
QUOTA_HOLD = "agents_quota_hold"  # Go's usage limit is used up: no jobs until `until`; cleared by resume()
INTERVALS = ("1m", "5m", "15m", "1h", "1d")


class ControlError(ValueError):
    pass


def queue_research(session: Session, interval: str, note: str = "") -> AgentJob:
    if interval not in INTERVALS:
        raise ControlError(f"interval must be one of {', '.join(INTERVALS)}")
    job = AgentJob(agent="researcher", model="", target={"interval": interval, "reason": "queued by hand"},
                   prompt=research_prompt(interval, "queued by hand", note))
    session.add(job)
    session.flush()
    return job


def queue_review(session: Session, slug: str, note: str = "") -> AgentJob:
    bot = session.scalars(select(Bot).where(Bot.slug == slug)).first()
    if bot is None:
        raise ControlError(f"no bot {slug}")
    job = AgentJob(agent="reviewer", model="", bot_id=bot.id, target={"slug": slug},
                   prompt=f"Review of {slug} requested by hand. {note}")
    session.add(job)
    session.flush()
    return job


def cancel_job(session: Session, job_id: int) -> AgentJob:
    job = session.get(AgentJob, job_id)
    if job is None or job.status != "queued":
        raise ControlError("only queued jobs can be cancelled")
    job.status = "cancelled"
    job.finished_at = utcnow()
    return job


def pause(session: Session, reason: str) -> None:
    set_state(session, PAUSED, {"reason": reason, "since": utcnow().isoformat()})


def resume(session: Session) -> bool:
    """Clear a pause and any Go quota hold (the next job runs now)."""
    rows = [r for r in (session.get(SystemState, PAUSED), session.get(SystemState, QUOTA_HOLD)) if r is not None]
    for row in rows:
        session.delete(row)
    return bool(rows)
