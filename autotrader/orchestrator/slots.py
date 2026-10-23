"""Keep the arena full: one researcher job per open slot, aimed at an under-used interval."""

from collections import Counter

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import ARENA, AgentJob, Bot, BotStage, Submission
from autotrader.evaluator.policy import Policy
from autotrader.orchestrator.config import AgentsConfig

OCCUPYING = (BotStage.PAPER_PROBATION, BotStage.PAPER_PROVEN, BotStage.LIVE, BotStage.PAUSED)
IN_FLIGHT_JOB = ("queued", "running")


def active_bots(session: Session) -> list[Bot]:
    return list(session.scalars(select(Bot).where(Bot.stage.in_(OCCUPYING), Bot.group == ARENA)).all())


def in_flight(session: Session) -> int:
    """Research that may still turn into a bot: open researcher jobs and unresolved submissions."""
    jobs = session.scalars(select(AgentJob).where(AgentJob.agent == "researcher",
                                                  AgentJob.status.in_(IN_FLIGHT_JOB))).all()
    subs = session.scalars(select(Submission).where(Submission.parent_strategy_id.is_(None),
                                                    Submission.status.in_(("audit_pending", "approved")))
                           ).all()
    busy_jobs = {s.job_id for s in subs}
    return len(subs) + sum(1 for j in jobs if j.id not in busy_jobs)


def open_slots(session: Session, policy: Policy) -> int:
    return max(policy.slots - len(active_bots(session)) - in_flight(session), 0)


def choose_interval(session: Session, cfg: AgentsConfig, history_sessions) -> str | None:
    """The least-represented research interval that has enough archived history."""
    counts = Counter(b.strategy.interval for b in active_bots(session))
    for j in session.scalars(select(AgentJob).where(AgentJob.agent == "researcher",
                                                    AgentJob.status.in_(IN_FLIGHT_JOB))).all():
        if iv := (j.target or {}).get("interval"):
            counts[iv] += 1
    eligible = [iv for iv in cfg.research_intervals if history_sessions(iv) >= cfg.min_history_sessions]
    if not eligible:
        return None
    return min(eligible, key=lambda iv: (counts[iv], cfg.research_intervals.index(iv)))


def recently_failed(session: Session, agent: str, cfg: AgentsConfig, now: pd.Timestamp | None = None) -> bool:
    """A job of this role failed for good within the refill cooldown: don't queue another one yet."""
    now = now or pd.Timestamp.now(tz="UTC")
    since = (now - pd.Timedelta(hours=cfg.refill_cooldown_hours)).to_pydatetime()
    jobs = session.scalars(select(AgentJob).where(AgentJob.agent == agent, AgentJob.status == "failed",
                                                  AgentJob.finished_at >= since)).all()
    return any((j.result or {}).get("settled") for j in jobs)


def research_prompt(interval: str, reason: str, extra: str = "") -> str:
    text = (f"A slot is open in the arena. Research, write, test and submit one new {interval} "
            f"strategy.\nWhy this slot is open: {reason}\n")
    return text + (f"\n{extra}\n" if extra else "")


def fill_slots(session: Session, policy: Policy, cfg: AgentsConfig, history_sessions,
               now: pd.Timestamp | None = None) -> list[AgentJob]:
    """Give queued eradication jobs (from the evaluator) a target interval, then queue one
    researcher job for an unclaimed open slot. At most one queued researcher at a time."""
    created = []
    for job in session.scalars(select(AgentJob).where(AgentJob.agent == "researcher",
                                                      AgentJob.status == "queued")).all():
        if not (job.target or {}).get("interval"):
            bot = session.get(Bot, job.bot_id) if job.bot_id else None
            iv = bot.strategy.interval if bot else choose_interval(session, cfg, history_sessions)
            if iv not in cfg.research_intervals or history_sessions(iv) < cfg.min_history_sessions:
                iv = choose_interval(session, cfg, history_sessions)
            job.target = {"interval": iv, "reason": "replace an eradicated bot"}
    queued = session.scalars(select(AgentJob).where(AgentJob.agent == "researcher",
                                                    AgentJob.status == "queued")).first()
    if queued is None and open_slots(session, policy) > 0 and not recently_failed(session, "researcher", cfg, now):
        iv = choose_interval(session, cfg, history_sessions)
        if iv is not None:
            reason = f"the arena has {policy.slots} slots and only {len(active_bots(session))} bots"
            job = AgentJob(agent="researcher", model="", status="queued", target={
                "interval": iv, "reason": reason}, prompt=research_prompt(iv, reason))
            session.add(job)
            created.append(job)
    session.flush()
    return created


