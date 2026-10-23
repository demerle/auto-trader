"""Spend tracking against the agents' share of the OpenCode Go caps.

Spend is the actual cost opencode reports per job. A job starts only if its
estimated cost fits in every rolling window and fewer than `max_jobs_per_day`
jobs started in the last 24 hours (the reported cost undercounts what Go meters).
"""

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import AgentJob
from autotrader.orchestrator.config import AgentsConfig

WINDOWS = {"five_hours": pd.Timedelta(hours=5), "week": pd.Timedelta(days=7),
           "month": pd.Timedelta(days=30)}


@dataclass
class BudgetCheck:
    ok: bool
    estimate: float
    spend: dict[str, float]
    limits: dict[str, float]
    reason: str = ""

    def __str__(self) -> str:
        parts = [f"{w} ${self.spend[w]:.2f}/${self.limits[w]:.2f}" for w in WINDOWS]
        return ", ".join(parts)


def spend(session: Session, now: pd.Timestamp) -> dict[str, float]:
    start = now - max(WINDOWS.values())
    rows = session.execute(select(AgentJob.started_at, AgentJob.cost_usd)
                           .where(AgentJob.started_at >= start.to_pydatetime())).all()
    out = {}
    for w, delta in WINDOWS.items():
        cutoff = now - delta
        out[w] = float(sum(c or 0.0 for t, c in rows if _utc(t) >= cutoff))
    return out


def estimate(session: Session, role: str, cfg: AgentsConfig, n: int = 5) -> float:
    costs = session.scalars(select(AgentJob.cost_usd).where(
        AgentJob.agent == role, AgentJob.status.in_(("succeeded", "failed")), AgentJob.cost_usd > 0)
        .order_by(AgentJob.id.desc()).limit(n)).all()
    if costs:
        return float(sum(costs) / len(costs))
    return cfg.budget.seed_estimates.get(role, 1.0)


def jobs_started(session: Session, now: pd.Timestamp, hours: float = 24.0) -> int:
    since = now - pd.Timedelta(hours=hours)
    return sum(1 for t in session.scalars(select(AgentJob.started_at).where(
        AgentJob.started_at >= since.to_pydatetime())) if _utc(t) >= since)


def check(session: Session, role: str, cfg: AgentsConfig, now: pd.Timestamp | None = None) -> BudgetCheck:
    now = now or pd.Timestamp.now(tz="UTC")
    s = spend(session, now)
    est = estimate(session, role, cfg)
    limits = cfg.budget.limits
    over = [w for w in WINDOWS if s[w] + est > limits[w]]
    reason = f"{role} (~${est:.2f}) would exceed the {', '.join(over)} budget" if over else ""
    if not over and (n := jobs_started(session, now)) >= cfg.budget.max_jobs_per_day:
        reason = f"{n} agent jobs started in the last 24 hours (cap {cfg.budget.max_jobs_per_day})"
    return BudgetCheck(not reason, est, s, limits, reason)


def near_limit(check_: BudgetCheck, cfg: AgentsConfig) -> list[str]:
    return [w for w in WINDOWS if check_.spend[w] >= cfg.budget.alert_at * check_.limits[w]]


def _utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
