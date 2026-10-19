"""The orchestrator loop: keep slots filled, settle results, run the next affordable job."""

import logging
import time

import pandas as pd
from sqlalchemy import case, select
from sqlalchemy.orm import Session, sessionmaker

from autotrader.alerts import alert
from autotrader.db.models import AgentJob
from autotrader.engine.state import get_state, set_state
from autotrader.evaluator.policy import Policy
from autotrader.orchestrator import budget, pipeline, slots
from autotrader.orchestrator.config import AgentsConfig
from autotrader.orchestrator.runner import run_job, run_opencode

log = logging.getLogger(__name__)
PRIORITY = case({"auditor": 0, "researcher": 1, "reviewer": 2}, value=AgentJob.agent, else_=3)
BUDGET_ALERT = "budget_alert"
PAUSED = "agents_paused"  # set on provider auth errors; cleared by `autotrader agents resume`


_HISTORY: dict[tuple[str, str], int] = {}


def history_counter(archive, cfg: AgentsConfig | None = None):
    """Sessions of SPY history available at an interval (archive plus what the provider can
    backfill), as a proxy for what agents can research. Cached per day."""
    def count(interval: str) -> int:
        key = (interval, str(pd.Timestamp.now(tz="UTC").date()))
        if key not in _HISTORY:
            try:
                start = cfg.history_start(interval) if cfg else None
                df = archive.get("SPY", interval, start=start) if cfg else archive.load("SPY", interval)
            except Exception:  # noqa: BLE001 - no data means not researchable today
                log.warning("no %s history for SPY", interval, exc_info=True)
                df = None
            _HISTORY[key] = 0 if df is None or df.empty else int(
                df.index.tz_convert("America/New_York").normalize().nunique())
        return _HISTORY[key]
    return count


def next_job(session: Session) -> AgentJob | None:
    return session.scalars(select(AgentJob).where(AgentJob.status == "queued")
                           .order_by(PRIORITY, AgentJob.id).limit(1)).first()


def tick(sessions: sessionmaker[Session], archive, cfg: AgentsConfig, policy: Policy,
         runner=run_opencode, now: pd.Timestamp | None = None, dry_run: bool = False) -> list[str]:
    pinned = now is not None  # tests pin the clock; agents' holdout follows it
    now = now or pd.Timestamp.now(tz="UTC")
    lines: list[str] = []
    with sessions() as s:
        if paused := get_state(s, PAUSED):
            return [f"paused: {paused.get('reason')} (resume with `autotrader agents resume`)"]
        lines += pipeline.settle(s, policy, cfg)
        for j in slots.fill_slots(s, policy, cfg, history_counter(archive, cfg), now):
            lines.append(f"queued researcher job for a {j.target['interval']} slot")
        for j in pipeline.schedule_reviews(s, policy, cfg, now):
            lines.append(f"queued weekly review of {j.target['slug']}")
        job = next_job(s)
        if job is None:
            (s.rollback if dry_run else s.commit)()
            return lines + ["nothing to run"]
        chk = budget.check(s, job.agent, cfg, now)
        if dry_run:
            s.rollback()
            return lines + [f"next: job {job.id} {job.agent} ({'ok' if chk.ok else chk.reason}); {chk}"]
        _budget_alert(s, chk, cfg, now)
        s.commit()
        if not chk.ok:
            return lines + [f"waiting for budget: {chk.reason}; {chk}"]
        job_id = job.id
    job = run_job(sessions, job_id, archive, cfg, policy, runner=runner,
                  now=now if pinned else None)
    lines.append(f"job {job.id} {job.agent} on {job.model}: {job.status}, ${job.cost_usd:.3f}")
    with sessions() as s:
        if str((job.result or {}).get("error_type") or "").startswith("provider.auth"):
            j = s.get(AgentJob, job.id)
            j.status, j.attempts = "queued", max(j.attempts - 1, 0)  # not the job's fault
            reason = f"opencode provider auth failed: {(job.result or {}).get('error')}"
            set_state(s, PAUSED, {"reason": reason, "since": str(now)})
            alert("Agents paused", reason + " On the host: `opencode auth login` (new Go key), "
                  "`autotrader agents auth-export`, then `autotrader agents resume`.",
                  priority="high")
            s.commit()
            return lines + [f"paused: {reason}"]
        lines += pipeline.settle(s, policy, cfg)
        s.commit()
    return lines


def _budget_alert(session: Session, chk: budget.BudgetCheck, cfg: AgentsConfig, now: pd.Timestamp) -> None:
    hot = budget.near_limit(chk, cfg)
    state = get_state(session, BUDGET_ALERT)
    if hot and state.get("day") != str(now.date()):
        alert("Agent budget nearly used", f"{chk} (limits are the arena's share of the Go caps)")
        set_state(session, BUDGET_ALERT, {"day": str(now.date())})


def run_forever(sessions, archive, cfg: AgentsConfig, policy: Policy, poll_sec: float = 60.0) -> None:
    log.info("orchestrator started: models %s", cfg.models)
    with sessions() as s:  # jobs cut off by a restart count as failed attempts
        for job in s.scalars(select(AgentJob).where(AgentJob.status == "running")):
            job.status = "failed"
            job.result = dict(job.result or {}) | {"error": "orchestrator restarted mid-job"}
        s.commit()
    while True:
        try:
            for line in tick(sessions, archive, cfg, policy):
                log.info(line)
        except Exception:  # noqa: BLE001 - keep the loop alive
            log.exception("orchestrator tick failed")
        time.sleep(poll_sec)
