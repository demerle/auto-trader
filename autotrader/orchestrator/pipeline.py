"""What happens after an agent job finishes: audits, deployment, retries.

A strategy reaches paper trading only through `deploy`, which runs here in the
trusted orchestrator, never in an agent.
"""

import hashlib
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.alerts import alert
from autotrader.db.models import AgentJob, Bot, BotStage, Evaluation, Report, Submission
from autotrader.engine.roster import ensure_strategy_version
from autotrader.evaluator.policy import Policy
from autotrader.evaluator.run import clean_json
from autotrader.orchestrator.config import AgentsConfig
from autotrader.orchestrator.slots import active_bots, research_prompt
from autotrader.reports.daily import post_commentary
from autotrader.strategies import registry
from autotrader.strategies.lint import lint_source

log = logging.getLogger(__name__)


class DeployError(RuntimeError):
    pass


def _generated_path(sub: Submission) -> Path:
    return registry.GENERATED_DIR / f"{sub.name}_v{sub.version}.py"


def _slug(session: Session, sub: Submission) -> str:
    base = sub.name.replace("_", "-") + ("" if sub.version == "1" else f"-v{sub.version}")
    slug, n = base, 2
    while session.scalars(select(Bot).where(Bot.slug == slug)).first():
        slug, n = f"{base}-{n}", n + 1
    return slug


def deploy(session: Session, sub: Submission, policy: Policy, cfg: AgentsConfig) -> Bot:
    """Freeze the audited source into strategies/generated/, register it, create its bot."""
    if hashlib.sha256(sub.source.encode()).hexdigest() != sub.code_sha:
        raise DeployError("submission source does not match its recorded hash")
    if problems := lint_source(sub.source):
        raise DeployError("lint failed: " + "; ".join(problems))
    path = _generated_path(sub)
    if path.exists() and path.read_text() != sub.source:
        raise DeployError(f"{path.name} already exists with different code")
    path.write_text(sub.source)
    try:
        sv = ensure_strategy_version(session, sub.name, sub.version, author=f"job-{sub.job_id}",
                                     backtest_metrics=clean_json(sub.gate_metrics | {"gate_passed": 1.0}))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    bot = Bot(slug=_slug(session, sub), strategy_id=sv.id, params=sub.params or {},
              stage=BotStage.PAPER_PROBATION, account="paper",
              starting_capital=policy.base_capital, capital=policy.base_capital,
              notes=f"Written by agent job {sub.job_id}. Rationale: {sub.rationale[:1500]}")
    session.add(bot)
    session.flush()
    sub.status, sub.bot_id = "deployed", bot.id
    if cfg.auto_commit:
        _git_commit(path, f"Agent strategy {sub.name}@{sub.version} ({sub.interval}, job {sub.job_id})")
    alert(f"New bot {bot.slug}", f"{sub.name}@{sub.version} ({sub.interval}) passed the gate and "
          f"the audit; paper trading with ${bot.capital:,.0f}. OOS Sharpe "
          f"{sub.gate_metrics.get('sharpe', float('nan')):.2f}.")
    return bot


def _git_commit(path: Path, message: str) -> None:
    repo = path.parent
    try:
        inside = subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True, timeout=10)
        if inside.stdout.strip() != "true":
            return
        subprocess.run(["git", "-C", str(repo), "add", str(path)], check=True, timeout=30,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", message, "--", str(path)],
                       check=True, timeout=30, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not git-commit %s: %s", path, exc)


def queue_audits(session: Session) -> list[AgentJob]:
    """One auditor job per gated submission that has none yet."""
    have = {j.submission_id for j in session.scalars(select(AgentJob).where(AgentJob.agent == "auditor"))}
    out = []
    for sub in session.scalars(select(Submission).where(Submission.status == "audit_pending")):
        if sub.id not in have:
            job = AgentJob(agent="auditor", model="", status="queued", submission_id=sub.id,
                           bot_id=None, target={"submission_id": sub.id},
                           prompt=f"Audit submission {sub.id} ({sub.name}@{sub.version}, {sub.interval}).")
            session.add(job)
            out.append(job)
    session.flush()
    return out


def settle(session: Session, policy: Policy, cfg: AgentsConfig) -> list[str]:
    """Act on finished jobs and resolved audits. Idempotent; run every loop."""
    notes: list[str] = []
    for job in session.scalars(select(AgentJob).where(AgentJob.status == "failed")):
        if (job.result or {}).get("settled"):
            continue
        has_sub = session.scalars(select(Submission).where(Submission.job_id == job.id)).first()
        verdict = (job.agent == "auditor" and session.get(Submission, job.submission_id).audit
                   or job.agent == "supervisor" and bool(_commentary(session, job)))
        if not has_sub and not verdict and job.attempts < cfg.max_attempts:
            job.status = "queued"  # retry (on the fallback model)
            notes.append(f"job {job.id} ({job.agent}) failed; retrying")
        else:
            job.result = dict(job.result or {}) | {"settled": True}
            if not has_sub and not verdict:
                alert(f"Agent job {job.id} failed", f"{job.agent}: {(job.result or {}).get('error')}")
                notes.append(f"job {job.id} ({job.agent}) failed for good")

    queue_audits(session)
    notes += post_commentary(session)

    for sub in session.scalars(select(Submission).where(Submission.status == "audit_pending")):
        audit = sub.audit or {}
        if "approve" in audit:
            sub.status = "approved" if audit["approve"] else "rejected"
            notes.append(f"submission {sub.id} {sub.name}@{sub.version}: {sub.status} by audit")
            if not audit["approve"]:
                _requeue_after_reject(session, sub)
            continue
        auditors = session.scalars(select(AgentJob).where(AgentJob.agent == "auditor",
                                                          AgentJob.submission_id == sub.id)).all()
        if auditors and all(j.status == "failed" and (j.result or {}).get("settled") for j in auditors):
            sub.status = "rejected"
            sub.audit = {"approve": False, "findings": "audit could not be completed"}
            notes.append(f"submission {sub.id}: rejected (audit failed)")

    for sub in session.scalars(select(Submission).where(Submission.status == "approved")):
        challenger = sub.parent_strategy_id is not None
        if challenger and len(active_bots(session)) >= policy.slots:
            continue  # a new version waits for a free slot
        try:
            bot = deploy(session, sub, policy, cfg)
            notes.append(f"deployed {sub.name}@{sub.version} as {bot.slug}")
        except Exception as exc:  # noqa: BLE001
            sub.status = "rejected"
            sub.reasons = list(sub.reasons or []) + [f"deploy failed: {exc}"]
            log.exception("deploy of submission %s failed", sub.id)
            alert(f"Deploy failed: {sub.name}@{sub.version}", str(exc))
    session.flush()
    return notes


def _requeue_after_reject(session: Session, sub: Submission) -> None:
    if sub.parent_strategy_id is not None:
        return
    findings = (sub.audit or {}).get("findings", "")[:3000]
    reason = f"the auditor rejected {sub.name}@{sub.version}"
    session.add(AgentJob(agent="researcher", model="", status="queued",
                         target={"interval": sub.interval, "reason": reason},
                         prompt=research_prompt(sub.interval, reason,
                                                f"Auditor findings on that submission (learn from "
                                                f"them; do not resubmit the same code):\n{findings}")))


def schedule_reviews(session: Session, policy: Policy, cfg: AgentsConfig, now: pd.Timestamp) -> list[AgentJob]:
    """Weekly reviewer jobs for bots that have at least part of their window behind them."""
    if now.weekday() != cfg.review.weekday:
        return []
    out = []
    recent = now - pd.Timedelta(days=6)
    for bot in session.scalars(select(Bot).where(Bot.stage.in_((BotStage.PAPER_PROBATION,
                                                                BotStage.PAPER_PROVEN)))):
        window = policy.windows.get(bot.strategy.interval)
        created = pd.Timestamp(bot.created_at)
        created = created.tz_localize("UTC") if created.tzinfo is None else created
        days = int(np.busday_count(created.date(), now.date()))
        if window is None or days < window.days * cfg.review.min_window_frac:
            continue
        done = session.scalars(select(AgentJob).where(
            AgentJob.agent == "reviewer", AgentJob.bot_id == bot.id,
            AgentJob.created_at >= recent.to_pydatetime())).first()
        if done:
            continue
        last = session.scalars(select(Evaluation).where(Evaluation.bot_id == bot.id)
                               .order_by(Evaluation.id.desc()).limit(1)).first()
        job = AgentJob(agent="reviewer", model="", status="queued", bot_id=bot.id,
                       target={"slug": bot.slug},
                       prompt=f"Weekly review of {bot.slug}. Latest verdict: "
                              f"{last.decision + ': ' + last.rationale if last else 'none yet'}.")
        session.add(job)
        out.append(job)
    session.flush()
    return out


def _commentary(session: Session, job: AgentJob) -> str:
    rep = session.scalars(select(Report).where(Report.date == (job.target or {}).get("date"))).first()
    return rep.commentary if rep else ""


COMMENTARY_WINDOW = pd.Timedelta(hours=12)  # stale commentary is not worth paying for


def schedule_commentary(session: Session, cfg: AgentsConfig, now: pd.Timestamp) -> list[AgentJob]:
    """One supervisor job per fresh daily report; queued ones that went stale are dropped."""
    cutoff = (now - COMMENTARY_WINDOW).to_pydatetime()
    for job in session.scalars(select(AgentJob).where(AgentJob.agent == "supervisor",
                                                      AgentJob.status == "queued",
                                                      AgentJob.created_at < cutoff)):
        job.status = "cancelled"
        job.result = {"error": "report too old for commentary"}
    if not cfg.daily_commentary:
        return []
    rep = session.scalars(select(Report).order_by(Report.date.desc()).limit(1)).first()
    if rep is None or rep.commentary or _utc(rep.created_at) < now - COMMENTARY_WINDOW:
        return []
    for job in session.scalars(select(AgentJob).where(AgentJob.agent == "supervisor")):
        if (job.target or {}).get("date") == rep.date:
            return []
    job = AgentJob(agent="supervisor", model="", status="queued", target={"date": rep.date},
                   prompt=f"Write your commentary on the daily report for {rep.date}.")
    session.add(job)
    session.flush()
    return [job]


def _utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
