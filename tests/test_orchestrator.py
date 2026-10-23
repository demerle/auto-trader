"""Orchestrator end to end with a fake opencode that drives the real MCP server over HTTP."""

import asyncio
import json

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from autotrader.db.models import AgentJob, Base, Bot, BotStage, StrategyVersion, Submission, ToolCall
from autotrader.evaluator.policy import load_policy
from autotrader.orchestrator import budget, loop, slots
from autotrader.orchestrator.config import load_agents_config
from autotrader.orchestrator.runner import RunResult, opencode_config, parse_events
from autotrader.strategies import registry
from tests.test_agent_tools import LAX
from tests.test_arena import FakeArchive
from tests.test_sandbox import BUILTIN

NOW = pd.Timestamp("2024-07-01 12:00", tz="UTC")


class Archive(FakeArchive):
    """SPY (used to size researchable history) stands in as AAA."""

    def get(self, symbol, interval, start=None, end=None, refresh=True):
        return super().get("AAA" if symbol == "SPY" else symbol, interval, start, end, refresh)

    def load(self, symbol, interval):
        return self.bars["AAA"]


def strategy_source(name: str) -> str:
    src = (BUILTIN / "stoch_momentum.py").read_text()
    return (src.replace('universe = "core"', 'universe = ("AAA", "BBB", "CCC", "DDD")')
               .replace('name = "stoch_momentum"', f'name = "{name}"'))


async def _drive(url: str, calls: list[tuple[str, dict]]) -> list[str]:
    from mcp import Client

    out = []
    async with Client(url) as client:
        for name, args in calls:
            res = await client.call_tool(name, args)
            out.append(res.content[0].text)
    return out


def fake_opencode(approve: bool = True, fail_first_researcher: bool = False):
    seen = {"researcher": 0}

    def run(job, model, workspace, mcp_url, timeout, log_path):
        log_path.write_text("")
        if job.agent == "researcher":
            seen["researcher"] += 1
            if fail_first_researcher and seen["researcher"] == 1:
                return RunResult(1, False, cost_usd=0.2, error="boom")
            (workspace / "strategy.py").write_text(strategy_source(f"agent_mom_{job.id}"))
            asyncio.run(_drive(mcp_url, [
                ("run_backtest", {"path": "strategy.py"}),
                ("submit_strategy", {"path": "strategy.py", "rationale": "momentum persists"}),
                ("write_note", {"topic": "mom", "body": "tried momentum"})]))
            return RunResult(0, False, cost_usd=1.25, session_id="ses_r")
        if job.agent == "auditor":
            asyncio.run(_drive(mcp_url, [
                ("get_submission", {"submission_id": job.submission_id}),
                ("audit_verdict", {"submission_id": job.submission_id, "approve": approve,
                                   "findings": "ok" if approve else "lookahead on line 3"})]))
            return RunResult(0, False, cost_usd=0.3)
        raise AssertionError(job.agent)
    return run


@pytest.fixture
def arena(tmp_path, history, monkeypatch):
    monkeypatch.setenv("AT_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("AT_DATA_DIR", str(tmp_path / "data"))
    from autotrader.config import get_settings

    get_settings.cache_clear()
    gen = tmp_path / "generated"
    gen.mkdir()
    monkeypatch.setattr(registry, "GENERATED_DIR", gen)
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    cfg = load_agents_config().model_copy(update={
        "holdout_days": {"1d": 60}, "research_intervals": ["1d"], "auto_commit": False})
    policy = load_policy().model_copy(update={"gate": LAX, "slots": 1})
    yield sessions, Archive(history), cfg, policy, gen
    get_settings.cache_clear()


def run_ticks(arena, runner, n):
    loop._HISTORY.clear()
    sessions, archive, cfg, policy, _ = arena
    out = []
    for _ in range(n):
        out += loop.tick(sessions, archive, cfg, policy, runner=runner, now=NOW)
    return out


def test_open_slot_is_researched_audited_and_deployed(arena):
    sessions, archive, cfg, policy, gen = arena
    lines = run_ticks(arena, fake_opencode(), 3)
    assert any("deployed" in line for line in lines)
    with sessions() as s:
        bot = s.scalars(select(Bot)).one()
        assert bot.stage == BotStage.PAPER_PROBATION and bot.capital == policy.base_capital
        sv = s.get(StrategyVersion, bot.strategy_id)
        assert sv.author.startswith("job-") and sv.backtest_metrics["gate_passed"] == 1.0
        sub = s.scalars(select(Submission)).one()
        assert sub.status == "deployed"
        jobs = s.scalars(select(AgentJob).order_by(AgentJob.id)).all()
        assert [(j.agent, j.status) for j in jobs] == [("researcher", "succeeded"),
                                                       ("auditor", "succeeded")]
        assert budget.spend(s, NOW)["week"] == pytest.approx(1.55)
        assert s.scalars(select(ToolCall)).first() is not None
    # The deployed file is registered as a sandboxed strategy.
    cls = registry.get(sv.name, sv.version)
    assert registry.is_sandboxed(cls) and cls.code_sha == sv.code_sha
    assert (gen / f"{sv.name}_v1.py").exists()
    # Slot full: nothing more to do.
    assert run_ticks(arena, fake_opencode(), 1)[-1] == "nothing to run"


def test_rejected_audit_requeues_research_with_findings(arena):
    sessions, *_ = arena
    run_ticks(arena, fake_opencode(approve=False), 2)
    with sessions() as s:
        assert s.scalars(select(Submission)).one().status == "rejected"
        queued = s.scalars(select(AgentJob).where(AgentJob.status == "queued")).one()
        assert queued.agent == "researcher" and "lookahead on line 3" in queued.prompt
        assert s.scalars(select(Bot)).first() is None


def test_failed_job_retries_on_the_same_model(arena):
    sessions, archive, cfg, *_ = arena
    run_ticks(arena, fake_opencode(fail_first_researcher=True), 2)
    with sessions() as s:
        r = s.scalars(select(AgentJob).where(AgentJob.agent == "researcher")).first()
        assert r.attempts == 2 and r.status == "succeeded"
        assert r.model == cfg.model


def test_budget_blocks_jobs(arena):
    sessions, archive, cfg, policy, _ = arena
    with sessions() as s:
        s.add(AgentJob(agent="researcher", model="m", prompt="x", status="succeeded",
                       cost_usd=5.5, started_at=(NOW - pd.Timedelta(hours=1)).to_pydatetime()))
        s.commit()
    out = run_ticks(arena, fake_opencode(), 1)
    assert any("waiting for budget" in line for line in out)
    with sessions() as s:
        assert budget.estimate(s, "researcher", cfg) == pytest.approx(5.5)


def test_slot_accounting_and_interval_choice(arena):
    sessions, archive, cfg, policy, _ = arena
    cfg = cfg.model_copy(update={"research_intervals": ["1d", "1h"]})
    with sessions() as s:
        sv = StrategyVersion(name="x", version="1", interval="1d", module="m", code_sha="s")
        s.add(sv)
        s.flush()
        s.add(Bot(slug="x", strategy_id=sv.id, stage=BotStage.PAPER_PROBATION))
        s.flush()
        assert slots.open_slots(s, policy.model_copy(update={"slots": 3})) == 2
        assert slots.choose_interval(s, cfg, lambda iv: 100) == "1h"
        assert slots.choose_interval(s, cfg, lambda iv: 100 if iv == "1d" else 10) == "1d"


def test_parse_events_takes_last_cost_per_id():
    lines = [json.dumps({"type": "step", "sessionID": "ses_1",
                         "part": {"id": "p1", "cost": 0.1, "tokens": {"input": 10, "output": 2}}}),
             json.dumps({"type": "step", "part": {"id": "p1", "cost": 0.15,
                                                  "tokens": {"input": 12, "output": 3}}}),
             json.dumps({"type": "step", "part": {"id": "p2", "cost": 0.05,
                                                  "tokens": {"input": 5, "cache": {"read": 7}}}}),
             "not json",
             json.dumps({"type": "error", "error": {"message": "expired"}})]
    r = parse_events(lines)
    assert r.cost_usd == pytest.approx(0.2) and r.session_id == "ses_1" and r.error == "expired"
    assert r.tokens == {"input": 17, "output": 3, "cache_read": 7}


def test_opencode_config_denies_shell_and_outside_files():
    cfg = opencode_config("auditor", "opencode-go/x", "http://127.0.0.1:1/t/mcp")
    rules = {(r["action"], r["resource"]): r["effect"] for r in cfg["permissions"]}
    assert rules[("shell", "*")] == "deny" and rules[("external_directory", "*")] == "deny"
    assert rules[("edit", "*")] == "deny" and rules[("webfetch", "*")] == "deny"
    assert cfg["mcp"]["autotrader"]["type"] == "remote" and cfg["mcp"]["autotrader"]["timeout"] >= 600_000


def test_auth_error_pauses_without_burning_attempts(arena):
    sessions, *_ = arena

    def expired(job, model, workspace, mcp_url, timeout, log_path):
        log_path.write_text(json.dumps({"type": "error", "error": {
            "type": "provider.auth", "message": "This API key has expired."}}) + "\n")
        res = parse_events(log_path.read_text().splitlines())
        res.exit_code = 1
        return res

    out = run_ticks(arena, expired, 2)
    assert out[-1].startswith("paused:")
    with sessions() as s:
        job = s.scalars(select(AgentJob)).one()
        assert job.status == "queued" and job.attempts == 0


def test_only_the_configured_model_is_ever_used(arena):
    sessions, archive, cfg, *_ = arena
    assert cfg.model == "opencode-go/glm-5.3-flash"
    assert {cfg.model_for(r, a) for r in ("researcher", "auditor", "reviewer") for a in (1, 2, 3)} \
        == {cfg.model}
    oc = opencode_config("researcher", cfg.model, "http://127.0.0.1:1/t/mcp")
    assert oc["model"] == oc["small_model"] == oc["agents"]["researcher"]["model"] == cfg.model
    run_ticks(arena, fake_opencode(), 3)
    with sessions() as s:
        assert {j.model for j in s.scalars(select(AgentJob))} == {cfg.model}


def test_unknown_model_keys_are_rejected():
    from pydantic import ValidationError

    from autotrader.orchestrator.config import AgentsConfig

    with pytest.raises(ValidationError):
        AgentsConfig.model_validate({"model": "opencode-go/glm-5.3-flash",
                                     "models": {"researcher": "opencode-go/kimi-k3"}})


def quota_runner(window="weekly"):
    def run(job, model, workspace, mcp_url, timeout, log_path):
        log_path.write_text("")
        return RunResult(1, False, error="Go usage limit exceeded", error_type="provider.quota",
                         quota_window=window)
    return run


def test_quota_error_holds_all_jobs_and_alerts_once(arena, monkeypatch):
    sessions, archive, cfg, policy, _ = arena
    alerts = []
    monkeypatch.setattr(loop, "alert", lambda title, msg, **kw: alerts.append(title))
    loop._HISTORY.clear()
    out = loop.tick(sessions, archive, cfg, policy, runner=quota_runner(), now=NOW)
    assert any("holding agent jobs" in line for line in out)
    with sessions() as s:
        job = s.scalars(select(AgentJob)).one()
        assert job.status == "queued" and job.attempts == 0 and job.started_at is None  # not the job's fault
    assert loop.tick(sessions, archive, cfg, policy, runner=quota_runner(), now=NOW + pd.Timedelta(hours=1))[0] \
        .startswith("waiting for Go quota (weekly limit)")
    later = NOW + pd.Timedelta(hours=cfg.budget.quota_retry_hours["weekly"] + 0.1)
    loop.tick(sessions, archive, cfg, policy, runner=quota_runner(), now=later)  # probe: still out
    assert alerts == ["Agents waiting for Go quota"]
    out = loop.tick(sessions, archive, cfg, policy, runner=fake_opencode(),
                    now=later + pd.Timedelta(hours=cfg.budget.quota_retry_hours["weekly"] + 0.1))
    assert alerts == ["Agents waiting for Go quota", "Agents running again"]
    with sessions() as s:
        assert s.scalars(select(AgentJob).where(AgentJob.agent == "researcher")).one().status == "succeeded"


def test_daily_job_cap_blocks_jobs(arena):
    sessions, archive, cfg, policy, _ = arena
    with sessions() as s:
        for h in range(cfg.budget.max_jobs_per_day):
            s.add(AgentJob(agent="auditor", model="m", prompt="x", status="succeeded", cost_usd=0.0,
                           started_at=(NOW - pd.Timedelta(hours=h + 1)).to_pydatetime()))
        s.commit()
        chk = budget.check(s, "researcher", cfg, NOW)
        assert not chk.ok and "jobs started in the last 24 hours" in chk.reason
        assert budget.check(s, "researcher", cfg, NOW + pd.Timedelta(hours=24)).ok


def test_failed_research_alerts_once_and_cools_down_before_refilling(arena, monkeypatch):
    from autotrader.orchestrator import pipeline

    sessions, archive, cfg, policy, _ = arena
    alerts = []
    monkeypatch.setattr(pipeline, "alert", lambda title, msg, **kw: alerts.append(title))

    def broken(job, model, workspace, mcp_url, timeout, log_path):
        log_path.write_text("")
        return RunResult(1, False, error="PermissionError: nope")

    run_ticks(arena, broken, 3)  # two attempts, then failed for good
    with sessions() as s:
        jobs = s.scalars(select(AgentJob)).all()
        assert [(j.status, j.attempts) for j in jobs] == [("failed", 2)]  # no replacement queued
        assert slots.recently_failed(s, "researcher", cfg)
        s.add(AgentJob(agent="researcher", model="m", prompt="x", status="failed", attempts=2,
                       result={"error": "PermissionError: nope", "attempt": 2}))
        s.commit()
    run_ticks(arena, broken, 1)
    assert len(alerts) == 1  # the same failure again is only logged


def test_parse_events_reads_the_exhausted_go_limit():
    body = json.dumps({"type": "error", "error": {"type": "GoUsageLimitError", "message": "Go usage limit exceeded"},
                       "metadata": {"limitName": "weekly"}})
    lines = [json.dumps({"type": "error", "sessionID": "ses_q", "error": {
        "type": "provider.quota", "message": "Go usage limit exceeded", "status": 429, "response": {"body": body}}})]
    res = parse_events(lines)
    assert res.error_type == "provider.quota" and res.quota_window == "weekly"
