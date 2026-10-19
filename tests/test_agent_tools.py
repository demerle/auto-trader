"""Agent tools: role limits, workspace jail, sealed holdout, trial budget, gate, audit."""

import asyncio
import json

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from autotrader.db.models import AgentJob, Base, Submission, ToolCall
from autotrader.evaluator.policy import GateRules, load_policy
from autotrader.mcp_server.server import JobServer
from autotrader.mcp_server.tools import AgentTools, ToolFailure
from autotrader.orchestrator.config import load_agents_config
from tests.test_arena import FakeArchive
from tests.test_sandbox import generated_copy

NOW = pd.Timestamp("2024-07-01", tz="UTC")
LAX = GateRules(min_oos_sharpe=-10, min_dsr=0, max_drawdown=1, min_trades=0, min_excess_sharpe=-10,
                holdout_min_sharpe=-10, holdout_min_excess_sharpe=-10)

LOOKAHEAD = '''
import pandas as pd
from autotrader.strategies.base import Strategy


class Peek(Strategy):
    name = "peek"
    version = "1"
    interval = "1d"
    universe = ("AAA", "BBB", "CCC", "DDD")

    def features(self, df):
        return pd.DataFrame({"next": df["close"].shift(-1)}, index=df.index)

    def on_bar(self, ctx):
        return {s: 0.25 for s, f in ctx.features.items() if f["next"].iloc[-1] > f.index.size * 0}
'''


@pytest.fixture
def env(tmp_path, history):
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as s:
        job = AgentJob(agent="researcher", model="m", prompt="p", target={"interval": "1d"})
        s.add(job)
        s.commit()
    ws = tmp_path / "workspace" / f"job_{job.id}"
    ws.mkdir(parents=True)
    cfg = load_agents_config().model_copy(update={"holdout_days": {"1d": 60}})
    policy = load_policy().model_copy(update={"gate": LAX})

    def tools(role="researcher", job_id=job.id, **cfg_updates):
        c = cfg.model_copy(update=cfg_updates) if cfg_updates else cfg
        return AgentTools(job_id, role, ws, sessions, FakeArchive(history), c, policy, now=NOW)

    return tools, ws, sessions, job.id


def test_roles_only_get_their_tools(env):
    tools, *_ = env
    with pytest.raises(ToolFailure, match="not available"):
        tools("auditor").call("run_backtest", path="x.py")
    assert "audit_verdict" not in {f.__name__ for f in tools().functions()}


def test_workspace_jail(env, tmp_path):
    tools, ws, *_ = env
    outside = generated_copy(tmp_path)
    with pytest.raises(ToolFailure, match="outside your workspace"):
        tools().call("run_backtest", path=str(outside))
    with pytest.raises(ToolFailure, match="outside your workspace"):
        tools().call("run_backtest", path="../../stoch_sandboxed.py")


def test_holdout_is_invisible(env):
    tools, *_ = env
    out = tools().call("fetch_bars", symbol="AAA", interval="1d", last_n=5)
    info = json.loads(out.split("\n\n")[0])
    assert pd.Timestamp(info["last_visible"]) < pd.Timestamp(info["sealed_from"])


def test_backtest_counts_trials_and_budget(env):
    tools, ws, sessions, job_id = env
    path = generated_copy(ws)
    from autotrader.orchestrator.config import PerJob
    t = tools(per_job=PerJob(backtests=2, submissions=3))
    out = json.loads(t.call("run_backtest", path=path.name))
    assert out["trials_used"] == 1 and out["to"] < str(NOW - pd.Timedelta(days=60))
    t.call("run_backtest", path=path.name, params={"max_positions": 2})
    with pytest.raises(ToolFailure, match="trial budget"):
        t.call("run_backtest", path=path.name)
    with sessions() as s:
        calls = s.scalars(select(ToolCall).where(ToolCall.job_id == job_id)).all()
        assert [c.trials for c in calls] == [1, 1, 0] and len(calls[0].sharpes) == 1


def test_lookahead_submission_fails_gate(env):
    tools, ws, sessions, _ = env
    (ws / "peek.py").write_text(LOOKAHEAD)
    out = json.loads(tools().call("submit_strategy", path="peek.py", rationale="x"))
    assert out["verdict"] == "FAILED the gate"
    assert any("lookahead" in r for r in out["reasons"])


def test_submit_audit_flow(env):
    tools, ws, sessions, job_id = env
    path = generated_copy(ws)
    tools().call("run_backtest", path=path.name)
    out = json.loads(tools().call("submit_strategy", path=path.name, rationale="momentum"))
    assert out["verdict"].startswith("PASSED")
    with sessions() as s:
        sub = s.get(Submission, out["submission_id"])
        assert sub.status == "audit_pending" and sub.holdout_metrics and sub.n_trials >= 2
        auditor = AgentJob(agent="auditor", model="m", prompt="p", submission_id=sub.id)
        s.add(auditor)
        s.commit()
    # Same name@version cannot be submitted twice.
    with pytest.raises(ToolFailure, match="already exists"):
        tools().call("submit_strategy", path=path.name, rationale="again")
    a = tools("auditor", job_id=auditor.id)
    assert "## Source" in a.call("get_submission", submission_id=sub.id)
    with pytest.raises(ToolFailure, match="you are auditing"):
        a.call("get_submission", submission_id=sub.id + 1)
    a.call("audit_verdict", submission_id=sub.id, approve=True, findings="sound")
    with sessions() as s:
        assert s.get(Submission, sub.id).audit["approve"] is True


def test_wrong_interval_for_slot(env):
    tools, ws, sessions, job_id = env
    with sessions() as s:
        s.get(AgentJob, job_id).target = {"interval": "1h"}
        s.commit()
    path = generated_copy(ws)
    with pytest.raises(ToolFailure, match="slot is for a 1h"):
        tools().call("submit_strategy", path=path.name, rationale="x")


def test_mcp_server_over_http(env):
    from mcp import Client

    tools, *_ = env

    async def main(url):
        async with Client(url) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            res = await client.call_tool("get_universes", {})
            bad = await client.call_tool("fetch_bars", {"symbol": "NOPE", "interval": "1d"})
            return names, res.content[0].text, bad

    with JobServer(tools()) as server:
        names, text, bad = asyncio.run(main(server.url))
    assert "submit_strategy" in names and "audit_verdict" not in names
    assert "liquid_etfs" in text
    assert bad.is_error
