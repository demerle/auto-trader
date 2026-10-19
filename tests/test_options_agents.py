"""Gauntlet agents: role tools, the entry test through MCP, audit, deploy into a seat."""

import json

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from autotrader.db.models import OPTIONS, AgentJob, Base, Bot, Submission, WeeklyResult
from autotrader.evaluator.policy import load_policy
from autotrader.mcp_server.tools import AgentTools, ToolFailure
from autotrader.options.config import GauntletConfig, SmokeCfg
from autotrader.orchestrator import pipeline
from autotrader.orchestrator.config import load_agents_config
from autotrader.orchestrator.runner import permissions
from tests.test_options import _five_minute_bars

NOW = pd.Timestamp("2026-09-28 12:00", tz="UTC")
GCFG = GauntletConfig(underlyings={"etfs": ["SPY"]}, smoke=SmokeCfg(sessions=4, min_trades=2))

TOY = '''
from autotrader.options.strategy import Leg, OptionStrategy, Structure


class {cls}(OptionStrategy):
    name = "{name}"
    version = "1"
    underlyings = ("SPY",)
    description = "test"
    thesis = "buys a call every morning, sells it by lunch"

    def on_tick(self, ctx):
        c = ctx.chains.get("SPY")
        if c is None or ctx.minutes_since_open > 150:
            return {{}}
        pos = ctx.positions.get("t")
        if pos is not None:
            return {{"t": Structure(pos.legs, pos.qty)}}
        if ctx.closed_today:
            return {{}}
        row = c.atm(c.expiry(0, min_dte=1), "C")
        return {{"t": Structure((Leg(row["symbol"], {ratio}),), 1, "toy")}}
'''


class Archive:
    def __init__(self):
        b5 = _five_minute_bars((NOW - pd.Timedelta(days=50), NOW - pd.Timedelta(days=1)))
        d = b5.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                   "volume": "sum"}).dropna()
        vix = d.assign(close=16.0)
        self.data = {("SPY", "5m"): b5, ("SPY", "1d"): d, ("^VIX", "1d"): vix}

    def get(self, symbol, interval, start=None, end=None, refresh=True):
        df = self.data.get((symbol, interval))
        if df is None:
            return pd.DataFrame()
        return df if start is None else df[df.index >= start]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from autotrader.options import registry

    engine = create_engine(f"sqlite:///{tmp_path}/a.db")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as s:
        job = AgentJob(agent="options_researcher", model="m", prompt="p", target={"group": OPTIONS})
        s.add(job)
        s.commit()
    ws = tmp_path / "ws"
    ws.mkdir()
    gen = tmp_path / "generated"
    gen.mkdir()
    monkeypatch.setattr(registry, "GENERATED_DIR", gen)
    monkeypatch.setattr(AgentTools, "_gcfg", lambda self: GCFG)
    monkeypatch.setattr("autotrader.options.config.load_gauntlet_config", lambda path=None: GCFG)
    cfg = load_agents_config().model_copy(update={"auto_commit": False})

    def tools(role="options_researcher", job_id=job.id):
        return AgentTools(job_id, role, ws, sessions, Archive(), cfg, load_policy(), now=NOW)

    return tools, ws, sessions, cfg, gen


def test_options_roles_tools_and_permissions(env):
    tools, *_ = env
    names = {f.__name__ for f in tools().functions()}
    assert {"submit_options_strategy", "run_options_backtest", "get_options_graveyard"} <= names
    assert "run_backtest" not in names and "audit_verdict" not in names
    with pytest.raises(ToolFailure, match="not available"):
        tools("options_auditor").call("submit_options_strategy", path="x.py", rationale="r")
    rules = {(r["action"], r["effect"]) for r in permissions("options_auditor")}
    assert ("edit", "deny") in rules and ("webfetch", "deny") in rules
    assert ("webfetch", "deny") not in {(r["action"], r["effect"]) for r in permissions("options_researcher")}
    assert "Hard rules" in tools().call("get_options_api")


def test_entry_test_audit_and_deploy_into_a_seat(env):
    tools, ws, sessions, cfg, gen = env
    (ws / "naked.py").write_text(TOY.format(cls="Naked", name="naked_calls", ratio=-1))
    bad = json.loads(tools().call("submit_options_strategy", path="naked.py", rationale="r"))
    assert bad["verdict"].startswith("FAILED") and "opened 0" in bad["reasons"][0]

    (ws / "toy.py").write_text(TOY.format(cls="Toy", name="morning_calls", ratio=1))
    good = json.loads(tools().call("submit_options_strategy", path="toy.py", rationale="r"))
    assert good["verdict"].startswith("PASSED"), good
    with sessions() as s:
        pipeline.settle(s, load_policy(), cfg)
        audit = s.scalars(select(AgentJob).where(AgentJob.agent == "options_auditor")).one()
        sub = s.get(Submission, good["submission_id"])
        assert sub.status == "audit_pending" and audit.submission_id == sub.id
        s.commit()
    view = tools("options_auditor", audit.id).call("get_submission", submission_id=sub.id)
    assert "options gauntlet bot" in view and "morning_calls" in view
    tools("options_auditor", audit.id).call("audit_verdict", submission_id=sub.id, approve=True, findings="ok")
    with sessions() as s:
        notes = pipeline.settle(s, load_policy(), cfg)
        s.commit()
        bot = s.scalars(select(Bot).where(Bot.group == OPTIONS)).one()
        assert any("seated" in n for n in notes)
        assert bot.account == "paper-options" and bot.starting_capital == 5000 and bot.strategy.interval == "opt"
        assert s.scalars(select(WeeklyResult).where(WeeklyResult.bot_id == bot.id)).one().start_equity == 5000
    assert (gen / "morning_calls_v1.py").exists()


def test_rejected_options_submission_requeues_research(env):
    tools, ws, sessions, cfg, gen = env
    (ws / "toy.py").write_text(TOY.format(cls="Toy", name="morning_calls", ratio=1))
    good = json.loads(tools().call("submit_options_strategy", path="toy.py", rationale="r"))
    with sessions() as s:
        pipeline.settle(s, load_policy(), cfg)
        s.get(Submission, good["submission_id"]).audit = {"approve": False, "findings": "no exits"}
        pipeline.settle(s, load_policy(), cfg)
        s.commit()
        jobs = s.scalars(select(AgentJob).where(AgentJob.agent == "options_researcher",
                                                AgentJob.status == "queued")).all()
        assert any("no exits" in j.prompt for j in jobs)
