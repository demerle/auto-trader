"""Dashboard (login, CSRF, pages, actions), daily report, Discord posting, supervisor role."""

import json
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from autotrader import alerts as alerts_mod
from autotrader.alerts import _deliver as REAL_DELIVER
from autotrader.alerts import post_discord as REAL_POST_DISCORD
from autotrader.alerts import split_text
from autotrader.db.models import (
    AgentJob, AuditLog, Bot, BotStage, Evaluation, Report, SystemState, utcnow,
)
from autotrader.evaluator.policy import load_policy
from autotrader.orchestrator.config import load_agents_config
from tests.test_arena import make_db
from tests.test_evaluator import _seed_equity

TOKEN = "correct-horse-battery-staple"


class NoArchive:
    def get(self, *a, **k):
        raise FileNotFoundError("no bars in tests")

    def load(self, *a, **k):
        raise FileNotFoundError("no bars in tests")


class FakeArena:
    def __init__(self):
        self.killed, self.unkilled = [], 0

    def kill(self, reason):
        self.killed.append(reason)
        return "killed"

    def unkill(self):
        self.unkilled += 1


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    from autotrader.config import get_settings

    monkeypatch.setenv("AT_DASHBOARD_TOKEN", TOKEN)
    monkeypatch.setenv("AT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AT_DISCORD_WEBHOOK", "https://discord.test/api/webhooks/1/x")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db(tmp_path, settings_env):
    sessions, (b0, b1) = make_db(tmp_path, n_bots=2)
    _seed_equity(sessions, b0, [0.004, -0.001, 0.003] * 4)
    _seed_equity(sessions, b1, [-0.002, 0.001] * 6)
    log_dir = tmp_path / "data" / "agent_logs"
    log_dir.mkdir(parents=True)
    events = [{"type": "text", "part": {"id": "p1", "type": "text", "text": "Looking at **graveyard**"}},
              {"type": "tool_use", "part": {"id": "p2", "type": "tool", "tool": "autotrader_list_bots",
                                            "state": {"status": "completed", "input": {},
                                                      "output": "[]"}}},
              {"type": "error", "error": {"type": "provider.x", "message": "<b>boom</b>"}}]
    (log_dir / "job_1.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    with sessions() as s:
        s.add(AgentJob(agent="researcher", model="opencode-go/glm-5.3-flash", status="succeeded",
                       prompt="research", target={"interval": "1d"}, cost_usd=0.42,
                       started_at=utcnow(), finished_at=utcnow(),
                       transcript_path=str(log_dir / "job_1.jsonl")))
        s.add(AgentJob(agent="researcher", model="", status="queued", prompt="more",
                       target={"interval": "1h"}))
        s.add(Report(date="2026-01-20", body_md="## League\n- **b0** up", data={}))
        s.commit()
    return sessions


@pytest.fixture
def client(db):
    from autotrader.api import deps
    from autotrader.api.app import create_app

    arena = FakeArena()
    app = create_app()
    app.dependency_overrides.update({
        deps.get_sessions: lambda: db,
        deps.get_archive: lambda: NoArchive(),
        deps.get_agents_cfg: lambda: load_agents_config(),
        deps.get_policy: lambda: load_policy(),
        deps.arena_factory: lambda: (lambda: arena),
    })
    with TestClient(app, base_url="http://127.0.0.1") as c:
        c.arena = arena
        yield c


def login(c) -> None:
    r = c.post("/login", data={"token": TOKEN, "next": "/"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"


def csrf(c, path="/agents") -> str:
    return re.search(r'name="csrf" value="([0-9a-f]+)"', c.get(path).text).group(1)


PAGES = ["/", "/league", "/bots/b0", "/graveyard", "/agents", "/agents/jobs/1", "/reports",
         "/reports/2026-01-20", "/audit"]


def test_every_page_needs_login(client):
    for path in PAGES:
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login"), path
    assert client.get("/health").json() == {"ok": True}
    client.cookies.set("at_session", "forged.9999999999.deadbeef")
    assert client.get("/", follow_redirects=False).status_code == 303


def test_pages_render_after_login(client):
    login(client)
    for path in PAGES:
        r = client.get(path)
        assert r.status_code == 200, path
        assert "Content-Security-Policy" in r.headers and r.headers["X-Frame-Options"] == "DENY"
    league = client.get("/league").text
    assert "b0" in league and "b1" in league
    job = client.get("/agents/jobs/1").text
    assert "<strong>graveyard</strong>" in job and "autotrader_list_bots" in job
    assert "&lt;b&gt;boom&lt;/b&gt;" in job  # agent output is escaped
    assert client.get("/bots/nope").status_code == 404


def test_bad_token_and_open_redirect(client):
    r = client.post("/login", data={"token": "nope", "next": "/"}, follow_redirects=False)
    assert r.status_code == 303 and "error=1" in r.headers["location"]
    assert "at_session" not in client.cookies
    r = client.post("/login", data={"token": TOKEN, "next": "//evil.example/x"}, follow_redirects=False)
    assert r.headers["location"] == "/"


def test_actions_need_csrf_and_are_audited(client, db, sent_alerts):
    login(client)
    r = client.post("/actions/agents/pause", data={}, follow_redirects=False)
    assert r.status_code == 403
    r = client.post("/actions/agents/pause", data={"csrf": "0" * 32}, follow_redirects=False)
    assert r.status_code == 403
    r = client.post("/actions/agents/pause", data={"csrf": csrf(client)}, follow_redirects=False)
    assert r.status_code == 303
    with db() as s:
        assert s.get(SystemState, "agents_paused").value["reason"] == "paused from the dashboard"
        assert s.scalars(select(AuditLog)).one().action == "pause_agents"
    assert any(t == "Agents paused" for t, _ in sent_alerts)
    client.post("/actions/agents/resume", data={"csrf": csrf(client)})
    with db() as s:
        assert s.get(SystemState, "agents_paused") is None


def test_logged_out_post_is_rejected(client):
    r = client.post("/actions/kill", data={"csrf": "x", "confirm": "KILL"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert client.arena.killed == []


def test_kill_needs_typed_confirmation(client, db):
    login(client)
    token = csrf(client, "/")
    r = client.post("/actions/kill", data={"csrf": token, "reason": "test", "confirm": ""},
                    follow_redirects=False)
    assert r.status_code == 400 and client.arena.killed == []
    r = client.post("/actions/kill", data={"csrf": token, "reason": "test", "confirm": "kill"},
                    follow_redirects=False)
    assert r.status_code == 303 and client.arena.killed == ["dashboard: test"]
    with db() as s:
        assert s.scalars(select(AuditLog)).one().action == "kill"


def test_pause_bot_and_queue_and_cancel(client, db):
    login(client)
    token = csrf(client)
    client.post("/actions/bots/b0/pause", data={"csrf": token})
    with db() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == "b0")).one()
        assert bot.paused_until is not None
    client.post("/actions/bots/b0/unpause", data={"csrf": token})
    client.post("/actions/agents/queue", data={"csrf": token, "interval": "15m", "note": "try VWAP"})
    assert client.post("/actions/agents/queue", data={"csrf": token, "interval": "3m"},
                       follow_redirects=False).status_code == 400
    with db() as s:
        assert s.scalars(select(Bot).where(Bot.slug == "b0")).one().paused_until is None
        job = s.scalars(select(AgentJob).where(AgentJob.target["interval"].as_string() == "15m")).one()
        assert "try VWAP" in job.prompt
    client.post(f"/actions/agents/jobs/{job.id}/cancel", data={"csrf": token})
    assert client.post("/actions/agents/jobs/1/cancel", data={"csrf": token},
                       follow_redirects=False).status_code == 400  # not queued
    with db() as s:
        assert s.get(AgentJob, job.id).status == "cancelled"
        assert [a.action for a in s.scalars(select(AuditLog).order_by(AuditLog.id))] == [
            "pause_bot", "unpause_bot", "queue_research", "cancel_job"]


def test_refuses_to_start_without_token(monkeypatch, settings_env):
    from autotrader.api.app import create_app
    from autotrader.config import get_settings

    monkeypatch.setenv("AT_DASHBOARD_TOKEN", "")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="AT_DASHBOARD_TOKEN"):
        with TestClient(create_app()):
            pass


# ----- views, report, Discord -----

def test_league_ranks_by_sharpe(db):
    from autotrader import views
    from autotrader.evaluator.record import build_record

    with db() as s:
        rows = views.league(s, None)
        assert [r["slug"] for r in rows] == ["b0", "b1"]
        rec = build_record(s, s.scalars(select(Bot).where(Bot.slug == "b0")).one())
        assert rows[0]["sharpe"] == pytest.approx(rec.metrics["sharpe"])
        assert rows[0]["days"] == rec.days


def test_daily_report_contents_and_single_post(db, sent_alerts):
    from autotrader.reports import daily

    day = "2026-01-20"
    start, _ = daily.et_day_bounds(day)
    at = (start + pd.Timedelta(hours=17)).to_pydatetime()
    with db() as s:
        b1 = s.scalars(select(Bot).where(Bot.slug == "b1")).one()
        b1.stage = BotStage.ERADICATED
        s.add(Evaluation(bot_id=b1.id, ts=at, decision="eradicate", rationale="PSR 0.2 < 0.5"))
        s.add(Evaluation(bot_id=b1.id - 1, ts=at, decision="keep", rationale="fine"))
        s.add(Bot(slug="newbie", strategy_id=b1.strategy_id, stage=BotStage.PAPER_PROBATION, created_at=at))
        s.add(AgentJob(agent="auditor", model="m", status="succeeded", prompt="p", started_at=at,
                       finished_at=at, cost_usd=0.25))
        s.query(Report).delete()
        s.commit()
    now = start + pd.Timedelta(hours=20)
    rep = daily.publish_daily_report(db, NoArchive(), day, now=now)
    assert "**eradicate** b1: PSR 0.2 < 0.5" in rep.body_md
    assert "newbie" in rep.body_md and "Spent today $0.25" in rep.body_md
    assert rep.data["decisions"] == {"eradicate": 1, "keep": 1} and rep.posted_at is not None
    posts = [p for t, p in sent_alerts if t == "discord"]
    assert len(posts) == 1 and posts[0]["embeds"][0]["url"].endswith(f"/reports/{day}")
    daily.publish_daily_report(db, NoArchive(), day, now=now)  # rebuilt, not re-posted
    assert len([p for t, p in sent_alerts if t == "discord"]) == 1
    with db() as s:
        assert s.scalars(select(Report)).one().date == day


def test_discord_split_respects_limits(sent_alerts, settings_env):
    body = "\n".join(f"- line {i} " + "x" * 90 for i in range(200)) + "\n" + "y" * 9000
    chunks = split_text(body)
    assert all(len(c) <= alerts_mod.DISCORD_CHUNK for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == body.replace("\n", "")
    assert alerts_mod.post_report("T" * 300, body, "https://x.test/r")
    embeds = [p["embeds"][0] for t, p in sent_alerts if t == "discord"]
    assert len(embeds) == len(chunks) and len(embeds[0]["title"]) == 256
    assert all(len(e["description"]) <= alerts_mod.DISCORD_EMBED_MAX for e in embeds)
    assert all(sum(len(str(v)) for v in e.values()) < 6000 for e in embeds)


def test_discord_waits_out_rate_limit(monkeypatch):
    calls, sleeps = [], []

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def json(self):
            return {"retry_after": 0.5}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    codes = iter([429, 204])
    monkeypatch.setattr(alerts_mod.httpx, "post", lambda url, json, timeout: calls.append(json) or Resp(next(codes)))
    monkeypatch.setattr(alerts_mod.time, "sleep", sleeps.append)
    REAL_POST_DISCORD("https://discord.test/hook", {"content": "hi"})
    assert len(calls) == 2 and sleeps == [0.5]


def test_long_alert_is_truncated_for_discord(settings_env, sent_alerts):
    REAL_DELIVER("t", "m" * 5000, "default")  # posts through the (captured) post_discord
    (payload,) = [p for t, p in sent_alerts if t == "discord"]
    assert len(payload["content"]) == alerts_mod.DISCORD_CONTENT_MAX


# ----- supervisor -----

def _supervisor_job(db, day="2026-01-20") -> int:
    with db() as s:
        job = AgentJob(agent="supervisor", model="m", status="running", prompt="p", target={"date": day})
        s.add(job)
        s.commit()
        return job.id


def test_supervisor_tools_and_commentary(db, tmp_path):
    from autotrader.mcp_server.tools import ROLE_TOOLS, AgentTools, ToolFailure
    from autotrader.orchestrator.runner import permissions

    job_id = _supervisor_job(db)
    tools = AgentTools(job_id, "supervisor", tmp_path, db, NoArchive(), load_agents_config(), load_policy())
    assert [f.__name__ for f in tools.functions()] == list(ROLE_TOOLS["supervisor"])
    with pytest.raises(ToolFailure):
        tools.call("run_backtest", path="x.py")
    assert "**b0** up" in tools.call("get_report")
    with pytest.raises(ToolFailure, match="limit is 1500"):
        tools.call("write_commentary", text="x" * 1501)
    assert "saved" in tools.call("write_commentary", text="b0 leads; too early to tell.")
    with pytest.raises(ToolFailure, match="already"):
        tools.call("write_commentary", text="again")
    with db() as s:
        assert s.scalars(select(Report)).one().commentary == "b0 leads; too early to tell."
    rules = permissions("supervisor")
    assert {"action": "edit", "resource": "*", "effect": "deny"} in rules
    assert {"action": "webfetch", "resource": "*", "effect": "deny"} in rules


def test_commentary_is_scheduled_once_and_posted(db, sent_alerts):
    from autotrader.orchestrator import pipeline

    cfg = load_agents_config()
    now = pd.Timestamp.now(tz="UTC")
    with db() as s:
        (job,) = pipeline.schedule_commentary(s, cfg, now)
        assert job.target == {"date": "2026-01-20"}
        assert pipeline.schedule_commentary(s, cfg, now) == []
        s.commit()
        assert pipeline.schedule_commentary(s, cfg, now + pd.Timedelta(hours=13)) == []
        s.commit()
        assert s.get(AgentJob, job.id).status == "cancelled"  # went stale
        s.scalars(select(Report)).one().commentary = "Watch b1."
        notes = pipeline.settle(s, load_policy(), cfg)
        assert "posted supervisor commentary for 2026-01-20" in notes
        assert s.scalars(select(Report)).one().commentary_posted_at is not None
        assert not any("posted" in n for n in pipeline.settle(s, load_policy(), cfg))
    posts = [p for t, p in sent_alerts if t == "discord"]
    assert len(posts) == 1 and posts[0]["embeds"][0]["description"] == "Watch b1."


def test_render_helpers_escape():
    from autotrader.api.render import line_chart, markdown

    html = markdown("# Title\n- **bold** <script>x</script>\n1. one")
    assert "<h2>Title</h2>" in html and "<strong>bold</strong>" in html and "&lt;script&gt;" in html
    svg = line_chart([("a<b", [("2026-01-01", 1.0), ("2026-01-02", 2.0)])])
    assert "<svg" in svg and "a&lt;b" in svg
    assert "Not enough data" in line_chart([("a", [("2026-01-01", 1.0)])])
