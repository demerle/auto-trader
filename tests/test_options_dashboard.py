"""Gauntlet pages: behind the login, rendered with live data, actions CSRF-checked and audited."""

import pandas as pd
from sqlalchemy import select

from autotrader.db.models import (
    OPTIONS, AuditLog, Bot, BotStage, EquitySnapshot, Order, OrderAllocation, StrategyVersion, WeeklyResult,
    utcnow,
)
from autotrader.options.competition import week_of
from tests.test_dashboard import TOKEN, client, csrf, db, login, settings_env  # noqa: F401

GAUNTLET_PAGES = ["/options", "/options/trades", "/options/bots/g0", "/options/weeks", "/options/graveyard",
                  "/options/cull-preview"]


def _seed_gauntlet(db):
    now = pd.Timestamp.now(tz="UTC")
    week = week_of(now)
    with db() as s:
        sv = StrategyVersion(name="odte_opening_drive", version="1", interval="opt", module="x", code_sha="x",
                             description="0DTE drive")
        s.add(sv)
        s.flush()
        for i in range(3):
            b = Bot(slug=f"g{i}", strategy_id=sv.id, stage=BotStage.PAPER_PROBATION, account="paper-options",
                    group=OPTIONS, starting_capital=5000, capital=5000,
                    target_qty={"at": now.isoformat(), "placed": [], "rejected": ["x: <script>alert(1)</script>"]})
            s.add(b)
            s.flush()
            s.add(WeeklyResult(week=week, bot_id=b.id, start_equity=5000, sessions=2))
            for k, eq in enumerate([5000, 5000 + 400 * (i - 1)]):
                s.add(EquitySnapshot(bot_id=b.id, ts=(now - pd.Timedelta(minutes=10 - 5 * k)).to_pydatetime(),
                                     equity=eq, cash=eq - 300, positions={"spy-drive": {
                                         "legs": [["SPY261020C00770000", 1]], "qty": 2, "entry": 1.2, "mark": 1.5,
                                         "pnl": 60.0, "note": "broke the range <b>up</b>"}}))
            o = Order(client_order_id=f"op-x-{i}", account="paper-options", interval="opt", symbol="SPY", qty=2,
                      order_type="limit", time_in_force="day", limit_price=1.2, ref_price=1.18, status="filled",
                      filled_qty=2, filled_notional=2.4, legs=[{"symbol": "SPY261020C00770000", "ratio": 1}],
                      structure="spy-drive", created_at=utcnow())
            o.allocations = [OrderAllocation(bot_id=b.id, qty=2, filled_qty=2,
                                             reason={"intent": "open", "reason": "strategy entry", "note": "<i>n</i>"})]
            s.add(o)
        s.commit()
    return week


def test_gauntlet_pages_need_login_and_render(client, db):
    week = _seed_gauntlet(db)
    for path in GAUNTLET_PAGES + [f"/options/weeks/{week}"]:
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login"), path
    login(client)
    for path in GAUNTLET_PAGES + [f"/options/weeks/{week}"]:
        r = client.get(path)
        assert r.status_code == 200, (path, r.text[:500])
        assert "Options Gauntlet" in r.text
        assert "<script>alert(1)" not in r.text and "<b>up</b>" not in r.text and "<i>n</i>" not in r.text
    home = client.get("/").text
    assert "Long-term Arena" in home and "Options Gauntlet" in home and "g2" in home
    standings = client.get("/options").text
    assert "SPY 10/20 770 call" in standings and "cut line" in standings
    assert client.get("/league").status_code == 200


def test_options_kill_needs_csrf_and_confirmation_and_is_audited(client, db):
    from autotrader.api import deps

    class FakeGauntlet:
        killed: list = []
        unkilled = 0

        def kill(self, reason):
            self.killed.append(reason)
            return "ok"

        def unkill(self):
            FakeGauntlet.unkilled += 1

    g = FakeGauntlet()
    client.app.dependency_overrides[deps.gauntlet_factory] = lambda: (lambda: g)
    _seed_gauntlet(db)
    login(client)
    token = csrf(client, "/options")
    assert client.post("/actions/options/kill", data={"confirm": "KILL"}).status_code == 403
    assert client.post("/actions/options/kill", data={"csrf": token, "confirm": "nope"}).status_code == 400
    r = client.post("/actions/options/kill", data={"csrf": token, "confirm": "kill", "reason": "test"},
                    follow_redirects=False)
    assert r.status_code == 303 and g.killed == ["dashboard: test"]
    assert client.post("/actions/options/unkill", data={"csrf": token}, follow_redirects=False).status_code == 303
    with db() as s:
        actions = [a.action for a in s.scalars(select(AuditLog))]
    assert "options_kill" in actions and "options_unkill" in actions


def test_culled_bot_keeps_its_page_history_and_final_record(client, db):
    from autotrader.db.models import Fill

    week = _seed_gauntlet(db)
    with db() as s:
        g0 = s.scalars(select(Bot).where(Bot.slug == "g0")).one()
        r = s.scalars(select(WeeklyResult).where(WeeklyResult.bot_id == g0.id)).one()
        r.status, r.rank, r.ret, r.scored_at, r.autopsy = "culled", 3, -0.4, utcnow(), "AUTOPSY: bled premium"
        g0.stage, g0.notes = BotStage.ERADICATED, r.autopsy
        s.add(Fill(bot_id=g0.id, symbol="SPY261020C00770000", qty=2, price=1.2, structure="spy-drive"))
        s.commit()
    login(client)
    page = client.get("/options/bots/g0").text
    assert "Final record" in page and "cuts survived</span>" in page and "AUTOPSY: bled premium" in page
    assert "spy-drive" in page  # its trades are still listed
    assert "g0" in client.get("/options/graveyard").text and "g0" in client.get(f"/options/weeks/{week}").text
