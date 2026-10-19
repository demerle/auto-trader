"""The options gauntlet: pricing, risk, planner, simulator, desk, weekly competition, isolation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from autotrader.broker.base import BrokerOrder
from autotrader.db.models import (
    OPTIONS, AgentJob, Base, Bot, BotStage, CashFlow, EquitySnapshot, Fill, Order, StrategyVersion,
    WeeklyResult,
)
from autotrader.options import competition
from autotrader.options.book import OptionBook
from autotrader.options.cboe import parse as parse_cboe
from autotrader.options.config import GauntletConfig
from autotrader.options.desk import OptionsDesk
from autotrader.options.helpers import Chain
from autotrader.options.ledger import load_option_book
from autotrader.options.occ import is_option, occ_symbol, parse
from autotrader.options.planner import plan_tick
from autotrader.options.pricing import bs_greeks, bs_price, implied_vol, nowcast
from autotrader.options.risk import RiskError, structure_check, worst_value
from autotrader.options.sim import simulate
from autotrader.options.strategy import Leg, OptionStrategy, Structure
from autotrader.options.synthetic import synthetic_chain
from autotrader.risk.limits import OptionRiskLimits

CFG = GauntletConfig(underlyings={"etfs": ["SPY", "QQQ"], "stocks": ["NVDA"]})
LIM = OptionRiskLimits()
ET = "America/New_York"


def et(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=ET).tz_convert("UTC")


# ----- pricing and symbols -----

def test_black_scholes_parity_iv_and_greeks():
    s, k, t, iv, r = 100.0, 105.0, 30 / 365, 0.25, 0.04
    c, p = bs_price(s, k, t, iv, True, r), bs_price(s, k, t, iv, False, r)
    assert c - p == pytest.approx(s - k * np.exp(-r * t), abs=1e-6)
    assert implied_vol(c, s, k, t, True, r) == pytest.approx(iv, abs=1e-6)
    g = bs_greeks(s, k, t, iv, True, r)
    bumped = bs_price(s + 0.01, k, t, iv, True, r)
    assert (bumped - c) / 0.01 == pytest.approx(float(g["delta"]), abs=1e-3)


def test_occ_round_trip():
    sym = occ_symbol("SPY", date(2026, 10, 20), "C", 770.5)
    assert sym == "SPY261020C00770500" and is_option(sym) and not is_option("SPY")
    c = parse(sym)
    assert (c.underlying, c.expiry, c.right, c.strike) == ("SPY", date(2026, 10, 20), "C", 770.5)


def _payload():
    opts = []
    for k in (95, 100, 105):
        for right in ("C", "P"):
            opts.append({"option": f"XYZ261023{right}{k * 1000:08d}", "bid": 2.0, "ask": 2.2, "iv": 0.3,
                         "delta": 0.5 if right == "C" else -0.5, "gamma": 0.05, "theta": -0.1, "vega": 0.1,
                         "open_interest": 10, "volume": 5, "last_trade_price": 2.1})
    opts.append({"option": "XYZ271217C00100000", "bid": 9, "ask": 9.5, "iv": 0.3})  # too far out
    return {"timestamp": "2026-10-20 15:00:00", "data": {"symbol": "XYZ", "current_price": 100.0, "options": opts}}


def test_cboe_parse_filters_and_delays():
    snap = parse_cboe(_payload())
    assert snap.underlying == "XYZ" and snap.spot == 100.0 and len(snap.chain) == 6
    assert snap.quote_time == pd.Timestamp("2026-10-20 14:45", tz="UTC")


def test_nowcast_follows_the_underlying():
    snap = parse_cboe(_payload())
    then = snap.quote_time
    up = nowcast(snap.chain, 100.0, 102.0, then, then + pd.Timedelta(minutes=15))
    down = nowcast(snap.chain, 100.0, 98.0, then, then + pd.Timedelta(minutes=15))
    calls = up["right"] == "C"
    assert (up.loc[calls, "mark"].to_numpy() > down.loc[calls, "mark"].to_numpy()).all()
    assert (up.loc[~calls, "mark"].to_numpy() < down.loc[~calls, "mark"].to_numpy()).all()
    assert (up["est_ask"] > up["est_bid"]).all()


# ----- book and risk -----

def test_book_uses_the_contract_multiplier():
    b = OptionBook(5000)
    t = pd.Timestamp("2026-10-20 15:00", tz="UTC")
    b.fill("x", "SPY261020C00770000", 2, 1.50, 0.10, t)
    assert b.cash == pytest.approx(5000 - 300 - 0.10)
    assert b.equity({"SPY261020C00770000": 2.0}) == pytest.approx(5000 - 0.10 + 100)
    b.fill("x", "SPY261020C00770000", -2, 2.50, 0.10, t)
    assert not b.open and b.closed[0].pnl == pytest.approx(200 - 0.20)


def test_worst_case_values():
    assert worst_value([("SPY261020C00770000", 1)]) == 0
    assert worst_value([("SPY261020P00760000", -1), ("SPY261020P00755000", 1)]) == -500
    assert worst_value([("SPY261020C00770000", -1)]) == -np.inf
    with pytest.raises(RiskError):
        structure_check([("SPY261020C00770000", -1)], 1, -1.0, 5000, 0.6, 4, 50, ("SPY",))
    with pytest.raises(RiskError):  # $5 wide x 10 = $5,000 at risk > 60% of $5,000
        structure_check([("SPY261020P00760000", -1), ("SPY261020P00755000", 1)], 10, -0.5, 5000, 0.6, 4, 50, ("SPY",))
    with pytest.raises(RiskError):
        structure_check([("TSLA261020C00300000", 1)], 1, 1.0, 5000, 0.6, 4, 50, ("SPY",))
    assert structure_check([("SPY261020C00770000", 1)], 2, 1.0, 5000, 0.6, 4, 50, ("SPY",)) == 200


# ----- planner -----

def _chain(now, spot=767.0):
    return synthetic_chain("SPY", spot, now, 16.0, 13.0)


def test_planner_opens_closes_and_enforces_cutoffs():
    now = et("2026-10-20 10:00")
    chains = {"SPY": _chain(now)}
    c = chains["SPY"]
    row = c.nearest_delta(c.expiry(0), "C", 0.3)
    want = {"a": Structure((Leg(row["symbol"], 1),), 3, "test")}
    plan = plan_tick(now, want, OptionBook(5000), chains, 5000, LIM, CFG, False)
    assert [(ch.kind, ch.units) for ch in plan.changes] == [("open", 3)]

    book = OptionBook(5000)
    book.fill("a", row["symbol"], 3, float(row["mark"]), 0, now)
    assert plan_tick(now, {}, book, chains, 5000, LIM, CFG, False).changes[0].kind == "close"
    # 10 minutes before SPY's 15:30 0DTE cutoff the position is closed whatever the strategy says
    late = et("2026-10-20 15:21")
    forced = plan_tick(late, want, book, {"SPY": _chain(late)}, 5000, LIM, CFG, False)
    assert forced.changes[0].kind == "close" and "cutoff" in forced.changes[0].reason
    # no new same-day positions 45 minutes before the cutoff
    late_open = plan_tick(et("2026-10-20 14:50"), want, OptionBook(5000), {"SPY": _chain(et("2026-10-20 14:50"))},
                          5000, LIM, CFG, False)
    assert not late_open.changes and "cutoff" in late_open.rejected[0][1]


def test_planner_flattens_friday_and_rejects_undefined_risk_and_busts():
    fri = et("2026-10-23 15:50")
    chains = {"SPY": _chain(fri)}
    c = chains["SPY"]
    nxt = c.expiry(1)
    row = c.atm(nxt, "C")
    book = OptionBook(5000)
    book.fill("a", row["symbol"], 1, float(row["mark"]), 0, fri)
    plan = plan_tick(fri, {"a": Structure((Leg(row["symbol"], 1),), 1)}, book, chains, 5000, LIM, CFG, True)
    assert plan.changes[0].kind == "close" and "weekly" in plan.changes[0].reason

    mon = et("2026-10-26 11:00")
    chains = {"SPY": _chain(mon)}
    short_call = chains["SPY"].atm(chains["SPY"].expiry(1), "C")["symbol"]
    naked = plan_tick(mon, {"n": Structure((Leg(short_call, -1),), 1)}, OptionBook(5000), chains, 5000, LIM, CFG, False)
    assert not naked.changes and "undefined" in naked.rejected[0][1]
    busted = OptionBook(100)
    busted.fill("a", short_call, 1, 0.5, 0, mon)
    plan = plan_tick(mon, {"a": Structure((Leg(short_call, 1),), 1)}, busted, chains, 150, LIM, CFG, False, "busted")
    assert plan.changes[0].kind == "close"


# ----- simulator -----

class BuyAtm(OptionStrategy):
    name, underlyings = "buy_atm", ("SPY",)

    def on_tick(self, ctx):
        c = ctx.chains.get("SPY")
        if c is None:
            return {}
        if ctx.positions:
            pos = ctx.positions["atm"]
            return {} if ctx.minutes_since_open > 120 else {"atm": Structure(pos.legs, pos.qty)}
        if ctx.closed_today or ctx.minutes_since_open < 10:
            return {}
        e = c.expiry(0, min_dte=1)
        return {"atm": Structure((Leg(c.atm(e, "C")["symbol"], 1),), 2, "test entry")}


def _five_minute_bars(days):
    rows = []
    rng = np.random.default_rng(1)
    price = 700.0
    from autotrader.data.calendar import schedule

    for d, row in schedule(days[0], days[1]).iterrows():
        t = row["market_open"]
        while t < row["market_close"]:
            price *= np.exp(rng.normal(0, 0.001))
            rows.append((t, price, price * 1.001, price * 0.999, price, 1e5))
            t += pd.Timedelta(minutes=5)
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")


def test_simulator_trades_resets_weekly_and_stays_flat_over_weekends():
    start, end = pd.Timestamp("2026-09-14", tz="UTC"), pd.Timestamp("2026-09-25", tz="UTC")
    bars = _five_minute_bars((start, end))
    daily = bars.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                     "volume": "sum"}).dropna()
    s = BuyAtm()
    res = simulate(s.on_tick, s.underlyings, {"SPY": bars}, {"SPY": daily}, None, None, start, end, CFG, LIM)
    assert not res.errors
    assert res.opened == 10 and len(res.trades) == 10
    assert [w["start"] for w in res.weeks] == [5000.0, 5000.0]
    assert res.min_worst_equity >= 0


# ----- desk and database -----

class FakeOptionsBroker:
    account = "paper-options"

    def __init__(self):
        self.orders: dict[str, BrokerOrder] = {}
        self.sent: list[tuple] = []
        self.pos: dict[str, float] = {}

    def submit_option(self, coid, legs, qty, limit):
        self.sent.append((coid, legs, qty, limit))
        self.orders[coid] = BrokerOrder(coid, f"b-{coid}", legs[0][0] if len(legs) == 1 else "", qty, "new",
                                        legs=[{"symbol": s, "ratio_qty": abs(r), "filled_avg_price": None}
                                              for s, r, _ in legs] if len(legs) > 1 else None)
        return self.orders[coid]

    def fill(self, coid, filled, price, leg_prices=None):
        o = self.orders[coid]
        o.filled_qty, o.filled_avg_price, o.status = filled, price, "filled"
        if leg_prices and o.legs:
            for leg in o.legs:
                leg["filled_avg_price"] = leg_prices[leg["symbol"]]

    def get_order(self, coid):
        return self.orders[coid]

    def cancel(self, coid):
        if self.orders[coid].status not in ("filled",):
            self.orders[coid].status = "canceled"

    def cancel_all(self):
        for k in self.orders:
            self.cancel(k)

    def option_positions(self):
        return dict(self.pos)

    def stock_positions(self):
        return {}

    def equity(self):
        return 100_000.0


@pytest.fixture
def odb(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/o.db")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as s:
        sv = StrategyVersion(name="buy_atm", version="1", interval="opt", module="x", code_sha="x")
        arena_sv = StrategyVersion(name="stoch_momentum", version="1", interval="1d", module="x", code_sha="y")
        s.add_all([sv, arena_sv])
        s.flush()
        bots = [Bot(slug=f"o{i}", strategy_id=sv.id, stage=BotStage.PAPER_PROBATION, account="paper-options",
                    group=OPTIONS, starting_capital=5000, capital=5000) for i in range(5)]
        bots.append(Bot(slug="arena-1", strategy_id=arena_sv.id, stage=BotStage.PAPER_PROBATION))
        s.add_all(bots)
        s.commit()
    return sessions


def _change(kind, tag, legs, units):
    from autotrader.options.planner import Change

    return Change(tag, kind, legs, units, "test")


def test_desk_places_single_and_multileg_and_books_fills(odb):
    now = et("2026-10-20 10:00")
    chains = {"SPY": _chain(now)}
    c = chains["SPY"]
    e = c.expiry(1)
    a, b = c.atm(e, "C")["symbol"], c.atm(e, "C", offset=5)["symbol"]
    broker = FakeOptionsBroker()
    desk = OptionsDesk(broker, CFG)
    with odb() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        p1 = desk.place(s, bot, _change("open", "single", [(a, 2)], 2), chains)
        put_hi, put_lo = c.atm(e, "P")["symbol"], c.atm(e, "P", offset=-5)["symbol"]
        p2 = desk.place(s, bot, _change("open", "spread", [(put_hi, 3), (put_lo, -3)], 3), chains)
        assert p1.status == p2.status == "placed"
        coid1, legs1, qty1, _ = broker.sent[0]
        assert legs1 == [(a, 1, "open")] and qty1 == 2
        coid2, legs2, qty2, _ = broker.sent[1]
        assert [r for _, r, _ in legs2] == [1, -1] and qty2 == 3  # ratios in lowest terms
        broker.fill(coid1, 2, 1.25)
        broker.fill(coid2, 3, 0.80, {legs2[0][0]: 2.0, legs2[1][0]: 1.2})
        assert desk.sync(s) == 3
        s.commit()
        book = load_option_book(s, bot)
        assert book.open["single"].legs == {a: 2}
        assert book.cash == pytest.approx(5000 - 250 - 240 - 0.05 * (2 + 3 + 3))
        # another bot may not take the other side of a contract this bot holds
        other = s.scalars(select(Bot).where(Bot.slug == "o1")).one()
        assert desk.place(s, other, _change("open", "x", [(a, -1), (b, 1)], 1), chains).status == "conflict"


def test_desk_chases_toward_the_far_side_then_gives_up(odb):
    now = et("2026-10-20 10:00")
    chains = {"SPY": _chain(now)}
    c = chains["SPY"]
    sym = c.atm(c.expiry(1), "C")["symbol"]
    broker = FakeOptionsBroker()
    desk = OptionsDesk(broker, CFG)
    with odb() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        desk.place(s, bot, _change("open", "t", [(sym, 1)], 1), chains)
        s.commit()
        limits = [broker.sent[-1][3]]
        for i in range(1, 6):
            later = now + pd.Timedelta(seconds=31 * i)
            for o in s.scalars(select(Order)).all():
                o.created_at = (later - pd.Timedelta(seconds=40)).to_pydatetime()
            desk.chase(s, later, chains)
            s.commit()
            limits.append(broker.sent[-1][3])
        assert len(broker.sent) == 1 + CFG.execution.chase_steps
        assert limits[0] < limits[1] < limits[CFG.execution.chase_steps]
        assert limits[-1] == pytest.approx(float(c.row(sym)["est_ask"]))


def test_weekly_reset_score_and_cull(odb, sent_alerts):
    mon = et("2026-10-26 09:05")
    with odb() as s:
        bots = s.scalars(select(Bot).where(Bot.group == OPTIONS).order_by(Bot.id)).all()
        # last week's results left different cash in each bot
        for i, b in enumerate(bots):
            s.add(CashFlow(bot_id=b.id, amount=1000 * (i - 2), reason="pnl"))
        s.flush()
        competition.start_week(s, mon, CFG)
        assert competition.start_week(s, mon, CFG) == []  # idempotent
        for b in bots:
            assert load_option_book(s, b).cash == pytest.approx(5000)
        # the week: returns -40%, -10%, 0, +20%, +50%; o4 joined Thursday
        rets = [-0.4, -0.1, 0.0, 0.2, 0.5]
        for b, r in zip(bots, rets):
            s.add(CashFlow(bot_id=b.id, amount=5000 * r, reason="pnl", ts=et("2026-10-30 15:00").to_pydatetime()))
            res = competition.current_result(s, b)
            res.sessions = 5 if b.slug != "o4" else 2
        s.flush()
        fri = et("2026-10-30 16:31")
        preview = competition.score_week(s, fri, CFG, apply=False)
        assert [x["slug"] for x in preview["standings"]] == ["o4", "o3", "o2", "o1", "o0"]
        out = competition.score_week(s, fri, CFG)
        culled = {x["slug"] for x in out["standings"] if x["culled"]}
        assert culled == {"o0", "o1"}
        assert {b.slug for b in bots if b.stage == BotStage.ERADICATED} == culled
        jobs = s.scalars(select(AgentJob).where(AgentJob.agent == "options_researcher")).all()
        assert len(jobs) == 2 and "Autopsy" in jobs[0].prompt
        statuses = {r.bot.slug: r.status for r in s.scalars(select(WeeklyResult))}
        assert statuses["o4"] == "exempt" and statuses["o2"] == "survived"
        assert competition.score_week(s, fri, CFG)["skipped"]
    assert any("Gauntlet" in str(x) for x in sent_alerts)


def test_practice_week_culls_nobody_and_new_bot_is_exempt(odb, sent_alerts):
    mon = et("2026-10-26 09:05")
    with odb() as s:
        competition.start_week(s, mon, CFG)
        for r in s.scalars(select(WeeklyResult)):
            r.sessions = 5
        s.flush()
        out = competition.score_week(s, et("2026-10-30 16:31"), CFG, practice=True)
        assert not any(x["culled"] for x in out["standings"])


# ----- isolation from the long-term arena -----

def test_arena_ignores_the_options_account(odb):
    from autotrader.broker.reconcile import reconcile
    from autotrader.ledger.store import ledger_positions

    with odb() as s:
        o0 = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        arena = s.scalars(select(Bot).where(Bot.slug == "arena-1")).one()
        s.add(Fill(bot_id=o0.id, symbol="SPY261020C00770000", qty=2, price=1.0, structure="t"))
        s.add(Fill(bot_id=arena.id, symbol="SPY", qty=5, price=700.0))
        s.commit()
        assert ledger_positions(s, "paper") == {"SPY": 5}

        class ArenaBroker:
            account = "paper"

            def positions(self):
                return {"SPY": 5}
        assert reconcile(s, ArenaBroker()) == []
        from autotrader.orchestrator.slots import active_bots

        assert [b.slug for b in active_bots(s)] == ["arena-1"]


def test_options_kill_leaves_the_arena_alone(odb, sent_alerts):
    from autotrader.engine.state import is_killed
    from autotrader.options.engine import Gauntlet

    broker = FakeOptionsBroker()
    g = Gauntlet(odb, broker, None, CFG, __import__("autotrader.risk.limits", fromlist=["RiskLimits"]).RiskLimits())
    with odb() as s:
        o0 = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        s.add(Fill(bot_id=o0.id, symbol="SPY261120C00770000", qty=2, price=1.0, structure="t"))
        s.commit()
    g.kill("test")
    with odb() as s:
        assert is_killed(s, "paper-options") and not is_killed(s, "paper")
    assert broker.sent and broker.sent[-1][3] is None  # market close order
    assert broker.sent[-1][1] == [("SPY261120C00770000", -1, "close")]


# ----- strategy safety -----

GOOD = '''
from autotrader.options.helpers import units_for
from autotrader.options.strategy import Leg, OptionStrategy, Structure


class Toy(OptionStrategy):
    name = "toy"
    version = "1"
    underlyings = ("SPY",)
    thesis = "test"

    def on_tick(self, ctx):
        c = ctx.chains["SPY"]
        row = c.atm(c.expiry(0, min_dte=1), "C")
        return {"t": Structure((Leg(row["symbol"], 1),), 1, "toy")}
'''


def test_generated_option_strategy_runs_sandboxed_and_lint_blocks_io(tmp_path):
    from autotrader.options.context import build_context
    from autotrader.options.registry import sandboxed_class
    from autotrader.strategies.lint import lint_source

    assert lint_source(GOOD) == []
    assert lint_source(GOOD + "\nimport os\n")
    path = tmp_path / "toy_v1.py"
    path.write_text(GOOD)
    cls = sandboxed_class(path)
    now = et("2026-10-20 10:00")
    chains = {"SPY": _chain(now)}
    ctx = build_context(now, chains, {"SPY": 767.0}, {}, {}, 16.0, 13.0, OptionBook(5000), {}, 5000,
                        et("2026-10-20 09:30"), et("2026-10-20 16:00"), False)
    s = cls()
    try:
        out = s.on_tick(ctx)
    finally:
        s.close()
    assert list(out) == ["t"] and out["t"].legs[0].symbol.startswith("SPY2610")


def test_gauntlet_tick_places_books_and_snapshots(odb, sent_alerts):
    from autotrader.options.cboe import ChainSnapshot
    from autotrader.options.engine import Gauntlet
    from autotrader.risk.limits import RiskLimits

    now = et("2026-10-20 10:30")
    snap_chain = _chain(now - pd.Timedelta(minutes=15))
    cols = ["symbol", "underlying", "expiry", "right", "strike", "bid", "ask", "last", "iv", "delta", "gamma",
            "theta", "vega", "oi", "volume"]
    snap = ChainSnapshot("SPY", now, now - pd.Timedelta(minutes=15), 767.0, snap_chain.df[cols].copy())
    bars = _five_minute_bars((pd.Timestamp("2026-10-13", tz="UTC"), pd.Timestamp("2026-10-20", tz="UTC")))
    bars = bars * (767.0 / bars["close"].iloc[-1])

    class Arch:
        def get(self, symbol, interval, start=None, end=None, refresh=True):
            return bars if symbol == "SPY" and interval in ("1m", "5m", "1d") else pd.DataFrame()

    broker = FakeOptionsBroker()
    g = Gauntlet(odb, broker, Arch(), CFG, RiskLimits(), chain_archive=type("A", (), {"save": lambda self, s: None})(),
                 chain_fetcher=lambda u: snap, strategy_loader=lambda bot: BuyAtm())
    rep = g.tick(now)
    assert len(broker.sent) == 5, str(rep)  # every options bot opens; arena bot untouched
    coid = broker.sent[0][0]
    broker.fill(coid, 2, 3.10)
    rep2 = g.tick(now + pd.Timedelta(minutes=5))
    with odb() as s:
        o0 = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        book = load_option_book(s, o0)
        assert book.open["atm"].legs and book.cash == pytest.approx(5000 - 620 - 0.10)
        snaps = s.scalars(select(EquitySnapshot).where(EquitySnapshot.bot_id == o0.id)).all()
        assert len(snaps) == 2 and snaps[-1].positions["atm"]["qty"] == 2
        assert s.scalars(select(Fill)).all()[0].structure == "atm"
    assert "o0" in str(rep2)
