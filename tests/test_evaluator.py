import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from autotrader.db.models import AgentJob, Bot, BotStage, CashFlow, EquitySnapshot, Evaluation
from autotrader.evaluator.allocator import apply_allocation, plan_allocation
from autotrader.evaluator.decide import decide
from autotrader.evaluator.policy import load_policy
from autotrader.evaluator.record import daily_returns
from autotrader.evaluator.run import evaluate_all
from tests.test_arena import make_db

P = load_policy()
D = BotStage.PAPER_PROBATION


def test_daily_returns_ignore_capital_flows():
    snaps = [(pd.Timestamp("2026-01-05 21:00", tz="UTC"), 10_100.0),
             (pd.Timestamp("2026-01-06 21:00", tz="UTC"), 15_150.0)]
    flows = [(pd.Timestamp("2026-01-05 22:00", tz="UTC"), 5_000.0)]
    r = daily_returns(10_000, snaps, flows)
    assert r.iloc[0] == pytest.approx(0.01)
    assert r.iloc[1] == pytest.approx(15_150 / 15_100 - 1)


def test_probation_keeps_young_bots():
    v = decide(D, "1d", 10, 3, {"psr": 0.1, "max_drawdown": -0.05}, P)
    assert v.action == "keep" and "probation" in v.rationale


def test_drawdown_kills_any_time():
    assert decide(D, "1d", 6, 1, {"max_drawdown": -0.2}, P).action == "eradicate"


def test_matured_low_psr_eradicated():
    assert decide(D, "1d", 45, 20, {"psr": 0.3, "max_drawdown": -0.05}, P).action == "eradicate"


def test_inactive_bot_eradicated_after_max_window():
    assert decide(D, "1d", 130, 2, {"psr": 0.9, "max_drawdown": -0.01}, P).action == "eradicate"


def test_backtest_shortfall_eradicated():
    m = {"psr": 0.8, "max_drawdown": -0.05, "backtest_z": -2.5}
    assert decide(D, "1d", 45, 20, m, P).action == "eradicate"


def test_proven_bot_flagged_for_live():
    m = {"psr": 0.97, "alpha": 0.05, "excess_return": 0.03, "backtest_z": 0.2,
         "max_drawdown": -0.04, "sharpe": 2.1}
    assert decide(D, "1d", 45, 20, m, P).action == "flag_live"
    assert decide(BotStage.PAPER_PROVEN, "1d", 45, 20, m, P).action == "keep"
    m2 = m | {"psr": 0.7}
    assert decide(BotStage.PAPER_PROVEN, "1d", 45, 20, m2, P).action == "unflag"


def test_league_bottom_needs_streak_and_spy_shortfall():
    m = {"psr": 0.7, "max_drawdown": -0.05, "excess_return": -0.02}
    assert decide(D, "1d", 45, 20, m, P, league_bottom=True, prior_bottom_streak=0).action == "keep"
    assert decide(D, "1d", 45, 20, m, P, league_bottom=True, prior_bottom_streak=1).action == "eradicate"
    winning = m | {"excess_return": 0.01}
    assert decide(D, "1d", 45, 20, winning, P, league_bottom=True, prior_bottom_streak=5).action == "keep"


def _seed_equity(sessions, bot_id, daily_rets, start="2026-01-02"):
    days = pd.bdate_range(start, periods=len(daily_rets))
    eq = 10_000 * np.cumprod(1 + np.asarray(daily_rets))
    with sessions() as s:
        for d, e in zip(days, eq):
            s.add(EquitySnapshot(bot_id=bot_id, ts=(pd.Timestamp(d) + pd.Timedelta(hours=21)).tz_localize("UTC"),
                                 equity=float(e), cash=float(e)))
        s.commit()


def test_eradication_queues_research_and_records_graveyard(tmp_path):
    sessions, (bot_id,) = make_db(tmp_path)
    _seed_equity(sessions, bot_id, [-0.01] * 30)  # -26%: drawdown kill
    with sessions() as s:
        (o,) = evaluate_all(s, P, None)
        s.commit()
        assert o.verdict.action == "eradicate"
        bot = s.get(Bot, bot_id)
        assert bot.stage == BotStage.ERADICATED and "drawdown" in bot.notes
        job = s.scalars(select(AgentJob)).one()
        assert job.agent == "researcher" and job.status == "queued" and bot.slug in job.prompt
        assert s.scalars(select(Evaluation)).one().decision == "eradicate"


def test_allocator_tilts_but_conserves_capital(tmp_path):
    sessions, ids = make_db(tmp_path, n_bots=3)
    rng = np.random.default_rng(0)
    for bot_id, mu in zip(ids, (0.004, 0.0, -0.004)):
        _seed_equity(sessions, bot_id, rng.normal(mu, 0.01, 60))
    with sessions() as s:
        moves = apply_allocation(s, P, plan_allocation(s, P))
        s.commit()
        caps = {b.id: b.capital for b in s.scalars(select(Bot)).all()}
        flows = s.scalars(select(CashFlow)).all()
    assert sum(caps.values()) == pytest.approx(30_000)
    assert caps[ids[0]] > caps[ids[1]] > caps[ids[2]]
    assert all(5_000 * 0.99 <= c <= 30_000 for c in caps.values())
    assert sum(f.amount for f in flows) == pytest.approx(0.0, abs=1e-6)
    assert moves


def test_engine_closes_out_eradicated_bot(tmp_path, history):
    from autotrader.backtest.costs import ZERO_COSTS
    from autotrader.broker.fake import FakeBroker
    from autotrader.data.calendar import bar_close_time
    from autotrader.engine.arena import Arena
    from autotrader.ledger.store import ledger_positions
    from tests.test_arena import LOOSE, FakeArchive, TestStoch

    sessions, (bot_id,) = make_db(tmp_path)
    broker = FakeBroker()
    arena = Arena(sessions, broker, FakeArchive(history), LOOSE, ZERO_COSTS,
                  strategy_loader=lambda bot: TestStoch(bot.params))
    days = history["AAA"].index
    i = 0
    while not broker.positions():
        broker.fill_all({s: df.loc[days[i], "open"] for s, df in history.items()})
        arena.tick("1d", now=bar_close_time(days[i], "1d"))
        i += 1
    with sessions() as s:
        s.get(Bot, bot_id).stage = BotStage.ERADICATED
        s.commit()
    report = arena.tick("1d", now=bar_close_time(days[i], "1d"))
    assert "closing out" in str(report)
    broker.fill_all({s: df.loc[days[i + 1], "open"] for s, df in history.items()})
    arena.tick("1d", now=bar_close_time(days[i + 1], "1d"))
    assert broker.positions() == {}
    with sessions() as s:
        assert ledger_positions(s) == {}


def test_gate_reports_and_benchmarks(universe):
    from autotrader.evaluator.gate import run_gate
    from tests.test_arena import TestStoch

    bars = {s: df for s, df in universe.items()}
    bars = {"AAA": bars["AAA"], "BBB": bars["BBB"], "CCC": bars["CCC"], "DDD": bars["DDD"]}
    res = run_gate(TestStoch, bars, P.gate, params={"max_positions": 2})
    assert {"sharpe", "dsr", "benchmark_sharpe", "excess_sharpe", "daily_mean", "daily_std"} <= set(res.metrics)
    assert res.passed == (not res.reasons)
