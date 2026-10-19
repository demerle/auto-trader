"""The live engine, driven by a fake broker over history, must reproduce the backtest."""

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from autotrader.backtest.costs import ZERO_COSTS
from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.broker.fake import FakeBroker
from autotrader.data.calendar import bar_close_time, schedule
from autotrader.db.models import Base, Bot, BotStage, EquitySnapshot, StrategyVersion
from autotrader.engine.arena import Arena
from autotrader.risk.limits import RiskLimits
from autotrader.strategies.builtin.stoch_momentum import StochMomentum
from tests.conftest import make_bars

SYMS = ("AAA", "BBB", "CCC", "DDD")
LOOSE = RiskLimits(max_position_pct=1.0, min_price=0.0, bot_daily_loss_pct=1.0,
                   account_max_drawdown_pct=1.0)


class TestStoch(StochMomentum):
    __test__ = False
    universe = SYMS


def make_db(tmp_path, n_bots=1):
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as s:
        sv = StrategyVersion(name="stoch_momentum", version="1", interval="1d", module="x",
                             code_sha="x")
        s.add(sv)
        s.flush()
        bots = [Bot(slug=f"b{i}", strategy_id=sv.id, stage=BotStage.PAPER_PROBATION,
                    params={"max_positions": 2}) for i in range(n_bots)]
        s.add_all(bots)
        s.commit()
        return sessions, [b.id for b in bots]


class FakeArchive:
    def __init__(self, bars):
        self.bars = bars

    def get(self, symbol, interval, start=None, end=None, refresh=True):
        df = self.bars[symbol]
        return df[df.index <= end] if end is not None else df

    def load(self, symbol, interval):
        return self.bars[symbol]


@pytest.fixture
def history():
    days = schedule(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-06-30", tz="UTC")).index
    out = {}
    for i, s in enumerate(SYMS):
        df = make_bars(len(days), seed=i)
        df.index = days
        out[s] = df
    return out


def test_live_engine_matches_backtest(tmp_path, history):
    sessions, _ = make_db(tmp_path)
    broker = FakeBroker()
    arena = Arena(sessions, broker, FakeArchive(history), LOOSE, ZERO_COSTS,
                  strategy_loader=lambda bot: TestStoch(bot.params))
    days = history["AAA"].index
    for ts in days:
        broker.fill_all({s: df.loc[ts, "open"] for s, df in history.items()})
        arena.tick("1d", now=bar_close_time(ts, "1d"))

    expected = run_backtest(TestStoch({"max_positions": 2}), history,
                            BacktestConfig(initial_cash=10_000), ZERO_COSTS)
    with sessions() as s:
        snaps = s.scalars(select(EquitySnapshot).order_by(EquitySnapshot.ts)).all()
    live = pd.Series([x.equity for x in snaps], index=days[: len(snaps)])
    assert len(expected.trades) > 5
    assert live.to_numpy() == pytest.approx(expected.equity.to_numpy(), rel=1e-9)


def test_dry_run_changes_nothing(tmp_path, history):
    sessions, _ = make_db(tmp_path)
    arena = Arena(sessions, FakeBroker(), FakeArchive(history), LOOSE, ZERO_COSTS,
                  strategy_loader=lambda bot: TestStoch(bot.params))
    arena.tick("1d", now=bar_close_time(history["AAA"].index[-1], "1d"), dry_run=True)
    with sessions() as s:
        assert s.scalars(select(EquitySnapshot)).first() is None


def test_kill_switch_flattens_and_halts(tmp_path, history):
    sessions, (bot_id,) = make_db(tmp_path)
    broker = FakeBroker()
    arena = Arena(sessions, broker, FakeArchive(history), LOOSE, ZERO_COSTS,
                  strategy_loader=lambda bot: TestStoch(bot.params))
    days = history["AAA"].index
    for ts in days[:200]:
        broker.fill_all({s: df.loc[ts, "open"] for s, df in history.items()})
        arena.tick("1d", now=bar_close_time(ts, "1d"))
    assert broker.positions(), "expected open positions before the kill"
    arena.kill("test")
    broker.fill_all({s: df["close"].iloc[199] for s, df in history.items()})
    report = arena.tick("1d", now=bar_close_time(days[200], "1d"))
    assert "kill switch is ON" in str(report)
    assert broker.positions() == {}
    from autotrader.ledger.store import ledger_positions
    with sessions() as s:
        assert ledger_positions(s) == {}
