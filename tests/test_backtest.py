import pandas as pd
import pytest

from autotrader.backtest.costs import ZERO_COSTS, CostModel
from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.strategies.base import Strategy, StrategyContext
from autotrader.strategies.builtin.stoch_momentum import StochMomentum
from tests.conftest import make_bars


class BuyAndHold(Strategy):
    name = "test_bh"
    interval = "1d"
    universe = ("AAA",)

    def on_bar(self, ctx: StrategyContext):
        return None if ctx.weights else {"AAA": 1.0}


class Flipper(Strategy):
    """Alternates long/short each bar; heavy trading to exercise costs and shorts."""
    name = "test_flip"
    interval = "1d"
    universe = ("AAA",)

    def on_bar(self, ctx):
        return {"AAA": -1.0 if ctx.weight("AAA") > 0 else 1.0}


FRAC = BacktestConfig(fractional=True)


def test_buy_and_hold_matches_price_return(bars):
    res = run_backtest(BuyAndHold(), {"AAA": bars}, FRAC, ZERO_COSTS)
    # Sized at the close of bar 0, filled at the open of bar 1, held to the end.
    qty = 10_000 / bars.close.iloc[0]
    expected = 10_000 - qty * bars.open.iloc[1] + qty * bars.close.iloc[-1]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert res.trades[0].qty == pytest.approx(qty)
    assert len(res.trades) == 1
    assert res.trades[0].ts == bars.index[1]
    assert res.trades[0].price == bars.open.iloc[1]


def test_costs_reduce_returns(bars):
    free = run_backtest(Flipper(), {"AAA": bars}, FRAC, ZERO_COSTS)
    costly = run_backtest(Flipper(), {"AAA": bars}, FRAC, CostModel(slippage_bps=10))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert any(t.qty < 0 for t in costly.trades)


def test_no_lookahead_when_future_bars_present(universe):
    cut = universe["AAA"].index[250]
    truncated = {s: df[df.index <= cut] for s, df in universe.items()}
    full = run_backtest(StochMomentum({"max_positions": 2}), universe)
    part = run_backtest(StochMomentum({"max_positions": 2}), truncated)
    pd.testing.assert_series_equal(full.equity[full.equity.index <= cut], part.equity)
    assert [t for t in full.trades if t.ts <= cut] == part.trades


def test_start_uses_history_as_warmup_only(bars):
    start = bars.index[100]
    res = run_backtest(BuyAndHold(), {"AAA": bars}, FRAC, ZERO_COSTS, start=start)
    assert res.equity.index[0] == start
    assert res.trades[0].ts == bars.index[101]


def test_integer_shares_and_cash_stays_sane(bars):
    res = run_backtest(BuyAndHold(), {"AAA": bars}, BacktestConfig(initial_cash=1_000), ZERO_COSTS)
    assert res.trades[0].qty == int(res.trades[0].qty)
    assert res.equity.min() > 0


def test_realized_pnl_on_round_trip():
    bars = make_bars(50)
    res = run_backtest(Flipper(), {"AAA": bars}, FRAC, ZERO_COSTS)
    realized = [t.realized_pnl for t in res.trades if t.realized_pnl is not None]
    assert realized and res.metrics()["closing_trades"] == len(realized)
