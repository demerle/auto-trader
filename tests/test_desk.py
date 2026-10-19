import pytest

from autotrader.broker.desk import Intent, apportion, net_intents


def test_apportion_sums_exactly():
    out = apportion(10, {1: 1, 2: 1, 3: 1})
    assert sum(out.values()) == 10 and sorted(out.values()) == [3, 3, 4]


def test_single_bot_passes_through():
    plan = net_intents([Intent(1, "AAPL", 10)], {})
    assert [(o.symbol, o.qty, o.allocations) for o in plan.external] == [("AAPL", 10, {1: 10})]
    assert not plan.internal and not plan.deferred


def test_opposing_intents_cross_internally_and_net_rest():
    plan = net_intents([Intent(1, "AAPL", 10), Intent(2, "AAPL", 5), Intent(3, "AAPL", -6)], {})
    internal = {(b, q) for b, _, q in plan.internal}
    assert sum(q for _, q in internal if q > 0) == 6 and (3, -6) in internal
    (ext,) = plan.external
    assert ext.qty == 9 and sum(ext.allocations.values()) == 9
    # every bot ends up with exactly what it asked for
    got = {1: 0.0, 2: 0.0, 3: 0.0}
    for b, _, q in plan.internal:
        got[b] += q
    for b, q in ext.allocations.items():
        got[b] += q
    assert got == {1: 10, 2: 5, 3: -6}


def test_fully_crossed_sends_nothing():
    plan = net_intents([Intent(1, "SPY", 7), Intent(2, "SPY", -7)], {})
    assert not plan.external and len(plan.internal) == 2


def test_zero_cross_is_split_and_remainder_deferred():
    # account long 5; bot wants to go to -3 (sell 8): only sell 5 now
    plan = net_intents([Intent(1, "TSLA", -8)], {"TSLA": 5})
    (ext,) = plan.external
    assert ext.qty == -5 and ext.allocations == {1: -5}
    assert plan.deferred == [(1, "TSLA", -3)]


def test_fills_are_attributed_pro_rata(tmp_path):
    from autotrader.backtest.costs import ZERO_COSTS
    from autotrader.broker.desk import Desk
    from autotrader.broker.fake import FakeBroker
    from autotrader.ledger.store import ledger_positions
    from tests.test_arena import make_db

    sessions, bots = make_db(tmp_path, n_bots=2)
    broker = FakeBroker()
    desk = Desk(broker, ZERO_COSTS)
    with sessions() as session:
        plan = net_intents([Intent(bots[0], "AAPL", 30), Intent(bots[1], "AAPL", 10)], {})
        (order,) = desk.execute(session, plan, "1d", {"AAPL": 100.0}, "market", "opg")
        broker.fill(order.client_order_id, 101.0, qty=20)  # partial
        desk.sync(session)
        broker.fill(order.client_order_id, 103.0)
        desk.sync(session)
        session.commit()
        assert ledger_positions(session) == {"AAPL": 40}
        assert order.status == "filled"
        assert order.filled_notional == pytest.approx(20 * 101 + 20 * 103)
