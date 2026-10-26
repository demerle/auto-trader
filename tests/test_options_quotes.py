"""Real-time quotes (Schwab), the desk's live re-quote and staleness guard, and the shadow ledger."""

import pandas as pd
import pytest
from sqlalchemy import select

from autotrader.db.models import Bot, Fill, WeeklyResult
from autotrader.options import competition
from autotrader.options.pricing import nowcast
from autotrader.options.quotes import (
    CboeQuotes, FailoverQuotes, Quote, QuoteError, parse_schwab_chain, schwab_symbol,
)
from autotrader.options.shadow import shadow_cost, shadow_prices, touch_price
from tests.test_options import CFG, FakeOptionsBroker, _chain, _change, et, odb  # noqa: F401

SYM = "SPY261027C00680000"


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class FakeQuotes:
    live = True

    def __init__(self, quotes=None, fail=False):
        self.book = dict(quotes or {})
        self.fail = fail
        self.asked: list[list[str]] = []

    def chain(self, underlying, max_dte, moneyness):
        raise QuoteError("no chains here")

    def quotes(self, symbols):
        self.asked.append(list(symbols))
        if self.fail:
            raise QuoteError("down")
        return {s: self.book[s] for s in symbols if s in self.book}

    def spot(self, symbols):
        return {}


def q(bid, ask, size=50, age=0.0):
    return Quote(bid, ask, size, size, _now() - pd.Timedelta(seconds=age))


# ----- Schwab parsing -----

def test_schwab_symbols_and_chain_parse():
    assert schwab_symbol(SYM) == "SPY   261027C00680000"
    assert schwab_symbol("NVDA261030P00210000") == "NVDA  261030P00210000"
    now = pd.Timestamp("2026-10-27 14:00", tz="UTC")
    contract = {"putCall": "CALL", "symbol": "SPY   261027C00680000", "bid": 1.02, "ask": 1.04, "last": 1.03,
                "bidSize": 120, "askSize": 80, "volatility": 18.5, "delta": 0.31, "gamma": 0.05,
                "theta": -0.4, "vega": 0.1, "openInterest": 5000, "totalVolume": 12000,
                "quoteTimeInLong": int(now.timestamp() * 1000)}
    far = dict(contract, symbol="SPY   261027C00400000", volatility=-999.0, delta="NaN")
    payload = {"symbol": "SPY", "status": "SUCCESS", "isDelayed": False, "underlying": {"last": 678.5},
               "callExpDateMap": {"2026-10-27:0": {"680.0": [contract], "400.0": [far]}}, "putExpDateMap": {}}
    snap = parse_schwab_chain(payload, now)
    assert snap.live and snap.quote_time == now and snap.spot == 678.5
    assert list(snap.chain["symbol"]) == [SYM]  # the 400 strike is outside the moneyness band
    row = snap.chain.iloc[0]
    assert row["bid"] == 1.02 and row["ask_size"] == 80 and row["iv"] == pytest.approx(0.185)
    delayed = parse_schwab_chain(dict(payload, isDelayed=True), now)
    assert not delayed.live and delayed.quote_time == now - pd.Timedelta(minutes=15)


def test_live_chain_marks_are_the_real_mid_and_touch():
    now = et("2026-10-20 10:00")
    df = _chain(now).df.copy()
    df["bid"], df["ask"] = 1.00, 1.10
    live = nowcast(df, 767.0, 767.0, now - pd.Timedelta(seconds=5), now, live=True)
    assert list(live["mark"]) == pytest.approx([1.05] * len(live))
    assert (live["est_bid"] == 1.00).all() and (live["est_ask"] == 1.10).all()
    old = nowcast(df, 767.0, 767.0, now - pd.Timedelta(minutes=15), now, live=True)  # too old: modelled
    assert ((old["est_ask"] - old["mark"]) > 0.05).all()  # spread widened for staleness


def test_failover_alerts_once_and_falls_back(monkeypatch):
    from autotrader.options import cboe

    calls = []
    monkeypatch.setattr(cboe, "fetch_chain", lambda u, d, m: calls.append(u) or "cboe-snap")
    alerts = []
    f = FailoverQuotes(FakeQuotes(fail=True), CboeQuotes(), retry_sec=3600,
                       alert=lambda t, m: alerts.append(t))
    assert f.live
    assert f.chain("SPY", 45, 0.25) == "cboe-snap" and not f.live
    assert f.quotes([SYM]) == {} and f.chain("QQQ", 45, 0.25) == "cboe-snap"
    assert len(alerts) == 1 and calls == ["SPY", "QQQ"]


# ----- desk -----

def test_desk_prices_off_live_quotes_and_guards_stale_ones(odb):
    from autotrader.options.desk import OptionsDesk

    now = et("2026-10-20 10:00")
    chains = {"SPY": _chain(now)}
    c = chains["SPY"]
    a = c.atm(c.expiry(1), "C")["symbol"]
    broker = FakeOptionsBroker()
    quotes = FakeQuotes({a: q(2.00, 2.10)})
    desk = OptionsDesk(broker, CFG, quotes)
    with odb() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        assert desk.place(s, bot, _change("open", "t", [(a, 2)], 2), chains).status == "placed"
        assert broker.sent[-1][3] == pytest.approx(2.05)  # the real mid, not the synthetic chain's
        broker.fill(broker.sent[-1][0], 2, 2.05)
        desk.sync(s)
        desk.quotes = FakeQuotes({a: q(2.00, 2.10, age=60)})
        p = desk.place(s, bot, _change("open", "u", [(a, 1)], 1), chains)
        assert p.status == "deferred" and "real-time" in p.detail
        p = desk.place(s, bot, _change("close", "t", [(a, -2)], 2), chains)  # closes still go out
        assert p.status == "placed" and p.order.allocations[0].reason["priced"] == "chain"
        # the chase re-quotes: the next step is priced off the new live quote
        desk.quotes = FakeQuotes({a: q(3.00, 3.20)})
        p.order.created_at = (_now() - pd.Timedelta(seconds=40)).to_pydatetime()
        s.flush()
        assert desk.chase(s, _now(), chains) == 1
        assert broker.sent[-1][3] == pytest.approx(-3.075)  # sell: mid 3.10, a quarter of the way to the bid


def test_desk_records_the_shadow_fill(odb):
    from autotrader.options.desk import OptionsDesk

    now = et("2026-10-20 10:00")
    chains = {"SPY": _chain(now)}
    c = chains["SPY"]
    a = c.atm(c.expiry(1), "C")["symbol"]
    broker = FakeOptionsBroker()
    desk = OptionsDesk(broker, CFG, FakeQuotes({a: Quote(2.00, 2.10, 10, 3, _now())}))
    with odb() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        p = desk.place(s, bot, _change("open", "t", [(a, 7)], 7), chains)
        coid = broker.sent[-1][0]
        broker.fill(coid, 7, 2.05)
        broker.orders[coid].filled_at = p.order.created_at  # marketable: judged at the submit quote
        desk.sync(s)
        s.commit()
        f = s.scalars(select(Fill)).one()
        # 7 contracts vs 3 shown at the ask: three displayed lots -> two ticks through the ask
        assert f.price == 2.05 and f.shadow_price == pytest.approx(2.12)
        assert f.quote["basis"] == "submit" and f.quote["ask"] == 2.10
        assert shadow_cost(s, bot.id) == pytest.approx((2.12 - 2.05) * 7 * 100)


# ----- shadow math -----

def test_shadow_prices_single_mleg_and_never_better_than_paper():
    quote = {"bid": 1.00, "ask": 1.04, "bid_size": 20, "ask_size": 20}
    assert touch_price(SYM, 5, quote) == 1.04 and touch_price(SYM, -5, quote) == 1.00
    assert touch_price(SYM, 45, quote) == pytest.approx(1.06)  # 3 lots of 20
    assert shadow_prices([(SYM, 5)], {SYM: 1.02}, {SYM: quote}) == {SYM: 1.04}
    assert shadow_prices([(SYM, 5)], {SYM: 1.06}, {SYM: quote}) == {SYM: 1.06}  # paper already worse
    assert shadow_prices([(SYM, -5)], {SYM: 1.02}, {SYM: quote}) == {SYM: 1.00}
    assert shadow_prices([(SYM, 5)], {SYM: 1.02}, {}) is None
    other = "SPY261027C00690000"
    q2 = {"bid": 0.40, "ask": 0.42, "bid_size": 20, "ask_size": 20}
    # a debit spread filled at 0.60 net; legging in costs 1.04 - 0.40 = 0.64
    got = shadow_prices([(SYM, 1), (other, -1)], {SYM: 1.02, other: 0.42}, {SYM: quote, other: q2})
    assert got == {SYM: 1.04, other: 0.40}


def test_live_ready_needs_winning_real_money_weeks(odb):
    with odb() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == "o0")).one()
        weeks = ["2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26"]
        for i, w in enumerate(weeks):
            s.add(WeeklyResult(week=w, bot_id=bot.id, start_equity=5000, ret=0.10, shadow_ret=0.05 if i else -0.01,
                               scored_at=pd.Timestamp(w, tz="UTC").to_pydatetime(),
                               started_at=pd.Timestamp(w, tz="UTC").to_pydatetime()))
        s.commit()
        got = competition.live_ready(s, bot.id)
        assert not got["ready"] and "losing week" in got["why"]
        s.scalars(select(WeeklyResult).where(WeeklyResult.week == "2026-10-05")).one().shadow_ret = 0.02
        s.commit()
        assert competition.live_ready(s, bot.id)["ready"]
