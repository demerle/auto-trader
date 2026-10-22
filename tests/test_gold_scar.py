"""gold_scar: entry rule, ITM strike, size split across tags, stop through the open; calibrated 0DTE pricing."""

import numpy as np
import pandas as pd

from autotrader.data.calendar import schedule
from autotrader.options.builtin.gold_scar import GoldScar
from autotrader.options.config import GauntletConfig
from autotrader.options.occ import parse
from autotrader.options.sim import simulate
from autotrader.options.strategy import decode_targets, encode_targets
from autotrader.options.synthetic import ZERO_DTE_PER_VIX1D, SyntheticChains
from autotrader.risk.limits import OptionRiskLimits

CFG = GauntletConfig(underlyings={"etfs": ["SPY", "QQQ"]})
LIM = OptionRiskLimits()
ET = "America/New_York"


def _day_path(kind: str) -> np.ndarray:
    """78 five-minute closes for one session, starting at 700."""
    k = np.arange(78)
    if kind == "trend":  # +0.4% by 10:30, keeps grinding up
        return 700 * (1 + 0.0004 * k)
    if kind == "fail":  # +0.4% by 10:30, back below the open by 11:00, stays there
        up = 700 * (1 + 0.0004 * k)
        return np.where(k <= 11, up, 700 * (1 - 0.001 - 0.0001 * (k - 12)))
    return 700 + 0.05 * np.sin(k)  # flat


def _bars(kinds: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for (_, row), kind in zip(schedule(start, end).iterrows(), kinds):
        for i, c in enumerate(_day_path(kind)):
            t = row["market_open"] + pd.Timedelta(minutes=5 * i)
            o = 700.0 if i == 0 else rows[-1][4]
            rows.append((t, o, max(o, c) + 0.01, min(o, c) - 0.01, c, 1e5))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")


def _run(kinds, params=None):
    start, end = pd.Timestamp("2026-09-14", tz="UTC"), pd.Timestamp("2026-09-16", tz="UTC")
    bars = _bars(kinds, start, end)
    daily = bars.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                     "volume": "sum"}).dropna()
    s = GoldScar(params)
    targets = []

    def tick(ctx):
        out = s.on_tick(ctx)
        targets.append((ctx.et, out))
        return decode_targets(encode_targets(out))

    res = simulate(tick, ("QQQ",), {"QQQ": bars}, {"QQQ": daily}, None, None, start, end, CFG, LIM)
    return res, targets


def test_enters_itm_calls_at_1030_on_a_trend_day_and_holds_to_the_flatten():
    res, targets = _run(["trend", "flat", "flat"])
    assert not res.errors
    entries = [(t, out) for t, out in targets if out and t.hour * 60 + t.minute == 630]
    assert entries, "should enter at 10:30"
    t, out = entries[0]
    legs = [leg for s in out.values() for leg in s.legs]
    c = parse(legs[0].symbol)
    assert c.right == "C" and c.strike < 700 * 1.0044  # in the money
    assert c.expiry == t.date()
    # one entry, held into the afternoon, closed by the engine's 15:20 flatten (not before 15:00)
    assert res.opened == len(out)
    closed = pd.Timestamp(res.trades[0]["closed"]).tz_convert(ET)
    assert closed.hour * 60 + closed.minute >= 15 * 60
    assert all(t["opened"][:10] == "2026-09-14" for t in res.trades)  # flat days: no trade


def test_exits_when_qqq_crosses_back_through_the_open():
    res, _ = _run(["fail", "flat", "flat"])
    assert res.opened >= 1
    closed = pd.Timestamp(res.trades[0]["closed"]).tz_convert(ET)
    assert closed.hour * 60 + closed.minute <= 11 * 60


def test_large_size_is_split_into_tags_of_at_most_50():
    res, targets = _run(["trend", "flat", "flat"], {"budget_pct": 0.6})
    out = next(o for _, o in targets if o)
    assert all(s.qty <= 50 for s in out.values())


def test_synthetic_0dte_is_priced_off_calibrated_vix1d():
    now = pd.Timestamp("2026-09-14 12:30", tz=ET).tz_convert("UTC")
    chain = SyntheticChains()("SPY", 600.0, now, vix=18.0, vix1d=12.0)  # own cache: templates are per day
    e = chain.expiry(0)
    atm = chain.atm(e, "C")
    assert abs(float(atm["iv"]) - ZERO_DTE_PER_VIX1D * 0.12) < 0.01
    later = chain.atm(chain.expiry(0, min_dte=5), "C")
    assert abs(float(later["iv"]) - 0.18) < 0.02  # later expiries sit near VIX
    puts, calls = chain.puts(e), chain.calls(e)
    otm_put = float(puts[puts["strike"] == 595]["iv"].iloc[0])
    otm_call = float(calls[calls["strike"] == 605]["iv"].iloc[0])
    assert otm_put > otm_call > float(atm["iv"]) * 0.95  # a smile, steeper on the put side
