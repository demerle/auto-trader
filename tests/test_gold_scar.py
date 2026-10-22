"""gold_scar v2: the packaged model runs on plain bars, the strategy trades its calls, bar_symbols reach the
context; calibrated 0DTE pricing."""

import numpy as np
import pandas as pd
import pytest

from autotrader.data.calendar import schedule
from autotrader.options.builtin import gold_scar
from autotrader.options.builtin.gold_scar import GoldScar
from autotrader.options.config import GauntletConfig
from autotrader.options.ml import features as F
from autotrader.options.ml import model
from autotrader.options.occ import parse
from autotrader.options.sim import simulate
from autotrader.options.strategy import decode_targets, encode_targets
from autotrader.options.synthetic import ZERO_DTE_PER_VIX1D, SyntheticChains
from autotrader.risk.limits import OptionRiskLimits

CFG = GauntletConfig(underlyings={"etfs": ["SPY", "QQQ", "IWM"]})
LIM = OptionRiskLimits()
ET = "America/New_York"
START, END = pd.Timestamp("2026-09-08", tz="UTC"), pd.Timestamp("2026-09-16", tz="UTC")


def _bars(px: float, start=START, end=END, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, c = [], px
    for _, row in schedule(start, end).iterrows():
        for i in range(78):
            o = c
            c = o * (1 + 0.0008 * rng.standard_normal())
            rows.append((row["market_open"] + pd.Timedelta(minutes=5 * i), o, max(o, c) * 1.0002,
                         min(o, c) * 0.9998, c, 1e5 * (1 + rng.random())))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")


def _all_bars():
    return {s: _bars(100.0 + 50 * i, seed=i) for i, s in enumerate(F.SYMBOLS)}


def test_model_predicts_a_z_per_target_from_plain_bars():
    bars = _all_bars()
    today = pd.Timestamp("2026-09-15")
    asof = {s: b[b.index < pd.Timestamp("2026-09-15 13:45", tz="UTC")] for s, b in bars.items()}  # 9:45 ET
    z = model.predict(asof, 18.0, 2, today)
    assert set(z) == set(F.TARGETS) and all(np.isfinite(v) for v in z.values())
    assert model.predict(asof, 18.0, 2, pd.Timestamp("2026-09-16")) == {}  # not today's bars: no call
    # missing cross-asset feeds count as flat rather than failing
    assert set(model.predict({s: asof[s] for s in F.TARGETS}, None, 2, today)) == set(F.TARGETS)


def test_numpy_transformer_is_causal():
    W = model.weights()
    x = np.random.default_rng(1).standard_normal((100, F.BAR_DIM))
    s = np.zeros(F.STATIC_DIM)
    a = model.transformer(W, "t0/", x, s, int(W["tfm_layers"]), int(W["tfm_heads"]))
    x2 = x.copy()
    x2[60:] += 3.0
    b = model.transformer(W, "t0/", x2, s, int(W["tfm_layers"]), int(W["tfm_heads"]))
    assert np.allclose(a[:60], b[:60]) and not np.allclose(a[60:], b[60:])


def _run(monkeypatch, z, params=None):
    monkeypatch.setattr(gold_scar, "predict", lambda bars, vix, k, today: dict(z))
    bars = {"SPY": _bars(700.0)}
    daily = {"SPY": bars["SPY"].resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                                    "close": "last", "volume": "sum"}).dropna()}
    s = GoldScar(params)
    targets = []

    def tick(ctx):
        out = s.on_tick(ctx)
        targets.append((ctx.et, out))
        return decode_targets(encode_targets(out))

    start = pd.Timestamp("2026-09-15", tz="UTC")
    res = simulate(tick, s.underlyings, bars, daily, None, None, start, start, CFG, LIM, bar_symbols=s.bar_symbols)
    return res, targets


def test_strong_call_buys_deep_itm_0dte_calls_at_945_and_holds_to_the_flatten(monkeypatch):
    res, targets = _run(monkeypatch, {"SPY": 1.6, "QQQ": 0.2, "IWM": -2.0})
    assert not res.errors
    entries = [(t, out) for t, out in targets if out]
    t, out = entries[0]
    assert (t.hour, t.minute) == (9, 45)
    legs = [leg for s in out.values() for leg in s.legs]
    assert {parse(leg.symbol).underlying for leg in legs} == {"SPY"}  # IWM is not traded by default
    c = parse(legs[0].symbol)
    assert c.right == "C" and c.strike <= 700 * 1.01 - 4 and c.expiry == t.date()
    closed = pd.Timestamp(res.trades[0]["closed"]).tz_convert(ET)
    assert closed.hour * 60 + closed.minute >= 15 * 60


def test_weak_call_stays_out_and_size_splits_into_tags_of_50(monkeypatch):
    res, targets = _run(monkeypatch, {"SPY": -0.9})
    assert res.opened == 0
    res, targets = _run(monkeypatch, {"SPY": -1.2}, {"budget_pct": 0.6})
    out = next(o for _, o in targets if o)
    assert all(s.qty <= 50 for s in out.values())
    assert parse(next(iter(out.values())).legs[0].symbol).right == "P"


def test_bar_symbols_reach_the_context():
    seen = {}

    def tick(ctx):
        seen.update({s: len(b) for s, b in ctx.bars.items()})
        return {}

    bars = {"SPY": _bars(700.0), "HYG": _bars(80.0)}
    start = pd.Timestamp("2026-09-15", tz="UTC")
    simulate(tick, ("SPY",), bars, {}, None, None, start, start, CFG, LIM, bar_symbols=("HYG",))
    assert seen.get("HYG", 0) > 0 and "HYG" not in CFG.universe


@pytest.mark.parametrize("missing", [5, 40])
def test_sessions_carry_prices_over_missing_slots(missing):
    b = _bars(100.0)
    b = b.drop(b.index[missing])
    days, a = F.sessions(b)
    assert not np.isnan(a["c"]).any() and a["c"].shape[1] == F.SLOTS


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
