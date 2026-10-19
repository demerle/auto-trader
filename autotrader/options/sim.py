"""Replay a strategy over past sessions in 5-minute steps (smoke tests and agent backtests).

Fills are immediate at the far side of the (pessimistic) spread plus fees. The same
planner as the live engine enforces cutoffs, Friday flattening and risk limits, and
the book is reset to the starting equity at the start of every week, like the gauntlet.
"""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.data.calendar import schedule
from autotrader.options.book import OptionBook
from autotrader.options.config import GauntletConfig
from autotrader.options.context import build_context, intrinsic, marks_for
from autotrader.options.occ import parse
from autotrader.options.planner import leg_price, plan_tick
from autotrader.options.risk import worst_value
from autotrader.options.synthetic import synthetic_chain
from autotrader.risk.limits import OptionRiskLimits

log = logging.getLogger(__name__)
STEP = pd.Timedelta(minutes=5)


@dataclass
class SimResult:
    equity: pd.Series  # per tick
    weeks: list[dict] = field(default_factory=list)  # {week, start, end, ret}
    trades: list[dict] = field(default_factory=list)  # closed structures
    opened: int = 0
    rejected: list[tuple[str, str, str]] = field(default_factory=list)  # (time, tag, why)
    errors: list[str] = field(default_factory=list)
    min_worst_equity: float = math.inf
    ticks: int = 0

    def summary(self) -> dict:
        rets = [w["ret"] for w in self.weeks if w["ret"] is not None]
        pnl = [t["pnl"] for t in self.trades]
        return {
            "ticks": self.ticks, "structures_opened": self.opened, "structures_closed": len(self.trades),
            "weeks": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.items()} for w in self.weeks],
            "avg_week_return": round(float(np.mean(rets)), 4) if rets else None,
            "best_week": round(max(rets), 4) if rets else None,
            "worst_week": round(min(rets), 4) if rets else None,
            "win_rate": round(sum(p > 0 for p in pnl) / len(pnl), 3) if pnl else None,
            "avg_trade_pnl": round(float(np.mean(pnl)), 2) if pnl else None,
            "biggest_win": round(max(pnl), 2) if pnl else None,
            "biggest_loss": round(min(pnl), 2) if pnl else None,
            "rejections": len(self.rejected),
            "rejection_examples": [f"{t} {tag}: {why}" for t, tag, why in self.rejected[:8]],
            "errors": self.errors[:5],
        }


def _asof(df: pd.DataFrame | None, t: pd.Timestamp, bar: pd.Timedelta) -> pd.DataFrame:
    """Bars complete by time t (index = bar start)."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.index + bar <= t]


def _level(series: pd.Series | None, t: pd.Timestamp) -> float | None:
    if series is None or series.empty:
        return None
    s = series[series.index <= t]
    return float(s.iloc[-1]) if len(s) else None


def rel_vols(daily: Mapping[str, pd.DataFrame], t: pd.Timestamp, n: int = 20) -> dict[str, float]:
    def rv(df):
        d = df[df.index < t.normalize()]["close"]
        r = np.log(d).diff().dropna().iloc[-n:]
        return float(r.std()) if len(r) > 5 else None
    spy = rv(daily["SPY"]) if "SPY" in daily else None
    out = {}
    for s, df in daily.items():
        v = rv(df)
        out[s] = float(np.clip(v / spy, 0.7, 5.0)) if v and spy else 1.0
    return out


def simulate(on_tick: Callable, underlyings: tuple[str, ...], bars5m: Mapping[str, pd.DataFrame],
             daily: Mapping[str, pd.DataFrame], vix: pd.Series | None, vix1d: pd.Series | None,
             start: pd.Timestamp, end: pd.Timestamp, cfg: GauntletConfig, limits: OptionRiskLimits,
             capital: float | None = None, chain_source: Callable | None = None,
             max_errors: int = 3) -> SimResult:
    """on_tick(ctx) -> targets. chain_source(underlying, spot, t, vix, vix1d, rel) -> Chain."""
    capital = capital or cfg.capital
    chain_source = chain_source or (lambda u, spot, t, v, v1, rel: synthetic_chain(u, spot, t, v, v1, rel))
    fee = cfg.execution.fee_per_contract
    book = OptionBook(capital)
    res = SimResult(equity=pd.Series(dtype=float))
    eq_points: dict[pd.Timestamp, float] = {}
    sessions = schedule(start, end)
    last_marks: dict[str, float] = {}
    week_key, week_start_eq, week_peak = None, capital, capital

    def close_week(final_eq: float) -> None:
        if week_key is not None:
            res.weeks.append({"week": week_key, "start": week_start_eq, "end": final_eq,
                              "ret": final_eq / week_start_eq - 1 if week_start_eq else None})

    for i, (day, row) in enumerate(sessions.iterrows()):
        iso = day.isocalendar()
        key = f"{iso[0]}-W{iso[1]:02d}"
        last_of_week = i + 1 == len(sessions) or sessions.index[i + 1].isocalendar()[1] != iso[1]
        if key != week_key:
            eq = book.equity(last_marks) if book.open else book.cash
            close_week(eq)
            if book.open:  # should never happen: everything is flat by Friday
                res.errors.append(f"{key}: positions carried into a new week")
            book.cash_flow(capital - eq)  # weekly reset, like the gauntlet
            week_key, week_start_eq = key, capital
        open_t, close_t = row["market_open"], row["market_close"]
        rel = rel_vols(daily, open_t)
        t = open_t + STEP
        while t <= close_t:
            spot, chains, bars = {}, {}, {}
            for u in underlyings:
                b = _asof(bars5m.get(u), t, STEP)
                if b.empty or b.index[-1] < open_t - pd.Timedelta(days=4):
                    continue
                bars[u] = b.iloc[-400:]
                spot[u] = float(b["close"].iloc[-1])
            v, v1 = _level(vix, t), _level(vix1d, t)
            for u, s in spot.items():
                chains[u] = chain_source(u, s, t, v or 18.0, v1, rel.get(u, 1.0))
            held = book.positions()
            marks = marks_for(held, chains, spot, last_marks)
            last_marks.update(marks)
            ctx = build_context(t, chains, spot, bars, {u: d[d.index < open_t.normalize()]
                                                        for u, d in daily.items()},
                                v, v1, book, marks, week_start_eq, open_t, close_t, last_of_week)
            try:
                targets = on_tick(ctx) if t < close_t else {}
            except Exception as exc:  # noqa: BLE001 - recorded; the smoke test fails on errors
                res.errors.append(f"{t}: {type(exc).__name__}: {exc}"[:300])
                if len(res.errors) >= max_errors:
                    res.equity = pd.Series(eq_points)
                    return res
                targets = None
            plan = plan_tick(t, targets, book, chains, book.equity(marks), limits, cfg, last_of_week)
            for tag, why in plan.rejected:
                res.rejected.append((f"{t.tz_convert('America/New_York'):%m-%d %H:%M}", tag, why))
            for ch in plan.changes:
                note = (targets or {}).get(ch.tag).note if targets and ch.tag in (targets or {}) else ch.reason
                for sym, q in ch.legs:
                    u = parse(sym).underlying
                    p = leg_price(chains.get(u), sym, q)
                    if p is None:  # not in the chain any more: settle at intrinsic
                        p = last_marks.get(sym, intrinsic(sym, spot.get(u, 0.0)))
                    book.fill(ch.tag, sym, q, p, fee * abs(q), t, note)
                if ch.kind == "open":
                    res.opened += 1
            # Anything expiring today and still open at the close settles at intrinsic.
            if t >= close_t:
                today = t.tz_convert("America/New_York").date()
                for tag in list(book.open):
                    for sym, q in list(book.open.get(tag, _Empty).legs.items()):
                        if parse(sym).expiry <= today:
                            u = parse(sym).underlying
                            book.fill(tag, sym, -q, intrinsic(sym, spot.get(u, 0.0)), 0.0, t, "expired")
            marks = marks_for(book.positions(), chains, spot, last_marks)
            eq = book.equity(marks)
            eq_points[t] = eq
            worst = book.cash + sum(worst_value(list(p.legs.items())) for p in book.open.values())
            res.min_worst_equity = min(res.min_worst_equity, worst)
            res.ticks += 1
            t += STEP
        if last_of_week:
            close_week(book.equity(last_marks) if book.open else book.cash)
            week_key = None
    if week_key is not None:
        close_week(book.equity(last_marks) if book.open else book.cash)
    for c in book.closed:
        res.trades.append({"tag": c.tag, "pnl": c.pnl, "opened": str(c.opened_at), "closed": str(c.closed_at)})
    res.equity = pd.Series(eq_points)
    return res


class _Empty:
    legs: dict = {}
