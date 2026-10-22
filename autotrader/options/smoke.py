"""The gauntlet's entry test: a new bot must trade, and must never be able to lose more than it has.

It replays the last `smoke.sessions` sessions on synthetic chains (pessimistic
spreads) through the same planner and limits as the live engine. Profit is not
required; the weekly competition decides that.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from autotrader.data.calendar import schedule
from autotrader.options.config import GauntletConfig
from autotrader.options.history import load_history
from autotrader.options.sim import simulate
from autotrader.options.strategy import OptionStrategy
from autotrader.risk.limits import OptionRiskLimits
from autotrader.sandbox.runner import run_once


@dataclass
class SmokeResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def window(sessions: int, now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The last `sessions` complete sessions before today."""
    now = now or pd.Timestamp.now(tz="UTC")
    days = schedule(now - pd.Timedelta(days=sessions * 2 + 10), now - pd.Timedelta(days=1)).index
    days = days[days < now.normalize()][-sessions:]
    return days[0], days[-1]


def simulate_file(path: str | Path, underlyings: tuple[str, ...], cfg: GauntletConfig,
                  limits: OptionRiskLimits, start: pd.Timestamp, end: pd.Timestamp,
                  history: dict | None = None, params: dict | None = None, timeout: float = 600.0) -> dict:
    """Run the simulation inside a sandbox worker (agent code never runs in this process)."""
    history = history or load_history(underlyings, start, end)
    return run_once(path, "opt_simulate", timeout=timeout, params=params, load_op="opt_load",
                    history=history, start=start, end=end, cfg=cfg, limits=limits)


def simulate_class(cls: type[OptionStrategy], cfg: GauntletConfig, limits: OptionRiskLimits,
                   start: pd.Timestamp, end: pd.Timestamp, history: dict | None = None,
                   params: dict | None = None) -> dict:
    """In-process (trusted builtins only)."""
    from autotrader.options.strategy import decode_targets, encode_targets

    strategy = cls(params)
    history = history or load_history(tuple(cls.underlyings), start, end, bar_symbols=tuple(cls.bar_symbols))
    res = simulate(lambda ctx: decode_targets(encode_targets(strategy.on_tick(ctx))),
                   tuple(cls.underlyings), history["bars5m"], history["daily"], history["vix"],
                   history["vix1d"], start, end, cfg, limits, bar_symbols=tuple(cls.bar_symbols))
    return {"summary": res.summary(), "min_worst_equity": res.min_worst_equity,
            "trades": res.trades[-200:], "errors": res.errors}


def judge(out: dict, cfg: GauntletConfig) -> SmokeResult:
    s = out["summary"]
    reasons = []
    if out.get("errors"):
        reasons.append(f"the strategy raised errors: {out['errors'][:2]}")
    if s["structures_opened"] < cfg.smoke.min_trades:
        reasons.append(f"opened {s['structures_opened']} structures in {cfg.smoke.sessions} sessions "
                       f"(needs >= {cfg.smoke.min_trades}): it must actually trade")
    if out["min_worst_equity"] < -1.0:
        reasons.append(f"worst-case equity reached ${out['min_worst_equity']:,.0f}: "
                       "it could lose more than its money")
    return SmokeResult(not reasons, reasons, s)


def smoke_test(target, cfg: GauntletConfig, limits: OptionRiskLimits, underlyings: tuple[str, ...] = (),
               now: pd.Timestamp | None = None, history: dict | None = None,
               params: dict | None = None) -> SmokeResult:
    """target: a builtin OptionStrategy class, or the path of a generated strategy file."""
    start, end = window(cfg.smoke.sessions, now)
    try:
        if isinstance(target, type):
            out = simulate_class(target, cfg, limits, start, end, history, params)
        else:
            out = simulate_file(target, underlyings, cfg, limits, start, end, history, params)
    except Exception as exc:  # noqa: BLE001 - sandbox failures are a failed test, not a crash
        return SmokeResult(False, [f"simulation failed: {str(exc)[:500]}"])
    return judge(out, cfg)
