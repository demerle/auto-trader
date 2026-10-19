"""Turn a strategy's target structures into concrete changes, enforcing the rules.

Shared by the simulator (smoke tests, agent backtests) and the live engine, so a
bot faces the same cutoffs, flattening and risk limits in both.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from autotrader.options.book import OptionBook
from autotrader.options.config import GauntletConfig
from autotrader.options.helpers import Chain
from autotrader.options.occ import parse
from autotrader.options.risk import RiskError, structure_check, worst_value
from autotrader.options.strategy import Structure, Targets
from autotrader.risk.limits import OptionRiskLimits


@dataclass
class Change:
    tag: str
    kind: str  # open / close / resize
    legs: list[tuple[str, int]]  # (symbol, signed contracts to trade)
    units: int  # structure units traded
    reason: str
    note: str = ""

    @property
    def closing(self) -> bool:
        return self.kind == "close" or (self.kind == "resize" and self.units < 0)


@dataclass
class Plan:
    changes: list[Change] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (tag, why)


def _today_et(now: pd.Timestamp):
    return now.tz_convert("America/New_York").date()


def leg_price(chain: Chain | None, symbol: str, qty: float, side: str = "cross") -> float | None:
    """Per-share price to assume for trading `qty` of symbol: the far side ("cross"), or the mark."""
    if chain is None:
        return None
    row = chain.row(symbol)
    if row is None:
        return None
    if side == "mark":
        return float(row["mark"])
    return float(row["est_ask"] if qty > 0 else row["est_bid"])


def net_price(chains: Mapping[str, Chain], legs: list[tuple[str, int]], side: str = "cross") -> float | None:
    """Net price per structure unit (per share): + debit, - credit."""
    total = 0.0
    for sym, ratio in legs:
        p = leg_price(chains.get(parse(sym).underlying), sym, ratio, side)
        if p is None:
            return None
        total += ratio * p
    return total


def friday_flat_time(now: pd.Timestamp, limits: OptionRiskLimits) -> pd.Timestamp:
    hh, mm = map(int, limits.friday_flat_et.split(":"))
    return (pd.Timestamp(_today_et(now)) + pd.Timedelta(hours=hh, minutes=mm)) \
        .tz_localize("America/New_York").tz_convert("UTC")


def expiring_cutoff(symbol: str, now: pd.Timestamp, cfg: GauntletConfig) -> pd.Timestamp | None:
    """Alpaca's cutoff if this contract expires today, else None."""
    c = parse(symbol)
    if c.expiry != _today_et(now):
        return None
    return cfg.cutoff(c.underlying, c.expiry)


def plan_tick(now: pd.Timestamp, targets: Targets, book: OptionBook, chains: Mapping[str, Chain],
              equity: float, limits: OptionRiskLimits, cfg: GauntletConfig,
              week_last_session: bool, frozen: str | None = None) -> Plan:
    """frozen: a reason the bot may not add risk (busted, paused, kill switch)."""
    plan = Plan()
    today = _today_et(now)
    forced: dict[str, str] = {}
    flat_all = week_last_session and now >= friday_flat_time(now, limits)
    for tag, pos in book.open.items():
        for sym in pos.legs:
            c = parse(sym)
            if c.expiry < today:
                forced[tag] = "expired leg"
            cut = expiring_cutoff(sym, now, cfg)
            if cut is not None and now >= cut - pd.Timedelta(minutes=limits.flatten_cutoff_min):
                forced[tag] = f"expiry cutoff {cut.tz_convert('America/New_York'):%H:%M} ET"
        if flat_all:
            forced[tag] = "flat before the weekly scoring"
        if frozen == "busted":
            forced[tag] = "bot busted"

    def close(tag: str, reason: str) -> None:
        legs = [(s, -int(round(q))) for s, q in book.open[tag].legs.items()]
        plan.changes.append(Change(tag, "close", legs, book.units(tag), reason))

    for tag, why in forced.items():
        close(tag, why)
    if targets is None:
        return plan

    # Cash and worst-case equity as changes are applied (closes first, they free risk).
    open_tags = [t for t in book.open if t not in forced]
    for tag in open_tags:
        if tag not in targets or targets[tag].qty <= 0:
            close(tag, "strategy exit")
    worst_now = {t: worst_value(list(book.open[t].legs.items())) for t in open_tags
                 if t in targets and targets[t].qty > 0}
    cash = book.cash
    for ch in plan.changes:
        p = net_price(chains, ch.legs)
        cash -= (p if p is not None else 0.0) * 100

    n_open = len(worst_now)
    for tag, want in targets.items():
        if want.qty <= 0:
            continue
        legs = [(leg.symbol, int(leg.ratio)) for leg in want.legs]
        if tag in forced:
            continue
        if tag in book.open and tag in worst_now:
            have_key = tuple(sorted((leg.symbol, leg.ratio) for leg in book.legs_of(tag)))
            if have_key == want.key():
                diff = want.qty - book.units(tag)
                if diff < 0:
                    plan.changes.append(Change(tag, "resize", [(s, r * diff) for s, r in legs], diff,
                                               "strategy reduce", want.note))
                    continue
                if diff == 0:
                    continue
                ok, why, cost = _can_open(now, tag, legs, diff, chains, cash, equity, worst_now,
                                          n_open - 1, limits, cfg, frozen, flat_all)
                if ok:
                    cash -= cost
                    worst_now[tag] = worst_now[tag] + worst_value([(s, r * diff) for s, r in legs])
                    plan.changes.append(Change(tag, "resize", [(s, r * diff) for s, r in legs], diff,
                                               "strategy add", want.note))
                else:
                    plan.rejected.append((tag, why))
                continue
            # Different legs under the same tag: roll (close, then open below).
            close(tag, "roll")
            p = net_price(chains, plan.changes[-1].legs)
            cash -= (p if p is not None else 0.0) * 100
            worst_now.pop(tag, None)
            n_open -= 1
        ok, why, cost = _can_open(now, tag, legs, want.qty, chains, cash, equity, worst_now, n_open,
                                  limits, cfg, frozen, flat_all)
        if ok:
            cash -= cost
            worst_now[tag] = worst_value([(s, r * want.qty) for s, r in legs])
            n_open += 1
            plan.changes.append(Change(tag, "open", [(s, r * want.qty) for s, r in legs], want.qty,
                                       "strategy entry", want.note))
        else:
            plan.rejected.append((tag, why))
    return plan


def _can_open(now, tag, legs, qty, chains, cash, equity, worst_now, n_open, limits, cfg, frozen,
              flat_all) -> tuple[bool, str, float]:
    if frozen:
        return False, f"no new risk: {frozen}", 0.0
    if flat_all:
        return False, "no new risk: flat before the weekly scoring", 0.0
    if n_open >= limits.max_open_structures:
        return False, f"{limits.max_open_structures} structures already open", 0.0
    today = _today_et(now)
    for sym, _ in legs:
        try:
            c = parse(sym)
        except ValueError:
            return False, f"{sym} is not an OCC option symbol", 0.0
        if c.expiry < today:
            return False, f"{sym} has expired", 0.0
        cut = expiring_cutoff(sym, now, cfg)
        if cut is not None and now >= cut - pd.Timedelta(minutes=limits.open_cutoff_min):
            return False, f"{sym}: too close to the expiry cutoff", 0.0
    unit_price = net_price(chains, legs)
    if unit_price is None:
        return False, "a leg is not in the current chain", 0.0
    try:
        structure_check(legs, qty, unit_price, equity, limits.max_structure_risk_pct, limits.max_legs,
                        limits.max_contracts_per_order, cfg.universe)
    except RiskError as exc:
        return False, str(exc), 0.0
    cost = unit_price * qty * 100 + cfg.execution.fee_per_contract * sum(abs(r) * qty for _, r in legs)
    worst = worst_value([(s, r * qty) for s, r in legs])
    if cash - cost + sum(worst_now.values()) + worst < -1e-6:
        return False, "could lose more than the bot's equity", 0.0
    return True, "", cost
