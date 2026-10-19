"""Read models for the gauntlet (CLI, dashboard, reports, agent tools)."""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.data.calendar import schedule
from autotrader.db.models import OPTIONS, Bot, BotStage, EquitySnapshot, Order, WeeklyResult
from autotrader.engine.state import KILL_SWITCH, account_key, get_state
from autotrader.options.competition import week_bounds, week_of
from autotrader.options.config import GauntletConfig, load_gauntlet_config
from autotrader.options.ledger import load_option_book
from autotrader.options.occ import parse
from autotrader.options.risk import worst_value

RIGHT = {"C": "call", "P": "put"}


def _utc(dt) -> pd.Timestamp:
    t = pd.Timestamp(dt)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def describe(legs: list) -> str:
    """Legs [[symbol, ratio], ...] in plain words, e.g. 'SPY 10/20 770/775 call spread'."""
    try:
        cs = [(parse(s), int(r)) for s, r in legs]
    except ValueError:
        return " / ".join(f"{s} {r:+}" for s, r in legs)
    if not cs:
        return "-"
    u = cs[0][0].underlying
    exps = sorted({c.expiry for c, _ in cs})
    when = "/".join(f"{e:%m/%d}" for e in exps)

    def k(x: float) -> str:
        return f"{x:g}"
    rights = {c.right for c, _ in cs}
    if len(cs) == 1:
        c, r = cs[0]
        return f"{'long' if r > 0 else 'short'}{'' if abs(r) == 1 else f' {abs(r)}x'} {u} {when} {k(c.strike)} {RIGHT[c.right]}"
    if len(cs) == 2 and len(rights) == 1 and len(exps) == 1 and cs[0][1] == -cs[1][1]:
        lo, hi = sorted(cs, key=lambda x: x[0].strike)
        long_leg = lo if lo[1] > 0 else hi
        debit = (long_leg is lo) == (cs[0][0].right == "C")
        return f"{u} {when} {k(lo[0].strike)}/{k(hi[0].strike)} {RIGHT[lo[0].right]} {'debit' if debit else 'credit'} spread"
    if len(cs) == 2 and rights == {"C", "P"} and cs[0][1] == cs[1][1]:
        strikes = sorted({c.strike for c, _ in cs})
        kind = "straddle" if len(strikes) == 1 else "strangle"
        return f"{'long' if cs[0][1] > 0 else 'short'} {u} {when} {'/'.join(k(x) for x in strikes)} {kind}"
    if len(cs) == 4 and rights == {"C", "P"}:
        strikes = sorted(c.strike for c, _ in cs)
        short = sum(r for c, r in cs if c.strike in (strikes[1], strikes[2])) < 0
        return f"{'short' if short else 'long'} {u} {when} {'/'.join(k(x) for x in strikes)} iron condor"
    if len(cs) == 3 and sorted(abs(r) for _, r in cs) == [1, 1, 2] and len(rights) == 1:
        strikes = sorted(c.strike for c, _ in cs)
        return f"{u} {when} {'/'.join(k(x) for x in strikes)} {RIGHT[cs[0][0].right]} butterfly"
    return " / ".join(f"{r:+d} {u} {c.expiry:%m/%d} {k(c.strike)}{c.right}" for c, r in cs)


def _latest(session: Session, bot_id: int, before=None):
    q = select(EquitySnapshot).where(EquitySnapshot.bot_id == bot_id)
    if before is not None:
        q = q.where(EquitySnapshot.ts < before)
    return session.scalars(q.order_by(EquitySnapshot.ts.desc()).limit(1)).first()


def week_days(week: str, now: pd.Timestamp) -> tuple[int, int]:
    start, end = week_bounds(week)
    days = schedule(start, end - pd.Timedelta(days=1)).index
    today = now.tz_convert("America/New_York").normalize().tz_localize(None)
    done = sum(1 for d in days if d.tz_localize(None) <= today)
    return done, len(days)


def open_risk(positions: dict) -> float:
    """Worst-case further loss of the open structures: their mark value minus their worst value."""
    total = 0.0
    for p in (positions or {}).values():
        legs = [(s, r * p["qty"]) for s, r in p["legs"]]
        try:
            worst = worst_value(legs)
        except Exception:  # noqa: BLE001
            continue
        total += p["mark"] * p["qty"] * 100 - worst
    return total


def standings(session: Session, now: pd.Timestamp | None = None, cfg: GauntletConfig | None = None,
              week: str | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    cfg = cfg or load_gauntlet_config()
    week = week or week_of(now)
    day_start = now.tz_convert("America/New_York").normalize().tz_convert("UTC").to_pydatetime()
    results = {r.bot_id: r for r in session.scalars(select(WeeklyResult).where(WeeklyResult.week == week))}
    bots = {b.id: b for b in session.scalars(select(Bot).where(Bot.group == OPTIONS))
            if b.id in results or b.stage in (BotStage.PAPER_PROBATION, BotStage.PAPER_PROVEN)}
    rows = []
    for bot_id, bot in bots.items():
        r = results.get(bot_id)
        start = r.start_equity if r else bot.capital
        snap = _latest(session, bot_id)
        scored = r is not None and r.status not in ("running",)
        equity = r.end_equity if scored and r.end_equity is not None else (snap.equity if snap else start)
        before = _latest(session, bot_id, before=day_start)
        prev = before.equity if before and (r is None or _utc(before.ts) >= _utc(r.started_at)) else start
        since = r.started_at if r else bot.created_at
        trades = sum(1 for c in load_option_book(session, bot).closed
                     if c.closed_at is not None and c.closed_at >= _utc(since))
        positions = (snap.positions if snap else {}) or {}
        status = r.status if r else "waiting"
        if r is not None and r.status == "running":
            status = "busted" if r.busted else ("paused" if bot.paused_until and _utc(bot.paused_until) > now
                                                else "new (exempt)" if r.sessions + 1 < cfg.min_sessions_to_cull
                                                else "active")
        rows.append({"slug": bot.slug, "strategy": f"{bot.strategy.name}@{bot.strategy.version}",
                     "author": bot.strategy.author, "start": start, "equity": equity,
                     "ret": equity / start - 1 if start else 0.0, "day_pnl": equity - prev,
                     "trades": trades, "open": len(positions), "open_risk": open_risk(positions),
                     "positions": positions, "status": status, "busted": bool(r and r.busted),
                     "rank_final": r.rank if r else None, "max_dd": r.max_drawdown if r else None,
                     "sessions": r.sessions if r else 0, "description": bot.strategy.description,
                     "last_plan": bot.target_qty if isinstance(bot.target_qty, dict) else {}})
    rows.sort(key=lambda x: (x["rank_final"] is None, x["rank_final"] or 0,
                             -1 if x["busted"] else 0, -x["ret"]))
    if all(x["rank_final"] is None for x in rows):
        rows.sort(key=lambda x: (x["busted"], -x["ret"]))
    for i, x in enumerate(rows, 1):
        x["rank"] = i
    done, total = week_days(week, now)
    kill = get_state(session, account_key(KILL_SWITCH, cfg.account))
    return {"week": week, "day": done, "days": total, "rows": rows,
            "cut_line": max(len(rows) - cfg.cull, 0), "cull": cfg.cull, "practice": bool(
                cfg.practice_until and week <= cfg.practice_until),
            "killed": bool(kill.get("on")), "kill_reason": kill.get("reason"),
            "score_et": cfg.score_et, "capital": cfg.capital}


def trade_feed(session: Session, limit: int = 100, bot_slug: str | None = None, since=None) -> list[dict]:
    cfg = load_gauntlet_config()
    q = select(Order).where(Order.account == cfg.account).order_by(Order.id.desc()).limit(limit * 3)
    if since is not None:
        q = q.where(Order.created_at >= since)
    slugs = {b.id: b.slug for b in session.scalars(select(Bot).where(Bot.group == OPTIONS))}
    out = []
    for o in session.scalars(q):
        a = o.allocations[0] if o.allocations else None
        slug = slugs.get(a.bot_id, "?") if a else "?"
        if bot_slug and slug != bot_slug:
            continue
        reason = (a.reason or {}) if a else {}
        units = abs(o.qty)
        legs = o.legs or []
        filled_units = (o.filled_qty / abs(legs[0]["ratio"]) if len(legs) == 1 and legs else o.filled_qty)
        price = (o.filled_notional / o.filled_qty) if o.filled_qty else o.limit_price
        out.append({"id": o.id, "time": f"{_utc(o.created_at).tz_convert('America/New_York'):%a %m-%d %H:%M:%S}",
                    "ts": _utc(o.created_at).isoformat(), "bot": slug, "tag": o.structure or "",
                    "intent": reason.get("intent", "?"), "structure": describe([[x["symbol"], x["ratio"]] for x in legs]),
                    "qty": int(units), "filled": filled_units, "status": o.status, "price": price,
                    "limit": o.limit_price, "mid": o.ref_price, "chase": o.chase,
                    "reason": reason.get("reason", "") + (f": {reason['note']}" if reason.get("note") else ""),
                    "error": reason.get("error")})
        if len(out) >= limit:
            break
    return out


def equity_lines(session: Session, week: str, now: pd.Timestamp | None = None) -> dict[str, list[tuple]]:
    """slug -> [(ts, equity / start - 1)] for this week's chart."""
    start, end = week_bounds(week)
    out = {}
    for r in session.scalars(select(WeeklyResult).where(WeeklyResult.week == week)):
        pts = [(s.ts, s.equity / r.start_equity - 1) for s in session.scalars(
            select(EquitySnapshot).where(EquitySnapshot.bot_id == r.bot_id,
                                         EquitySnapshot.ts >= max(_utc(r.started_at), start).to_pydatetime(),
                                         EquitySnapshot.ts < end.to_pydatetime())
            .order_by(EquitySnapshot.ts))]
        if pts:
            out[r.bot.slug] = pts
    return out


def weeks(session: Session) -> list[dict]:
    rows = {}
    for r in session.scalars(select(WeeklyResult).order_by(WeeklyResult.week.desc())):
        w = rows.setdefault(r.week, {"week": r.week, "bots": 0, "culled": [], "winner": None, "best": None,
                                     "scored": False})
        w["bots"] += 1
        if r.status == "culled":
            w["culled"].append(r.bot.slug)
        if r.scored_at:
            w["scored"] = True
        if r.ret is not None and (w["best"] is None or r.ret > w["best"]):
            w["best"], w["winner"] = r.ret, r.bot.slug
    return list(rows.values())


def week_detail(session: Session, week: str) -> list[dict]:
    out = []
    for r in session.scalars(select(WeeklyResult).where(WeeklyResult.week == week)
                             .order_by(WeeklyResult.rank.is_(None), WeeklyResult.rank)):
        out.append({"slug": r.bot.slug, "strategy": f"{r.bot.strategy.name}@{r.bot.strategy.version}",
                    "rank": r.rank, "ret": r.ret, "start": r.start_equity, "end": r.end_equity,
                    "max_dd": r.max_drawdown, "trades": r.trades, "sessions": r.sessions, "status": r.status,
                    "busted": r.busted, "autopsy": r.autopsy})
    return out


def all_time(session: Session) -> list[dict]:
    """Cumulative (compounded) weekly return and weeks survived, per options bot."""
    agg: dict[str, dict] = {}
    for r in session.scalars(select(WeeklyResult).where(WeeklyResult.ret.is_not(None))):
        a = agg.setdefault(r.bot.slug, {"slug": r.bot.slug, "strategy": r.bot.strategy.name, "weeks": 0,
                                        "survived": 0, "growth": 1.0, "best": None, "stage": r.bot.stage.value,
                                        "author": r.bot.strategy.author})
        a["weeks"] += 1
        a["survived"] += r.status in ("survived", "exempt")
        a["growth"] *= 1 + r.ret
        a["best"] = r.ret if a["best"] is None else max(a["best"], r.ret)
    rows = [dict(a, cumulative=a["growth"] - 1) for a in agg.values()]
    return sorted(rows, key=lambda x: -x["cumulative"])


def graveyard(session: Session, limit: int = 50) -> list[dict]:
    out = []
    for b in session.scalars(select(Bot).where(Bot.group == OPTIONS, Bot.stage == BotStage.ERADICATED)
                             .order_by(Bot.stage_since.desc()).limit(limit)):
        last = session.scalars(select(WeeklyResult).where(WeeklyResult.bot_id == b.id)
                               .order_by(WeeklyResult.id.desc()).limit(1)).first()
        out.append({"slug": b.slug, "strategy": f"{b.strategy.name}@{b.strategy.version}",
                    "author": b.strategy.author, "died": str(b.stage_since)[:10],
                    "week": last.week if last else None, "ret": last.ret if last else None,
                    "autopsy": b.notes})
    return out
