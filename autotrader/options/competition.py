"""The weekly competition: reset Monday, score and cull the bottom N after the last session.

Week return = equity at the scoring time / start equity - 1. Everything is flat by then
(Friday flattening), so it is realized. Ties go to the smaller max drawdown. A bot with
fewer than `min_sessions_to_cull` sessions this week is exempt unless it busted.
"""

import json
import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.alerts import alert, post_report
from autotrader.config import get_settings
from autotrader.db.models import (
    OPTIONS, AgentJob, Bot, BotStage, CashFlow, EquitySnapshot, Order, Submission, WeeklyResult, utcnow,
)
from autotrader.engine.state import get_state, set_state
from autotrader.options.config import GauntletConfig
from autotrader.options.ledger import load_option_book
from autotrader.options.roster import active_option_bots

log = logging.getLogger(__name__)
WEEK_STATE = "gauntlet_week"  # {"started": week, "scored": week, "session_counted": date}
RESEARCHER = "options_researcher"


def week_of(now: pd.Timestamp) -> str:
    d = now.tz_convert("America/New_York").date()
    return (pd.Timestamp(d) - pd.Timedelta(days=d.weekday())).date().isoformat()


def _utc(dt) -> pd.Timestamp:
    t = pd.Timestamp(dt)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def current_result(session: Session, bot: Bot, week: str | None = None) -> WeeklyResult | None:
    q = select(WeeklyResult).where(WeeklyResult.bot_id == bot.id)
    if week:
        q = q.where(WeeklyResult.week == week)
    return session.scalars(q.order_by(WeeklyResult.id.desc()).limit(1)).first()


def current_week_start(session: Session, bot: Bot) -> float:
    r = current_result(session, bot)
    return r.start_equity if r is not None and r.status == "running" else bot.capital


def join_week(session: Session, bot: Bot, now: pd.Timestamp, start_equity: float) -> WeeklyResult:
    week = week_of(now)
    r = current_result(session, bot, week)
    if r is None:
        r = WeeklyResult(week=week, bot_id=bot.id, start_equity=start_equity, started_at=now.to_pydatetime())
        session.add(r)
        session.flush()
    return r


def start_week(session: Session, now: pd.Timestamp, cfg: GauntletConfig) -> list[str]:
    """Monday: every active options bot back to exactly `capital`, a fresh WeeklyResult each."""
    week = week_of(now)
    state = get_state(session, WEEK_STATE)
    if state.get("started") == week:
        return []
    lines = []
    for bot in active_option_bots(session):
        book = load_option_book(session, bot)
        if book.open:
            lines.append(f"{bot.slug}: still holds {list(book.open)}; reset skipped")
            alert(f"{bot.slug}: not flat at the weekly reset", f"Open: {list(book.open)}")
            continue
        delta = cfg.capital - book.cash
        if abs(delta) > 0.005:
            session.add(CashFlow(bot_id=bot.id, amount=delta, reason=f"weekly reset to ${cfg.capital:,.0f} ({week})",
                                 ts=now.to_pydatetime()))
        bot.capital = cfg.capital
        bot.paused_until = None
        join_week(session, bot, now, cfg.capital)
        lines.append(f"{bot.slug}: ${book.cash:,.2f} -> ${cfg.capital:,.0f}")
    set_state(session, WEEK_STATE, dict(state, started=week))
    return lines


def count_session(session: Session, now: pd.Timestamp) -> None:
    """After the close: every active bot with a running result gets one more session this week."""
    day = now.tz_convert("America/New_York").date().isoformat()
    state = get_state(session, WEEK_STATE)
    if state.get("session_counted") == day:
        return
    week = week_of(now)
    for bot in active_option_bots(session):
        r = current_result(session, bot, week)
        if r is not None and r.status == "running":
            r.sessions += 1
    set_state(session, WEEK_STATE, dict(state, session_counted=day))


def mark_busted(session: Session, bot: Bot, now: pd.Timestamp) -> None:
    r = current_result(session, bot, week_of(now))
    if r is not None and not r.busted:
        r.busted = True
        alert(f"{bot.slug} BUSTED", "Equity fell below the bust line; frozen until the weekly scoring.")


def _max_drawdown(session: Session, bot: Bot, since) -> float:
    eq = [s.equity for s in session.scalars(select(EquitySnapshot).where(
        EquitySnapshot.bot_id == bot.id, EquitySnapshot.ts >= since).order_by(EquitySnapshot.ts))]
    if not eq:
        return 0.0
    arr = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float(np.min(arr / np.where(peak > 0, peak, 1) - 1))


def trade_log(session: Session, bot: Bot, since=None) -> list[dict]:
    """Closed and open structures with entry/exit, P&L and the reasons given."""
    book = load_option_book(session, bot)
    orders = session.scalars(select(Order).join(Order.allocations)
                             .where(Order.account == bot.account)
                             .order_by(Order.id)).unique().all()
    notes: dict[str, list[str]] = {}
    for o in orders:
        a = o.allocations[0]
        if a.bot_id != bot.id or not a.filled_qty:
            continue
        r = a.reason or {}
        notes.setdefault(o.structure or "", []).append(
            f"{pd.Timestamp(o.created_at):%a %H:%M} {r.get('intent', '?')} x{a.filled_qty:g} "
            f"{' / '.join(f'{leg['symbol']} {leg['ratio']:+d}' for leg in o.legs or [])} "
            f"@ {o.filled_notional / o.filled_qty if o.filled_qty else 0:.2f} ({r.get('reason', '')}"
            f"{': ' + r['note'] if r.get('note') else ''})")
    out = []
    for c in book.closed:
        if since is not None and c.closed_at is not None and c.closed_at < _utc(since):
            continue
        out.append({"tag": c.tag, "pnl": round(c.pnl, 2), "opened": str(c.opened_at)[:16],
                    "closed": str(c.closed_at)[:16], "orders": notes.get(c.tag, [])[-6:]})
    return out


def autopsy(session: Session, bot: Bot, result: WeeklyResult, ranked: list[dict], market: dict) -> str:
    from autotrader.options import registry

    trades = trade_log(session, bot, result.started_at)
    wins = [t for t in trades if t["pnl"] > 0]
    try:
        thesis = registry.get(bot.strategy.name, bot.strategy.version).thesis
    except KeyError:
        thesis = (bot.strategy.backtest_metrics or {}).get("thesis", "")
    best = [r for r in ranked if r["slug"] != bot.slug][:2]
    lines = [
        f"{bot.slug} ({bot.strategy.name}@{bot.strategy.version}) was CULLED in the week of {result.week}: "
        f"rank {result.rank}/{len(ranked)}, week return {result.ret:+.1%}, max drawdown {result.max_drawdown:.1%}, "
        f"{len(trades)} closed structures ({len(wins)} winners){', BUSTED' if result.busted else ''}.",
        f"Thesis: {thesis}",
        f"Market that week: {json.dumps(market)}",
        "Winners this week: " + "; ".join(f"{r['slug']} {r['ret']:+.1%} ({r['trades']} trades): {r['thesis'][:200]}"
                                          for r in best),
        "Trade log (newest last):",
    ]
    for t in trades[-25:]:
        lines.append(f"- {t['tag']}: P&L ${t['pnl']:+,.2f}, {t['opened']} -> {t['closed']}")
        lines += [f"    {x}" for x in t["orders"]]
    return "\n".join(lines)[:12000]


def market_summary(archive, now: pd.Timestamp, week_start: pd.Timestamp) -> dict:
    out = {}
    for sym in ("SPY", "QQQ", "IWM", "^VIX"):
        try:
            d = archive.get(sym, "1d", start=week_start - pd.Timedelta(days=7), refresh=False)
            before = d[d.index < week_start.normalize()]
            if len(d) and len(before):
                out[sym] = {"week_change": round(float(d["close"].iloc[-1] / before["close"].iloc[-1] - 1), 4),
                            "close": round(float(d["close"].iloc[-1]), 2)}
        except Exception:  # noqa: BLE001
            continue
    return out


def score_week(session: Session, now: pd.Timestamp, cfg: GauntletConfig, archive=None,
               apply: bool = True, practice: bool = False) -> dict:
    """Rank this week's results; cull the bottom `cull` cullable bots (unless practice)."""
    week = week_of(now)
    state = get_state(session, WEEK_STATE)
    if apply and state.get("scored") == week:
        return {"week": week, "skipped": "already scored"}
    from autotrader.options import registry

    rows = []
    for r in session.scalars(select(WeeklyResult).where(WeeklyResult.week == week,
                                                         WeeklyResult.status == "running")):
        bot = r.bot
        book = load_option_book(session, bot)
        end = book.cash
        if book.open:  # should be flat by now; fall back to the last marked equity
            snap = session.scalars(select(EquitySnapshot).where(EquitySnapshot.bot_id == bot.id)
                                   .order_by(EquitySnapshot.ts.desc()).limit(1)).first()
            end = snap.equity if snap else book.cash
        trades = [c for c in book.closed if c.closed_at is not None and c.closed_at >= _utc(r.started_at)]
        try:
            thesis = registry.get(bot.strategy.name, bot.strategy.version).thesis
        except KeyError:
            thesis = ""
        rows.append({"result": r, "bot": bot, "slug": bot.slug, "end": float(end),
                     "ret": float(end) / r.start_equity - 1 if r.start_equity else 0.0,
                     "max_dd": _max_drawdown(session, bot, r.started_at), "trades": len(trades),
                     "cullable": r.busted or r.sessions >= cfg.min_sessions_to_cull,
                     "busted": r.busted, "thesis": thesis, "sessions": r.sessions})
    rows.sort(key=lambda x: (-1 if x["busted"] else 0, x["ret"], x["max_dd"]), reverse=True)
    for i, x in enumerate(rows, 1):
        x["rank"] = i
    cullable = [x for x in rows if x["cullable"]]
    doomed = {x["slug"] for x in cullable[-cfg.cull:]} if not practice and cfg.cull else set()
    market = market_summary(archive, now, pd.Timestamp(week, tz="UTC")) if archive is not None else {}
    summary = {"week": week, "practice": practice, "standings": [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()
         if k in ("rank", "slug", "ret", "max_dd", "trades", "busted", "cullable", "sessions", "end")}
        | {"culled": x["slug"] in doomed} for x in rows]}
    if not apply:
        return summary

    ranked = [{"slug": x["slug"], "ret": x["ret"], "trades": x["trades"], "thesis": x["thesis"]} for x in rows]
    for x in rows:
        r, bot = x["result"], x["bot"]
        r.end_equity, r.ret, r.max_drawdown, r.trades = x["end"], x["ret"], x["max_dd"], x["trades"]
        r.rank, r.scored_at = x["rank"], utcnow()
        if x["slug"] in doomed:
            r.status = "culled"
            r.autopsy = autopsy(session, bot, r, ranked, market)
            bot.stage, bot.stage_since = BotStage.ERADICATED, utcnow()
            bot.notes = r.autopsy
            session.add(AgentJob(agent=RESEARCHER, model="", status="queued", bot_id=bot.id,
                                 target={"group": OPTIONS, "replaces": bot.slug, "week": week},
                                 prompt=f"{bot.slug} was culled from the options gauntlet this week. "
                                        f"Study its autopsy (get_options_graveyard) and the survivors, then "
                                        f"build a replacement that wins next week.\n\nAutopsy:\n{r.autopsy[:4000]}"))
        else:
            r.status = "survived" if x["cullable"] else "exempt"
    set_state(session, WEEK_STATE, dict(state, scored=week))
    session.flush()
    _announce(summary)
    return summary


def _announce(summary: dict) -> None:
    lines = [f"**Options Gauntlet, week of {summary['week']}**" + (" (practice, no cull)" if summary["practice"] else ""), ""]
    for s in summary["standings"]:
        flag = " ❌ CULLED" if s["culled"] else (" (exempt)" if not s["cullable"] else "")
        lines.append(f"{s['rank']}. {s['slug']}: {s['ret']:+.1%} ({s['trades']} trades, max DD "
                     f"{s['max_dd']:.1%}){' BUSTED' if s['busted'] else ''}{flag}")
    culled = [s["slug"] for s in summary["standings"] if s["culled"]]
    if culled:
        lines += ["", f"Out: {', '.join(culled)}. Replacements are being researched for Monday."]
    url = f"{get_settings().dashboard_url}/options/weeks/{summary['week']}"
    try:
        if not post_report(f"Options Gauntlet: week of {summary['week']}", "\n".join(lines[2:]), url):
            alert("Options Gauntlet results", "\n".join(lines))
    except Exception:  # noqa: BLE001
        log.exception("could not post gauntlet results")


# ----- slots -----

def in_flight(session: Session) -> int:
    jobs = session.scalars(select(AgentJob).where(AgentJob.agent == RESEARCHER,
                                                  AgentJob.status.in_(("queued", "running")))).all()
    subs = session.scalars(select(Submission).where(
        Submission.interval == "opt", Submission.status.in_(("submitted", "gated", "audit_pending", "approved")))).all()
    return len(jobs) + len(subs)


def open_slots(session: Session, cfg: GauntletConfig) -> int:
    return max(cfg.slots - len(active_option_bots(session)) - in_flight(session), 0)


def fill_slots(session: Session, cfg: GauntletConfig, reason: str = "an open seat in the gauntlet") -> list[AgentJob]:
    """Queue research for empty seats, once the gauntlet has started (it is started by seeding it)."""
    if session.scalars(select(Bot.id).where(Bot.group == OPTIONS).limit(1)).first() is None:
        return []
    out = []
    for _ in range(open_slots(session, cfg)):
        job = AgentJob(agent=RESEARCHER, model="", status="queued", target={"group": OPTIONS, "reason": reason},
                       prompt="There is an open seat in the options gauntlet. Build a bot that wins next week.")
        session.add(job)
        out.append(job)
    session.flush()
    return out


def week_bounds(week: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date.fromisoformat(week)).tz_localize("America/New_York").tz_convert("UTC")
    return start, start + pd.Timedelta(days=7)
