"""The daily report: built after the evaluation, stored, and posted to Discord.

The supervisor agent later adds commentary, which is posted as a follow-up.
"""

import logging
from collections import Counter

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from autotrader.alerts import post_report
from autotrader.config import get_settings
from autotrader.db.models import ARENA, AgentJob, Bot, Evaluation, Report, Submission, utcnow
from autotrader.engine.state import get_state
from autotrader.views import (
    account_overview, et_day_bounds, league, num, recent_alerts, today_et,
)

log = logging.getLogger(__name__)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def _usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def spy_day_return(spy_returns: pd.Series | None, day: str) -> float | None:
    if spy_returns is None or spy_returns.empty:
        return None
    hit = spy_returns[spy_returns.index.strftime("%Y-%m-%d") == day]
    return float(hit.iloc[-1]) if len(hit) else None


def build_daily_report(session: Session, day: str, spy_returns: pd.Series | None, cfg, policy,
                       now: pd.Timestamp | None = None) -> tuple[str, dict]:
    from autotrader.orchestrator import budget, slots
    from autotrader.orchestrator.control import PAUSED

    now = now or pd.Timestamp.now(tz="UTC")
    start, end = (t.to_pydatetime() for t in et_day_bounds(day))
    acct = account_overview(session, now)
    spy_day = spy_day_return(spy_returns, day)
    table = league(session, spy_returns)

    verdicts = session.execute(select(Evaluation, Bot).join(Bot, Bot.id == Evaluation.bot_id)
                               .where(Evaluation.ts >= start, Evaluation.ts < end)
                               .order_by(Evaluation.id)).all()
    decisions = Counter(e.decision for e, _ in verdicts)
    notable = [{"slug": b.slug, "decision": e.decision, "rationale": e.rationale}
               for e, b in verdicts if e.decision != "keep"]
    new_bots = [{"slug": b.slug, "strategy": f"{b.strategy.name}@{b.strategy.version}",
                 "interval": b.strategy.interval, "author": b.strategy.author}
                for b in session.scalars(select(Bot).where(Bot.created_at >= start, Bot.created_at < end, Bot.group == ARENA))]

    jobs = session.scalars(select(AgentJob).where(AgentJob.finished_at >= start,
                                                  AgentJob.finished_at < end)).all()
    job_counts = Counter((j.agent, j.status) for j in jobs if j.status not in ("queued", "running"))
    spent_today = sum(j.cost_usd or 0.0 for j in session.scalars(
        select(AgentJob).where(AgentJob.started_at >= start, AgentJob.started_at < end)))
    chk = budget.check(session, "researcher", cfg, now)
    subs = [{"name": f"{x.name}@{x.version}", "interval": x.interval, "status": x.status,
             "reasons": list(x.reasons or [])[:3]}
            for x in session.scalars(select(Submission).where(Submission.created_at >= start,
                                                              Submission.created_at < end))]
    paused = get_state(session, PAUSED)
    alerts = recent_alerts(session, limit=50, since=start, until=end)
    n_active = len(slots.active_bots(session))
    n_open = slots.open_slots(session, policy)

    lines = [f"**Equity** {_usd(acct['equity'])} ({'+' if acct['day_pnl'] >= 0 else ''}"
             f"{_usd(acct['day_pnl'])}, {_pct(acct['day_return'])} today) · SPY {_pct(spy_day)}",
             f"Kill switch: {'**ON** — ' + str(acct['kill_reason']) if acct['killed'] else 'off'}",
             f"Bots: {n_active}/{policy.slots} slots used, {n_open} open for research"]

    lines += ["", "## League (paper Sharpe)"]
    shown = table if len(table) <= 6 else table[:3] + [None] + table[-3:]
    for i, r in enumerate(shown):
        if r is None:
            lines.append("…")
            continue
        rank = i + 1 if i < 3 or len(table) <= 6 else len(table) - (len(shown) - 1 - i)
        sharpe = "n/a" if r["sharpe"] is None else f"{r['sharpe']:.2f}"
        star = " ⭐ live-eligible" if r["live_eligible"] else ""
        lines.append(f"{rank}. **{r['slug']}** ({r['interval']}) Sharpe {sharpe}, return "
                     f"{_pct(r['total_return'])} vs SPY {_pct(r['spy_return'])}, {r['trades']} trades, "
                     f"{r['days']}d{star}")
    if not table:
        lines.append("No bots under evaluation yet.")

    gauntlet = _gauntlet(session, now, start)
    if gauntlet:
        lines += ["", "## Options Gauntlet"] + gauntlet["lines"]

    lines += ["", "## Today's verdicts"]
    if verdicts:
        lines.append(", ".join(f"{n} {d}" for d, n in sorted(decisions.items())))
        lines += [f"- **{v['decision']}** {v['slug']}: {v['rationale']}" for v in notable]
    else:
        lines.append("No evaluation ran today.")

    if new_bots:
        lines += ["", "## New bots"]
        lines += [f"- **{b['slug']}** {b['strategy']} ({b['interval']}) by {b['author']}" for b in new_bots]

    lines += ["", "## Agents",
              f"Spent today ${spent_today:.2f} · {chk}" + (f" · **paused**: {paused.get('reason')}"
                                                            if paused else "")]
    if job_counts:
        lines.append("Jobs: " + ", ".join(f"{a} {s} ×{n}" for (a, s), n in sorted(job_counts.items())))
    lines += [f"- {s['name']} ({s['interval']}): {s['status']}"
              + (f" — {'; '.join(s['reasons'])}" if s["reasons"] and s["status"] == "gate_failed" else "")
              for s in subs]

    if alerts:
        lines += ["", f"## Alerts ({len(alerts)})"]
        for a in alerts[:10]:
            t = pd.Timestamp(a["ts"]).tz_convert("America/New_York")
            lines.append(f"- {t:%H:%M} {a['title']}")

    data = {"day": day, "account": acct, "spy_day_return": num(spy_day), "league": table,
            "decisions": dict(decisions), "verdicts": notable, "new_bots": new_bots,
            "agents": {"spent_today": spent_today, "spend": chk.spend, "limits": chk.limits,
                       "jobs": [{"agent": a, "status": s, "n": n} for (a, s), n in job_counts.items()],
                       "submissions": subs, "paused": paused or None},
            "alerts": alerts[:20], "slots": {"total": policy.slots, "bots": n_active, "open": n_open},
            "gauntlet": (gauntlet or {}).get("data")}
    return "\n".join(lines), data


def _gauntlet(session: Session, now: pd.Timestamp, day_start) -> dict | None:
    """Standings of the options gauntlet (its own account, weekly competition)."""
    from autotrader.options import views as ov

    try:
        st = ov.standings(session, now)
    except Exception:  # noqa: BLE001 - the arena report must not fail because of the gauntlet
        log.exception("gauntlet section failed")
        return None
    if not st["rows"]:
        return None
    feed = [t for t in ov.trade_feed(session, limit=500, since=day_start) if t["filled"]]
    lines = [f"Week of {st['week']}, day {st['day']}/{st['days']}"
             + (" (practice, no cull)" if st["practice"] else f"; bottom {st['cull']} culled after the last session")
             + f" · {len(feed)} fills today" + (" · **options kill switch ON**" if st["killed"] else "")]
    for r in st["rows"]:
        cut = " ✂️" if r["rank"] > st["cut_line"] else ""
        lines.append(f"{r['rank']}. **{r['slug']}** {_pct(r['ret'])} this week ({'+' if r['day_pnl'] >= 0 else ''}"
                     f"{_usd(r['day_pnl'])} today), {r['trades']} closed trades, {r['open']} open"
                     f"{' · ' + r['status'] if r['status'] not in ('active', 'running') else ''}"
                     f" · {r['record_text']}{cut}")
    data = {"week": st["week"], "standings": [{k: r[k] for k in ("rank", "slug", "ret", "day_pnl", "trades",
                                                                  "open", "status", "record")} for r in st["rows"]],
            "fills_today": len(feed)}
    return {"lines": lines, "data": data}


def publish_daily_report(sessions: sessionmaker[Session], archive, day: str | None = None,
                         post: bool = True, now: pd.Timestamp | None = None) -> Report:
    """Build (or rebuild) the report for an ET date, store it and post it to Discord once."""
    from autotrader.evaluator.jobs import spy_daily_returns
    from autotrader.evaluator.policy import load_policy
    from autotrader.orchestrator.config import load_agents_config

    now = now or pd.Timestamp.now(tz="UTC")
    day = day or today_et(now)
    spy = spy_daily_returns(archive, now)
    with sessions() as s:
        body, data = build_daily_report(s, day, spy, load_agents_config(), load_policy(), now)
        rep = s.scalars(select(Report).where(Report.date == day)).first()
        if rep is None:
            rep = Report(date=day, body_md=body, data=data)
            s.add(rep)
        else:
            rep.body_md, rep.data = body, data
        s.commit()
        if post and rep.posted_at is None:
            url = f"{get_settings().dashboard_url.rstrip('/')}/reports/{day}"
            try:
                if post_report(f"Arena report {day}", body, url):
                    rep.posted_at = utcnow()
                    s.commit()
            except Exception:  # noqa: BLE001 - the report is stored either way
                log.exception("could not post the daily report")
        return rep


def post_commentary(session: Session) -> list[str]:
    """Post supervisor commentary that hasn't been posted yet (Discord, if configured)."""
    out = []
    for rep in session.scalars(select(Report).where(Report.commentary != "",
                                                    Report.commentary_posted_at.is_(None))):
        try:
            post_report(f"Supervisor on {rep.date}", rep.commentary,
                        f"{get_settings().dashboard_url.rstrip('/')}/reports/{rep.date}")
        except Exception:  # noqa: BLE001
            log.exception("could not post commentary for %s", rep.date)
            continue
        rep.commentary_posted_at = utcnow()
        out.append(f"posted supervisor commentary for {rep.date}")
    return out
