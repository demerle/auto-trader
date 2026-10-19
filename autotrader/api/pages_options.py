"""The options gauntlet's pages (behind the same login as the rest of the dashboard)."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from autotrader.api.deps import get_archive, get_gauntlet_cfg, get_sessions
from autotrader.api.pages import LOGIN, page
from autotrader.db.models import OPTIONS, Bot
from autotrader.options import views as ov
from autotrader.options.competition import score_week, week_of

router = APIRouter(prefix="/options")


def _chart(session, week: str) -> list:
    lines = ov.equity_lines(session, week)
    return [(slug, [(pd.Timestamp(t).isoformat(), v) for t, v in pts]) for slug, pts in lines.items()]


def time_to_score(now: pd.Timestamp, cfg, st: dict) -> str | None:
    from autotrader.data.calendar import schedule

    start = pd.Timestamp(st["week"]).tz_localize("UTC")
    days = schedule(start, start + pd.Timedelta(days=6)).index
    if not len(days):
        return None
    hh, mm = map(int, cfg.score_et.split(":"))
    at = (days[-1].tz_localize(None) + pd.Timedelta(hours=hh, minutes=mm)).tz_localize("America/New_York")
    left = at.tz_convert("UTC") - now
    if left.total_seconds() <= 0:
        return None
    d, h = left.days, left.seconds // 3600
    return f"{at:%a %H:%M} ET (in {d}d {h}h)" if d else f"{at:%a %H:%M} ET (in {h}h {left.seconds % 3600 // 60}m)"


def panel(session, cfg) -> dict:
    """For the home page."""
    st = ov.standings(session, cfg=cfg)
    st["score_at"] = time_to_score(pd.Timestamp.now(tz="UTC"), cfg, st)
    return st


@router.get("")
def standings_page(request: Request, _=LOGIN, sessions=Depends(get_sessions), cfg=Depends(get_gauntlet_cfg),
                   week: str | None = None):
    with sessions() as s:
        st = ov.standings(s, cfg=cfg, week=week)
        chart = _chart(s, st["week"])
        feed = ov.trade_feed(s, limit=12)
    st["score_at"] = time_to_score(pd.Timestamp.now(tz="UTC"), cfg, st)
    for r in st["rows"]:
        r["position_rows"] = [{"tag": t, "what": ov.describe(p["legs"]), **p} for t, p in (r["positions"] or {}).items()]
    return page(request, "options.html", st=st, chart=chart, feed=feed, cfg=cfg, refresh=30, section="gauntlet")


@router.get("/trades")
def trades_page(request: Request, _=LOGIN, sessions=Depends(get_sessions), bot: str | None = None):
    with sessions() as s:
        feed = ov.trade_feed(s, limit=200, bot_slug=bot)
    return page(request, "options_trades.html", feed=feed, bot=bot, refresh=30, section="gauntlet")


@router.get("/bots/{slug}")
def bot_page(request: Request, slug: str, _=LOGIN, sessions=Depends(get_sessions), cfg=Depends(get_gauntlet_cfg)):
    from autotrader.db.models import WeeklyResult
    from autotrader.options import registry

    with sessions() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == slug, Bot.group == OPTIONS)).first()
        if bot is None:
            raise HTTPException(404, f"no gauntlet bot {slug}")
        st = ov.standings(s, cfg=cfg)
        row = next((r for r in st["rows"] if r["slug"] == slug), None)
        results = [{"week": r.week, "rank": r.rank, "ret": r.ret, "trades": r.trades, "status": r.status,
                    "max_dd": r.max_drawdown, "busted": r.busted}
                   for r in s.scalars(select(WeeklyResult).where(WeeklyResult.bot_id == bot.id)
                                      .order_by(WeeklyResult.week.desc()))]
        lines = ov.equity_lines(s, st["week"])
        chart = [(slug, [(pd.Timestamp(t).isoformat(), v) for t, v in lines.get(slug, [])])]
        try:
            cls = registry.get(bot.strategy.name, bot.strategy.version)
            source, thesis = registry.source_of(cls), cls.thesis
        except KeyError:
            source, thesis = None, (bot.strategy.backtest_metrics or {}).get("thesis", "")
        info = {"slug": bot.slug, "strategy": f"{bot.strategy.name}@{bot.strategy.version}",
                "author": bot.strategy.author, "stage": bot.stage.value, "description": bot.strategy.description,
                "thesis": thesis, "notes": bot.notes, "paused_until": bot.paused_until,
                "created": bot.created_at, "smoke": bot.strategy.backtest_metrics}
        feed = ov.trade_feed(s, limit=60, bot_slug=slug)
    positions = [{"tag": t, "what": ov.describe(p["legs"]), **p} for t, p in ((row or {}).get("positions") or {}).items()]
    return page(request, "options_bot.html", bot=info, row=row, results=results, chart=chart, source=source,
                feed=feed, positions=positions, refresh=60, section="gauntlet")


@router.get("/weeks")
def weeks_page(request: Request, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        weeks, alltime = ov.weeks(s), ov.all_time(s)
    return page(request, "options_weeks.html", weeks=weeks, alltime=alltime, section="gauntlet")


@router.get("/weeks/{week}")
def week_page(request: Request, week: str, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        rows = ov.week_detail(s, week)
        if not rows:
            raise HTTPException(404, f"no gauntlet week {week}")
        chart = _chart(s, week)
    return page(request, "options_week.html", week=week, rows=rows, chart=chart, section="gauntlet")


@router.get("/graveyard")
def graveyard_page(request: Request, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        rows = ov.graveyard(s)
    return page(request, "options_graveyard.html", rows=rows, section="gauntlet")


@router.get("/cull-preview")
def cull_preview(request: Request, _=LOGIN, sessions=Depends(get_sessions), cfg=Depends(get_gauntlet_cfg),
                 archive=Depends(get_archive)):
    """Read-only: what the scoring would do if the week ended now."""
    from autotrader.options.loop import practice

    now = pd.Timestamp.now(tz="UTC")
    with sessions() as s:
        out = score_week(s, now, cfg, None, apply=False, practice=practice(cfg, week_of(now)))
        s.rollback()
    return page(request, "options_cull.html", out=out, cfg=cfg, section="gauntlet")
