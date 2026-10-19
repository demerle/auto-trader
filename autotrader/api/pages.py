"""Read-only dashboard pages (all behind the login)."""

import asyncio
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from autotrader import views
from autotrader.api import render
from autotrader.api.auth import COOKIE, SESSION_DAYS, check_token, new_session, require_login
from autotrader.api.deps import get_agents_cfg, get_archive, get_policy, get_sessions
from autotrader.config import get_settings
from autotrader.db.models import AgentJob, AuditLog, Report, Submission, ToolCall
from autotrader.evaluator.jobs import spy_daily_returns

HERE = Path(__file__).parent
LOGIN = Depends(require_login)

templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters.update(money=render.money, pct=render.pct, fnum=render.fnum, when=render.when,
                             sign=render.sign_class, markdown=render.markdown)
templates.env.globals.update(line_chart=render.line_chart)

router = APIRouter()
_failed_logins: list[float] = []


def page(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"csrf": getattr(request.state, "csrf", ""),
                                                      "nav": name.split(".")[0], **ctx})


def _spy(archive):
    return spy_daily_returns(archive, pd.Timestamp.now(tz="UTC"))


# ----- login -----

def _safe_next(target: str | None) -> str:
    u = urlparse(target or "/")
    return target if target and not u.scheme and not u.netloc and target.startswith("/") \
        and not target.startswith("//") else "/"


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/", error: str = ""):
    return page(request, "login.html", next=_safe_next(next), error=error)


@router.post("/login")
async def login(request: Request, token: str = Form(""), next: str = Form("/")):
    now = time.monotonic()
    _failed_logins[:] = [t for t in _failed_logins if now - t < 300]
    if len(_failed_logins) >= 10:
        raise HTTPException(429, "too many failed logins; wait five minutes")
    if not check_token(token):
        _failed_logins.append(now)
        await asyncio.sleep(1.0)
        return RedirectResponse(f"/login?error=1&next={quote(_safe_next(next))}", status_code=303)
    resp = RedirectResponse(_safe_next(next), status_code=303)
    resp.set_cookie(COOKIE, new_session(), max_age=SESSION_DAYS * 86400, httponly=True,
                    samesite="strict", secure=get_settings().dashboard_secure_cookie, path="/")
    return resp


@router.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE, path="/")
    return resp


# ----- pages -----

@router.get("/", response_class=HTMLResponse)
def overview(request: Request, _=LOGIN, sessions=Depends(get_sessions), archive=Depends(get_archive),
             cfg=Depends(get_agents_cfg), policy=Depends(get_policy)):
    with sessions() as s:
        acct = views.account_overview(s)
        agents = views.agents_overview(s, cfg, policy, limit=8)
        alerts = views.recent_alerts(s, limit=12)
        rep = views.latest_report(s)
        table = views.league(s, _spy(archive))
        report = {"date": rep.date, "body_md": rep.body_md, "commentary": rep.commentary} if rep else None
    return page(request, "overview.html", acct=acct, agents=agents, alerts=alerts, report=report,
                league=table[:5], refresh=60)


@router.get("/league", response_class=HTMLResponse)
def league_page(request: Request, _=LOGIN, sessions=Depends(get_sessions), archive=Depends(get_archive),
                sort: str = "sharpe"):
    with sessions() as s:
        rows = views.league(s, _spy(archive))
        roster = views.roster(s)
    if sort in ("total_return", "excess_return", "psr", "max_drawdown", "days", "trades", "capital"):
        rows.sort(key=lambda r: r[sort] if r[sort] is not None else float("-inf"), reverse=True)
    return page(request, "league.html", rows=rows, roster=roster, sort=sort)


@router.get("/bots/{slug}", response_class=HTMLResponse)
def bot_page(request: Request, slug: str, _=LOGIN, sessions=Depends(get_sessions),
             archive=Depends(get_archive)):
    with sessions() as s:
        bot = views.bot_detail(s, slug, _spy(archive))
    if bot is None:
        raise HTTPException(404, f"no bot {slug}")
    return page(request, "bot.html", bot=bot)


@router.get("/graveyard", response_class=HTMLResponse)
def graveyard_page(request: Request, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        rows = views.graveyard(s)
    return page(request, "graveyard.html", rows=rows)


@router.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request, _=LOGIN, sessions=Depends(get_sessions), cfg=Depends(get_agents_cfg),
                policy=Depends(get_policy)):
    with sessions() as s:
        o = views.agents_overview(s, cfg, policy)
    return page(request, "agents.html", o=o, intervals=cfg.research_intervals,
                refresh=60 if any(j["status"] == "running" for j in o["jobs"]) else None)


@router.get("/agents/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: int, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        job = s.get(AgentJob, job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        row = views.job_row(job)
        calls = [{"ts": c.ts.isoformat(), "tool": c.tool, "ok": c.ok, "ms": c.duration_ms,
                  "trials": c.trials, "args": c.args}
                 for c in s.scalars(select(ToolCall).where(ToolCall.job_id == job_id).order_by(ToolCall.id))]
        subs = [views.submission_row(x) for x in s.scalars(
            select(Submission).where(Submission.job_id == job_id))]
        prompt = job.prompt
    entries = []
    path = row["transcript_path"]
    if path:
        p = Path(path)
        log_dir = get_settings().agent_log_dir.resolve()
        if p.resolve().is_relative_to(log_dir) and p.is_file():
            entries = render.transcript(p.read_text(errors="replace").splitlines())
    return page(request, "job.html", job=row, calls=calls, subs=subs, prompt=prompt, entries=entries,
                refresh=30 if row["status"] == "running" else None)


@router.get("/agents/submissions/{sub_id}", response_class=HTMLResponse)
def submission_page(request: Request, sub_id: int, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        x = s.get(Submission, sub_id)
        if x is None:
            raise HTTPException(404, f"no submission {sub_id}")
        sub, source = views.submission_row(x), x.source
    return page(request, "submission.html", sub=sub, source=source)


@router.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        rows = [{"date": r.date, "posted": r.posted_at is not None, "commentary": bool(r.commentary)}
                for r in s.scalars(select(Report).order_by(Report.date.desc()).limit(120))]
    return page(request, "reports.html", rows=rows)


@router.get("/reports/{day}", response_class=HTMLResponse)
def report_page(request: Request, day: str, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        r = s.scalars(select(Report).where(Report.date == day)).first()
        if r is None:
            raise HTTPException(404, f"no report for {day}")
        rep = {"date": r.date, "body_md": r.body_md, "commentary": r.commentary,
               "posted_at": r.posted_at.isoformat() if r.posted_at else None}
    return page(request, "report.html", report=rep)


@router.get("/audit", response_class=HTMLResponse)
def audit_page(request: Request, _=LOGIN, sessions=Depends(get_sessions)):
    with sessions() as s:
        rows = [{"ts": a.ts.isoformat(), "action": a.action, "target": a.target, "detail": a.detail}
                for a in s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(200))]
        alerts = views.recent_alerts(s, limit=100)
    return page(request, "audit.html", rows=rows, alerts=alerts)
