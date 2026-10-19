"""Dashboard actions: POST only, login + CSRF, each one written to the audit log and alerted.

Promotion to live money is phase 7 and has no action here.
"""

import logging

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from autotrader.alerts import alert
from autotrader.api.auth import require_csrf
from autotrader.api.deps import arena_factory, gauntlet_factory, get_sessions
from autotrader.db.models import AuditLog, Bot
from autotrader.orchestrator import control

log = logging.getLogger(__name__)
router = APIRouter(prefix="/actions")
CSRF = Depends(require_csrf)
FAR_FUTURE = pd.Timedelta(days=3650)


def _audit(sessions, request: Request, action: str, target: str = "", **detail) -> None:
    client = request.client.host if request.client else "?"
    with sessions() as s:
        s.add(AuditLog(action=action, target=target, detail={"client": client, **detail}))
        s.commit()


def _bot_page(sessions, slug: str) -> str:
    from autotrader.db.models import OPTIONS

    with sessions() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == slug)).first()
        return f"/options/bots/{slug}" if bot is not None and bot.group == OPTIONS else f"/bots/{slug}"


@router.post("/kill")
async def kill(request: Request, reason: str = Form(""), confirm: str = Form(""), _=CSRF,
               sessions=Depends(get_sessions), arena_factory=Depends(arena_factory)):
    if confirm.strip().upper() != "KILL":
        raise HTTPException(400, "type KILL to confirm")
    why = f"dashboard: {reason.strip() or 'no reason given'}"
    _audit(sessions, request, "kill", "arena", reason=why)
    report = await run_in_threadpool(lambda: arena_factory().kill(why))  # alerts by itself
    log.warning("kill switch from dashboard: %s", report)
    return RedirectResponse("/", status_code=303)


@router.post("/unkill")
async def unkill(request: Request, _=CSRF, sessions=Depends(get_sessions),
                 arena_factory=Depends(arena_factory)):
    _audit(sessions, request, "unkill", "arena")
    await run_in_threadpool(lambda: arena_factory().unkill())
    alert("Kill switch off", "Turned off from the dashboard; the engine trades again.", priority="high")
    return RedirectResponse("/", status_code=303)


@router.post("/bots/{slug}/pause")
def pause_bot(request: Request, slug: str, _=CSRF, sessions=Depends(get_sessions)):
    with sessions() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == slug)).first()
        if bot is None:
            raise HTTPException(404, f"no bot {slug}")
        bot.paused_until = (pd.Timestamp.now(tz="UTC") + FAR_FUTURE).to_pydatetime()
        s.commit()
    _audit(sessions, request, "pause_bot", slug)
    alert(f"Bot {slug} paused", "Paused from the dashboard: it keeps its positions but takes no new risk.")
    return RedirectResponse(_bot_page(sessions, slug), status_code=303)


@router.post("/bots/{slug}/unpause")
def unpause_bot(request: Request, slug: str, _=CSRF, sessions=Depends(get_sessions)):
    with sessions() as s:
        bot = s.scalars(select(Bot).where(Bot.slug == slug)).first()
        if bot is None:
            raise HTTPException(404, f"no bot {slug}")
        bot.paused_until = None
        s.commit()
    _audit(sessions, request, "unpause_bot", slug)
    alert(f"Bot {slug} unpaused", "Unpaused from the dashboard.")
    return RedirectResponse(_bot_page(sessions, slug), status_code=303)


@router.post("/agents/pause")
def pause_agents(request: Request, _=CSRF, sessions=Depends(get_sessions)):
    with sessions() as s:
        control.pause(s, "paused from the dashboard")
        s.commit()
    _audit(sessions, request, "pause_agents")
    alert("Agents paused", "Paused from the dashboard; no new agent jobs start.")
    return RedirectResponse("/agents", status_code=303)


@router.post("/agents/resume")
def resume_agents(request: Request, _=CSRF, sessions=Depends(get_sessions)):
    with sessions() as s:
        resumed = control.resume(s)
        s.commit()
    _audit(sessions, request, "resume_agents", resumed=resumed)
    if resumed:
        alert("Agents resumed", "Resumed from the dashboard.")
    return RedirectResponse("/agents", status_code=303)


@router.post("/agents/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: int, _=CSRF, sessions=Depends(get_sessions)):
    with sessions() as s:
        try:
            control.cancel_job(s, job_id)
        except control.ControlError as exc:
            raise HTTPException(400, str(exc)) from None
        s.commit()
    _audit(sessions, request, "cancel_job", str(job_id))
    return RedirectResponse("/agents", status_code=303)


@router.post("/agents/queue")
def queue_research(request: Request, interval: str = Form(...), note: str = Form(""), _=CSRF,
                   sessions=Depends(get_sessions)):
    with sessions() as s:
        try:
            job = control.queue_research(s, interval, note.strip()[:2000])
        except control.ControlError as exc:
            raise HTTPException(400, str(exc)) from None
        s.commit()
        job_id = job.id
    _audit(sessions, request, "queue_research", str(job_id), interval=interval, note=note[:500])
    return RedirectResponse("/agents", status_code=303)


# ----- the options gauntlet (its own account and kill switch) -----

@router.post("/options/kill")
async def options_kill(request: Request, reason: str = Form(""), confirm: str = Form(""), _=CSRF,
                       sessions=Depends(get_sessions), gauntlet_factory=Depends(gauntlet_factory)):
    if confirm.strip().upper() != "KILL":
        raise HTTPException(400, "type KILL to confirm")
    why = f"dashboard: {reason.strip() or 'no reason given'}"
    _audit(sessions, request, "options_kill", "gauntlet", reason=why)
    report = await run_in_threadpool(lambda: gauntlet_factory().kill(why))  # alerts by itself
    log.warning("options kill switch from dashboard: %s", report)
    return RedirectResponse("/options", status_code=303)


@router.post("/options/unkill")
async def options_unkill(request: Request, _=CSRF, sessions=Depends(get_sessions),
                         gauntlet_factory=Depends(gauntlet_factory)):
    _audit(sessions, request, "options_unkill", "gauntlet")
    await run_in_threadpool(lambda: gauntlet_factory().unkill())
    alert("Options kill switch off", "Turned off from the dashboard; the gauntlet trades again.", priority="high")
    return RedirectResponse("/options", status_code=303)
