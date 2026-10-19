"""Engine main loop: tick each interval right after its bars close, sync fills continuously."""

import logging
import time

import pandas as pd
from sqlalchemy import select

from autotrader.alerts import alert
from autotrader.broker.reconcile import reconcile
from autotrader.config import get_settings
from autotrader.data.calendar import latest_bar_close, session_for
from autotrader.db.models import Bot, StrategyVersion
from autotrader.engine.arena import ACTIVE_STAGES, Arena
from autotrader.engine.state import get_state, set_state

log = logging.getLogger(__name__)
LAST_TICK = "last_tick"
LAST_EVAL = "last_evaluation"


def evaluation_due(now: pd.Timestamp, at_et: str, last_date: str | None) -> str | None:
    """The session date to evaluate, if today's evaluation time has passed and it hasn't run."""
    if session_for(now) is None:
        return None
    local = now.tz_convert("America/New_York")
    hh, mm = map(int, at_et.split(":"))
    today = local.date().isoformat()
    return today if (local.hour, local.minute) >= (hh, mm) and last_date != today else None


def active_intervals(arena: Arena) -> set[str]:
    with arena.sessions() as session:
        rows = session.scalars(
            select(StrategyVersion.interval).join(Bot, Bot.strategy_id == StrategyVersion.id)
            .where(Bot.stage.in_(ACTIVE_STAGES))).all()
    return set(rows)


def run_forever(arena: Arena, poll_sec: float = 10.0, reconcile_every_sec: float = 600.0) -> None:
    from autotrader.evaluator.jobs import run_daily
    from autotrader.evaluator.policy import load_policy
    from autotrader.reports.daily import publish_daily_report

    settings = get_settings()
    delay = pd.Timedelta(seconds=settings.engine_data_delay_sec)
    last_reconcile = 0.0
    while True:
        now = pd.Timestamp.now(tz="UTC")
        try:
            with arena.sessions() as session:
                arena.desk.sync(session)
                last = get_state(session, LAST_TICK)
                session.commit()
            for interval in sorted(active_intervals(arena)):
                due = latest_bar_close(interval, now - delay, settings.daily_tick_et)
                if due is None or (interval in last and pd.Timestamp(last[interval]) >= due):
                    continue
                report = arena.tick(interval, now)
                log.info("%s", report)
                with arena.sessions() as session:
                    state = get_state(session, LAST_TICK)
                    state[interval] = due.isoformat()
                    set_state(session, LAST_TICK, state)
                    session.commit()
            with arena.sessions() as session:
                last_eval = get_state(session, LAST_EVAL).get("date")
            day = evaluation_due(now, settings.evaluate_et, last_eval)
            if day:
                lines = run_daily(arena.sessions, arena.archive, load_policy(), now)
                log.info("evaluation %s:\n%s", day, "\n".join(lines))
                with arena.sessions() as session:
                    set_state(session, LAST_EVAL, {"date": day})
                    session.commit()
                try:  # the report includes today's verdicts, so it follows the evaluation
                    publish_daily_report(arena.sessions, arena.archive, day, now=now)
                except Exception:  # noqa: BLE001
                    log.exception("daily report failed")
            if time.monotonic() - last_reconcile > reconcile_every_sec:
                with arena.sessions() as session:
                    drift = reconcile(session, arena.broker)
                if drift:
                    alert("Ledger/broker drift", "; ".join(
                        f"{d.symbol}: ledger {d.ledger:+g} vs broker {d.broker:+g}" for d in drift))
                last_reconcile = time.monotonic()
        except Exception:  # noqa: BLE001 - the engine must keep running
            log.exception("engine loop error")
        time.sleep(poll_sec)
