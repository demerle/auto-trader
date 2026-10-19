"""The options engine's main loop (its own container): ticks, chasing, the week's rhythm.

Every 10s: sync fills and chase working orders. Every 5 minutes in the session: a tick
(after the 5m bar closes). Chains for the whole options universe are recorded every
refresh. Mondays: the reset. After each close: expired-leg settlement and the session
count. After the week's last session: scoring and the cull.
"""

import logging
import time

import pandas as pd

from autotrader.alerts import alert
from autotrader.config import get_settings
from autotrader.data.calendar import schedule, session_for
from autotrader.engine.state import get_state, set_state
from autotrader.options import competition
from autotrader.options.engine import Gauntlet, week_last_session

log = logging.getLogger(__name__)
LAST = "gauntlet_last"  # {"tick": iso, "settled": date, "recorded": iso}


def _et_time(now: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    hh, mm = map(int, hhmm.split(":"))
    d = now.tz_convert("America/New_York").normalize()
    return (d + pd.Timedelta(hours=hh, minutes=mm)).tz_convert("UTC")


def first_session_of_week(now: pd.Timestamp) -> bool:
    if session_for(now) is None:
        return False
    prev = schedule(now.normalize() - pd.Timedelta(days=10), now.normalize() - pd.Timedelta(days=1))
    return len(prev) == 0 or prev.index[-1].isocalendar()[1] != now.tz_convert("America/New_York").isocalendar()[1]


def due_tick(now: pd.Timestamp, sess, cfg, last_tick: pd.Timestamp | None, delay_sec: int) -> pd.Timestamp | None:
    """The latest 5-minute boundary (from first_tick_et) that has passed by now - delay."""
    first = _et_time(now, cfg.first_tick_et)
    t = now - pd.Timedelta(seconds=delay_sec)
    if t < first or t > sess.market_close + pd.Timedelta(minutes=1):
        return None
    step = pd.Timedelta(minutes=cfg.tick_minutes)
    boundary = first + ((t - first) // step) * step
    boundary = min(boundary, sess.market_close)
    return boundary if last_tick is None or boundary > last_tick else None


def practice(cfg, week: str) -> bool:
    return bool(cfg.practice_until) and week <= str(cfg.practice_until)


def step(g: Gauntlet, now: pd.Timestamp, state: dict) -> dict:
    cfg = g.cfg
    sess = session_for(now)
    delay = get_settings().engine_data_delay_sec
    with g.sessions() as session:
        st = get_state(session, LAST)
        if sess is not None and now >= _et_time(now, cfg.reset_et) and first_session_of_week(now):
            lines = competition.start_week(session, now, cfg)
            if lines:
                log.info("weekly reset:\n%s", "\n".join(lines))
            session.commit()
    if sess is not None and sess.market_open <= now <= sess.market_close + pd.Timedelta(minutes=2):
        with g.sessions() as session:
            if g.desk.working(session):
                g.desk.chase(session, now, g.chains_now(g.spot, now))
                session.commit()
        last_tick = pd.Timestamp(st["tick"]) if st.get("tick") else None
        due = due_tick(now, sess, cfg, last_tick, delay)
        if due is not None:
            report = g.tick(now)
            log.info("%s", report)
            with g.sessions() as session:
                set_state(session, LAST, dict(get_state(session, LAST), tick=due.isoformat()))
                session.commit()
            # Record the rest of the universe too (history for future backtests).
            g.refresh_chains(cfg.universe)
        if time.monotonic() - state.get("reconciled", 0) > 600:
            state["reconciled"] = time.monotonic()
            try:
                drift, stock = g.reconcile()
                if drift:
                    alert("Options ledger/broker drift", "; ".join(f"{s}: ledger {a:+g} vs broker {b:+g}"
                                                                   for s, a, b in drift))
                if stock:
                    alert("Stock in the options account", f"{stock} (assignment?). Close it by hand or "
                                                          "with `autotrader options flatten-stock`.", priority="high")
            except Exception:  # noqa: BLE001
                log.exception("options reconcile failed")
    if sess is not None and now >= sess.market_close + pd.Timedelta(minutes=5):
        day = now.tz_convert("America/New_York").date().isoformat()
        with g.sessions() as session:
            st = get_state(session, LAST)
            if st.get("settled") != day:
                g.settle_expired(now)
                competition.count_session(session, now)
                set_state(session, LAST, dict(get_state(session, LAST), settled=day))
                session.commit()
            if week_last_session(now) and now >= _et_time(now, cfg.score_et):
                week = competition.week_of(now)
                summary = competition.score_week(session, now, cfg, g.archive, practice=practice(cfg, week))
                if "skipped" not in summary:
                    log.info("week scored: %s", summary)
                    competition.fill_slots(session, cfg, "replace culled bots")
                session.commit()
    return state


def run_forever(g: Gauntlet, poll_sec: float = 10.0) -> None:
    state: dict = {}
    while True:
        now = pd.Timestamp.now(tz="UTC")
        try:
            state = step(g, now, state)
        except Exception:  # noqa: BLE001 - the engine must keep running
            log.exception("gauntlet loop error")
        time.sleep(poll_sec)
