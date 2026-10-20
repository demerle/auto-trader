"""Collector daemon: keeps the bar archive current for every tracked universe.

Runs forever. Intraday intervals are refreshed shortly after each bar closes
during the session; daily bars are refreshed after the close.
"""

import logging
import time

import pandas as pd

from autotrader.config.universes import UNIVERSES
from autotrader.data.archive import BarArchive
from autotrader.data.calendar import is_open, session_for
from autotrader.data.intervals import BAR_DURATION, Interval

log = logging.getLogger(__name__)

DEFAULT_PLAN: dict[Interval, tuple[str, ...]] = {
    "1m": UNIVERSES["core"],
    "5m": UNIVERSES["core"],
    "15m": UNIVERSES["core"] + UNIVERSES["liquid_etfs"],
    "1h": UNIVERSES["core"] + UNIVERSES["liquid_etfs"],
    "1d": tuple(dict.fromkeys(UNIVERSES["core"] + UNIVERSES["liquid_etfs"] + UNIVERSES["mega_caps"])),
}


def _due(interval: Interval, now: pd.Timestamp, last_run: pd.Timestamp | None) -> bool:
    if interval == "1d":
        session = session_for(now)
        if session is None or now < session.market_close + pd.Timedelta(minutes=5):
            return last_run is None  # backfill once at startup
        return last_run is None or last_run < session.market_close
    if not is_open(now - pd.Timedelta(seconds=10)):
        return last_run is None
    step = BAR_DURATION[interval]
    # Wait a few seconds past the boundary so the provider has the closed bar.
    current_bucket = (now - pd.Timedelta(seconds=10)).floor(step)
    return last_run is None or last_run < current_bucket


def run_once(archive: BarArchive, plan: dict[Interval, tuple[str, ...]], interval: Interval) -> None:
    for symbol in plan[interval]:
        try:
            df = archive.update(symbol, interval)
            log.info("collected %s %s: %d bars, last %s", symbol, interval, len(df),
                     df.index[-1] if len(df) else None)
        except Exception:  # noqa: BLE001 - keep the daemon alive
            log.exception("collect failed for %s %s", symbol, interval)


def roster_plan(base: dict[Interval, tuple[str, ...]] = DEFAULT_PLAN) -> dict[Interval, tuple[str, ...]]:
    """The base plan plus every active bot's universe (new agent bots included)."""
    from sqlalchemy import select

    from autotrader.config.universes import resolve
    from autotrader.db.models import ARENA, Bot, BotStage
    from autotrader.db.session import session_factory
    from autotrader.strategies import registry

    plan = {iv: list(syms) for iv, syms in base.items()}
    try:
        with session_factory()() as s:
            for bot in s.scalars(select(Bot).where(Bot.stage.notin_((BotStage.ERADICATED,)), Bot.group == ARENA)):
                sv = bot.strategy
                try:
                    symbols = resolve(registry.get(sv.name, sv.version).universe)
                except Exception:  # noqa: BLE001 - a missing strategy must not stop collection
                    continue
                plan.setdefault(sv.interval, [])
                plan[sv.interval] += [x for x in symbols if x not in plan[sv.interval]]
    except Exception:  # noqa: BLE001 - no database: collect the base plan
        log.warning("could not read the roster; collecting the base plan", exc_info=True)
    try:  # the options gauntlet's underlyings, and the volatility indexes its bots read
        from autotrader.options.config import load_gauntlet_config
        from autotrader.options.history import VIX, VIX1D

        extra = {"1m": load_gauntlet_config().universe,
                 "5m": load_gauntlet_config().universe + (VIX, VIX1D),
                 "1d": load_gauntlet_config().universe + (VIX, VIX1D)}
        for iv, syms in extra.items():
            plan.setdefault(iv, [])
            plan[iv] += [x for x in syms if x not in plan[iv]]
    except Exception:  # noqa: BLE001
        log.warning("could not add the options universe", exc_info=True)
    return {iv: tuple(syms) for iv, syms in plan.items()}


def run_forever(plan: dict[Interval, tuple[str, ...]] | None = None, poll_sec: float = 5.0,
                replan_sec: float = 600.0) -> None:
    dynamic = plan is None
    plan = roster_plan() if dynamic else plan
    planned_at = time.monotonic()
    archive = BarArchive()
    last_run: dict[Interval, pd.Timestamp | None] = {i: None for i in BAR_DURATION}
    while True:
        if dynamic and time.monotonic() - planned_at > replan_sec:
            plan, planned_at = roster_plan(), time.monotonic()
        now = pd.Timestamp.now(tz="UTC")
        for interval in plan:
            if _due(interval, now, last_run[interval]):
                run_once(archive, plan, interval)
                last_run[interval] = now
        time.sleep(poll_sec)
