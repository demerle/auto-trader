"""The gauntlet's live engine: every 5 minutes each options bot sees fresh chains and trades.

Per tick: sync fills -> refresh CBOE chains (recorded) and Yahoo bars -> nowcast ->
each bot: book from fills, context, strategy (sandboxed if agent-written), planner
(cutoffs, Friday flat, risk), desk orders -> equity snapshots -> account drawdown check.
"""

import logging
import time
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from autotrader.alerts import alert
from autotrader.data.calendar import schedule, session_for
from autotrader.db.models import Bot, EquitySnapshot, Fill, utcnow
from autotrader.engine.state import ACCOUNT_HWM, KILL_SWITCH, account_key, get_state, is_killed, set_state
from autotrader.options import cboe, registry
from autotrader.options.archive import ChainArchive
from autotrader.options.config import GauntletConfig
from autotrader.options.context import build_context, intrinsic, marks_for
from autotrader.options.desk import OptionsDesk
from autotrader.options.helpers import Chain
from autotrader.options.history import VIX, VIX1D
from autotrader.options.ledger import account_positions, load_option_book
from autotrader.options.occ import parse
from autotrader.options.planner import Change, plan_tick
from autotrader.options.pricing import nowcast
from autotrader.options.roster import active_option_bots
from autotrader.options.strategy import OptionStrategy
from autotrader.risk.limits import RiskLimits
from autotrader.sandbox import SandboxError

log = logging.getLogger(__name__)


@dataclass
class GauntletReport:
    now: pd.Timestamp
    lines: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join([f"[gauntlet @ {self.now.tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET]"]
                         + [f"  {x}" for x in self.lines])


def week_last_session(now: pd.Timestamp) -> bool:
    today = session_for(now)
    if today is None:
        return False
    nxt = schedule(now.normalize() + pd.Timedelta(days=1), now.normalize() + pd.Timedelta(days=10))
    return len(nxt) == 0 or nxt.index[0].isocalendar()[1] != now.tz_convert("America/New_York").isocalendar()[1]


class Gauntlet:
    def __init__(self, sessions: sessionmaker[Session], broker, archive, cfg: GauntletConfig,
                 limits: RiskLimits, chain_archive: ChainArchive | None = None,
                 chain_fetcher=None, strategy_loader=None):
        self.sessions = sessions
        self.broker = broker
        self.archive = archive
        self.cfg = cfg
        self.limits = limits
        self.olimits = limits.options
        self.desk = OptionsDesk(broker, cfg)
        self.chain_archive = chain_archive or ChainArchive()
        self.fetch_chain = chain_fetcher or (lambda u: cboe.fetch_chain(u, cfg.chain.max_dte, cfg.chain.moneyness))
        self.load_strategy = strategy_loader or self._load_strategy
        self._strategies: dict[tuple, OptionStrategy] = {}
        self._snaps: dict[str, cboe.ChainSnapshot] = {}
        self._fetched: dict[str, float] = {}
        self._last_marks: dict[str, float] = {}
        self.spot: dict[str, float] = {}  # live underlying prices from the last tick

    @property
    def account(self) -> str:
        return self.broker.account

    # ----- strategies -----
    def _load_strategy(self, bot: Bot) -> OptionStrategy:
        cls = registry.get(bot.strategy.name, bot.strategy.version)
        if registry.is_sandboxed(cls) and cls.code_sha != bot.strategy.code_sha:
            raise SandboxError(f"{cls.name}@{cls.version} source changed since it was registered")
        key = (bot.id, getattr(cls, "code_sha", cls.__module__))
        if key not in self._strategies:
            self._strategies[key] = cls(bot.params)
        return self._strategies[key]

    def _drop_strategy(self, bot: Bot) -> None:
        for key in [k for k in self._strategies if k[0] == bot.id]:
            getattr(self._strategies.pop(key), "close", lambda: None)()

    # ----- data -----
    def refresh_chains(self, underlyings, force: bool = False) -> list[str]:
        """Fetch (and record) CBOE chains not refreshed within refresh_sec. Returns failures."""
        failed = []
        for u in underlyings:
            if not force and time.monotonic() - self._fetched.get(u, -1e9) < self.cfg.chain.refresh_sec - 15:
                continue
            try:
                snap = self.fetch_chain(u)
                self._snaps[u] = snap
                self._fetched[u] = time.monotonic()
                if self.cfg.chain.record:
                    self.chain_archive.save(snap)
            except Exception as exc:  # noqa: BLE001 - one bad chain must not stop the others
                log.warning("chain %s: %s", u, exc)
                failed.append(u)
        return failed

    def _bars(self, symbol: str, interval: str, days: int, refresh: bool = True) -> pd.DataFrame:
        try:
            return self.archive.get(symbol, interval, start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days),
                                    refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("bars %s %s: %s", symbol, interval, exc)
            return pd.DataFrame()

    def market(self, underlyings, now: pd.Timestamp, feeds=()) -> dict:
        """Bars, spot and VIX levels for the underlyings; `feeds` get 5m bars only (strategies' bar_symbols)."""
        bars, daily, spot = {}, {}, {}
        for sym in feeds:
            b5 = self._bars(sym, "5m", 7)
            bars[sym] = b5[b5.index + pd.Timedelta(minutes=5) <= now] if len(b5) else b5
        for u in underlyings:
            b5 = self._bars(u, "5m", 7)
            b5 = b5[b5.index + pd.Timedelta(minutes=5) <= now] if len(b5) else b5
            bars[u] = b5
            daily[u] = self._bars(u, "1d", 400, refresh=False)
            daily[u] = daily[u][daily[u].index < now.normalize()] if len(daily[u]) else daily[u]
            b1 = self._bars(u, "1m", 2)
            b1 = b1[b1.index + pd.Timedelta(minutes=1) <= now] if len(b1) else b1
            if len(b1) and now - b1.index[-1] < pd.Timedelta(minutes=10):
                spot[u] = float(b1["close"].iloc[-1])
            elif len(b5):
                spot[u] = float(b5["close"].iloc[-1])
        levels = {}
        for sym in (VIX, VIX1D):
            b = self._bars(sym, "5m", 3)
            levels[sym] = float(b["close"].iloc[-1]) if len(b) else None
        return {"bars": bars, "daily": daily, "spot": spot, "vix": levels[VIX], "vix1d": levels[VIX1D]}

    def chains_now(self, spot: dict[str, float], now: pd.Timestamp) -> dict[str, Chain]:
        out = {}
        today = now.tz_convert("America/New_York").date()
        for u, snap in self._snaps.items():
            if u not in spot:
                continue
            df = nowcast(snap.chain, snap.spot, spot[u], snap.quote_time, now)
            out[u] = Chain(u, spot[u], snap.quote_time, df, today)
        return out

    # ----- the tick -----
    def tick(self, now: pd.Timestamp | None = None, dry_run: bool = False) -> GauntletReport:
        now = now or pd.Timestamp.now(tz="UTC")
        report = GauntletReport(now)
        sess = session_for(now)
        if sess is None or not (sess.market_open <= now <= sess.market_close):
            report.lines.append("market closed")
            return report
        with self.sessions() as session:
            self.desk.sync(session)
            session.commit()
            if is_killed(session, self.account):
                report.lines.append("options kill switch is ON; not trading")
                return report
            bots = active_option_bots(session)
            if not bots:
                report.lines.append("no active options bots")
                return report
            strategies: dict[int, OptionStrategy] = {}
            for bot in bots:
                try:
                    strategies[bot.id] = self.load_strategy(bot)
                except Exception as exc:  # noqa: BLE001
                    self._pause(session, bot, now, f"strategy failed to load: {exc}", report)
            underlyings = sorted({u for s in strategies.values() for u in s.underlyings} & set(self.cfg.universe))
            failed = self.refresh_chains(underlyings)
            if failed:
                report.lines.append(f"chain fetch failed: {failed}")
            feeds = sorted({b for s in strategies.values() for b in s.bar_symbols} - set(underlyings))
            mk = self.market(underlyings, now, feeds)
            self.spot.update(mk["spot"])
            chains = self.chains_now(mk["spot"], now)
            last = week_last_session(now)
            from autotrader.options.competition import current_week_start

            for bot in bots:
                if bot.id not in strategies:
                    continue
                self._tick_bot(session, bot, strategies[bot.id], now, sess, last, mk, chains,
                               current_week_start(session, bot), report, dry_run)
            if not dry_run:
                self._check_drawdown(session, report)
                session.commit()
            else:
                session.rollback()
        return report

    def _tick_bot(self, session, bot, strategy, now, sess, last, mk, chains, week_start_eq, report, dry_run):
        book = load_option_book(session, bot)
        held = book.positions()
        marks = marks_for(held, chains, mk["spot"], self._last_marks)
        self._last_marks.update(marks)
        equity = book.equity(marks)
        frozen = None
        if equity < self.olimits.bust_equity:
            frozen = "busted"
        elif bot.paused_until is not None and pd.Timestamp(bot.paused_until).tz_convert("UTC") > now:
            frozen = "paused"
        busy = self.desk.busy_tags(session, bot.id)
        targets = None
        if frozen != "busted":
            ctx = build_context(now, {u: c for u, c in chains.items() if u in strategy.underlyings},
                                {u: s for u, s in mk["spot"].items() if u in strategy.underlyings},
                                {u: b for u, b in mk["bars"].items()
                                 if u in strategy.underlyings or u in strategy.bar_symbols},
                                {u: d for u, d in mk["daily"].items() if u in strategy.underlyings},
                                mk["vix"], mk["vix1d"], book, marks, week_start_eq,
                                sess.market_open, sess.market_close, last)
            try:
                targets = strategy.on_tick(ctx)
            except Exception as exc:  # noqa: BLE001 - a broken bot is paused, not the engine
                self._pause(session, bot, now, f"on_tick failed: {str(exc)[:300]}", report)
                frozen = "paused"
                targets = None
        if targets is not None:  # tags with orders still working are left alone this tick
            targets = {t: s for t, s in targets.items() if t not in busy} | \
                      {t: _hold(book, t) for t in busy if t in book.open}
        plan = plan_tick(now, targets, book, chains, equity, self.olimits, self.cfg, last, frozen)
        changes = [c for c in plan.changes if c.tag not in busy]
        placed, problems = [], list(plan.rejected)
        for ch in changes:
            if dry_run:
                placed.append(f"{ch.kind} {ch.tag} {ch.legs} ({ch.reason})")
                continue
            p = self.desk.place(session, bot, ch, chains)
            if p.status == "placed":
                placed.append(f"{ch.kind} {ch.tag} x{abs(ch.units)} ({ch.reason})")
            else:
                problems.append((ch.tag, f"{p.status}: {p.detail}"))
        bot.target_qty = {"at": now.isoformat(), "targets": sorted(targets) if targets else [],
                          "placed": placed, "rejected": [f"{t}: {w}" for t, w in problems][:20],
                          "frozen": frozen}
        if frozen == "busted":
            from autotrader.options.competition import mark_busted

            mark_busted(session, bot, now)
        if not dry_run:
            self._snapshot(session, bot, book, marks, now)
        report.lines.append(f"{bot.slug}: equity ${equity:,.0f}"
                            + (f" [{frozen}]" if frozen else "")
                            + (f"; {'; '.join(placed)}" if placed else "")
                            + (f"; rejected {problems}" if problems else ""))

    def _snapshot(self, session, bot, book, marks, now) -> None:
        snap = book.snapshot(marks)
        positions = {t: {"legs": [[leg.symbol, leg.ratio] for leg in s.legs], "qty": s.qty,
                         "entry": round(s.entry, 4), "mark": round(s.mark, 4), "pnl": round(s.pnl, 2),
                         "note": s.note} for t, s in snap.items()}
        ts = now.floor("min").to_pydatetime()
        existing = session.scalars(select(EquitySnapshot).where(EquitySnapshot.bot_id == bot.id,
                                                                EquitySnapshot.ts == ts)).first()
        if existing is not None:  # a re-run of the same minute replaces it
            existing.equity, existing.cash, existing.positions = book.equity(marks), book.cash, positions
            return
        session.add(EquitySnapshot(bot_id=bot.id, ts=ts, equity=book.equity(marks), cash=book.cash,
                                   positions=positions))

    def _pause(self, session, bot, now, why: str, report) -> None:
        self._drop_strategy(bot)
        sess = session_for(now)
        until = (sess.market_close if sess is not None else now) + pd.Timedelta(hours=16)
        bot.paused_until = until.to_pydatetime()
        report.lines.append(f"{bot.slug}: PAUSED ({why})")
        alert(f"{bot.slug} paused", f"{why}; no new risk until the next session. Open positions are still "
                                    "closed at the cutoffs.")

    # ----- account safety -----
    def _check_drawdown(self, session, report) -> None:
        try:
            equity = self.broker.equity()
        except Exception:  # noqa: BLE001
            return
        key = account_key(ACCOUNT_HWM, self.account)
        hwm = max(get_state(session, key).get("equity", equity), equity)
        set_state(session, key, {"equity": hwm})
        if equity < hwm * (1 - self.olimits.account_max_drawdown_pct):
            self._kill(session, f"options account drawdown {equity / hwm - 1:.1%}", report)

    def kill(self, reason: str) -> GauntletReport:
        report = GauntletReport(pd.Timestamp.now(tz="UTC"))
        with self.sessions() as session:
            self._kill(session, reason, report)
            session.commit()
        return report

    def _kill(self, session, reason: str, report) -> None:
        now = pd.Timestamp.now(tz="UTC")
        set_state(session, account_key(KILL_SWITCH, self.account),
                  {"on": True, "reason": reason, "at": now.isoformat()})
        try:
            self.broker.cancel_all()
        except Exception:  # noqa: BLE001
            log.exception("cancel_all failed")
        self.desk.sync(session)
        for o in self.desk.working(session):
            o.status, o.done_at = "canceled", utcnow()
        for bot in session.scalars(select(Bot).where(Bot.account == self.account)).all():
            book = load_option_book(session, bot)
            for tag in list(book.open):
                legs = [(s, -int(round(q))) for s, q in book.open[tag].legs.items()]
                p = self.desk.place(session, bot, Change(tag, "close", legs, -book.units(tag), "kill switch"),
                                    {}, market=True)
                report.lines.append(f"{bot.slug} {tag}: {p.status} {p.detail}")
        report.lines.append(f"OPTIONS KILL SWITCH ON: {reason}")
        alert("OPTIONS KILL SWITCH ON", f"{reason}. Options orders cancelled, positions closing at market. "
                                        "The long-term arena is not affected.", priority="urgent")

    def unkill(self) -> None:
        with self.sessions() as session:
            set_state(session, account_key(KILL_SWITCH, self.account), {"on": False})
            session.commit()

    def reconcile(self) -> tuple[list[tuple[str, float, float]], dict[str, float]]:
        """(option drift: symbol, ledger, broker), stray stock positions (assignment)."""
        with self.sessions() as session:
            ours = account_positions(session, self.account)
        theirs = self.broker.option_positions()
        drift = [(s, ours.get(s, 0.0), theirs.get(s, 0.0)) for s in sorted(set(ours) | set(theirs))
                 if abs(ours.get(s, 0.0) - theirs.get(s, 0.0)) > 1e-6]
        return drift, self.broker.stock_positions()

    def settle_expired(self, now: pd.Timestamp) -> list[str]:
        """After the close: legs that expired in our books settle at intrinsic (shouldn't happen)."""
        out = []
        today = now.tz_convert("America/New_York").date()
        sess = session_for(now)
        with self.sessions() as session:
            for bot in session.scalars(select(Bot).where(Bot.account == self.account)).all():
                book = load_option_book(session, bot)
                for tag, pos in list(book.open.items()):
                    for sym, q in list(pos.legs.items()):
                        c = parse(sym)
                        if c.expiry < today or (c.expiry == today and sess is not None and now >= sess.market_close):
                            spot = self._spot_at_close(c.underlying)
                            price = intrinsic(sym, spot) if spot else 0.0
                            session.add(Fill(bot_id=bot.id, symbol=sym, qty=-q, price=price, fees=0.0,
                                             internal=True, structure=tag))
                            out.append(f"{bot.slug} {tag} {sym} x{q:+g} settled at {price:.2f}")
            session.commit()
        if out:
            alert("Options expired in the books", "These were still open at expiry and were settled at "
                                                  "intrinsic value; check the broker:\n" + "\n".join(out))
        return out

    def _spot_at_close(self, u: str) -> float | None:
        b = self._bars(u, "5m", 3)
        return float(b["close"].iloc[-1]) if len(b) else None


def _hold(book, tag: str):
    from autotrader.options.strategy import Structure

    return Structure(book.legs_of(tag), book.units(tag))
