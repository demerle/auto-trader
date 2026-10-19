"""The live/paper engine: one tick runs every active bot of an interval.

Per tick: sync fills -> cancel stale orders -> fetch closed bars (Yahoo, via the
archive) -> quality gate -> each bot's strategy (same context, sizing and Book as
the backtester) -> risk clamps -> desk netting -> broker.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from autotrader.alerts import alert
from autotrader.backtest.costs import CostModel
from autotrader.broker.base import Broker
from autotrader.broker.desk import Desk, DeskPlan, Intent, net_intents
from autotrader.broker.sizing import clean_targets, order_deltas, target_quantities
from autotrader.data.calendar import drop_incomplete, is_open, session_for
from autotrader.data.intervals import PERIODS_PER_YEAR, Interval
from autotrader.data.quality import check_bars
from autotrader.db.models import Bot, BotStage, EquitySnapshot, StrategyVersion
from autotrader.engine.state import ACCOUNT_HWM, KILL_SWITCH, get_state, is_killed, set_state
from autotrader.ledger.book import Book
from autotrader.ledger.store import load_book, working_qty
from autotrader.risk.limits import RiskLimits, clamp_targets
from autotrader.strategies import registry
from autotrader.strategies.base import Strategy, make_context

log = logging.getLogger(__name__)

ACTIVE_STAGES = (BotStage.PAPER_PROBATION, BotStage.PAPER_PROVEN, BotStage.LIVE)
LIQUIDATE_STAGES = (BotStage.ERADICATED,)
FEATURE_WINDOW_MULT = 3  # features are computed on 3x lookback so EWMs have converged


@dataclass
class TickReport:
    interval: str
    now: pd.Timestamp
    bots: int = 0
    plan: DeskPlan | None = None
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"[{self.interval} @ {self.now:%Y-%m-%d %H:%M}Z] {self.bots} bots"]
        if self.plan:
            lines += [f"  order {o.symbol:6} {o.qty:+8.0f}  alloc {o.allocations}" for o in self.plan.external]
            lines += [f"  cross {s:6} {q:+8.0f}  bot {b}" for b, s, q in self.plan.internal]
            lines += [f"  defer {s:6} {q:+8.0f}  bot {b}" for b, s, q in self.plan.deferred]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _utc(dt) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _history_start(interval: Interval, n_bars: int, now: pd.Timestamp) -> pd.Timestamp:
    if interval == "1d":
        return now - pd.Timedelta(days=int(n_bars * 1.5) + 10)
    per_day = PERIODS_PER_YEAR[interval] / 252
    return now - pd.Timedelta(days=int(n_bars / per_day * 1.5) + 5)


class Arena:
    def __init__(self, sessions: sessionmaker[Session], broker: Broker, archive, limits: RiskLimits,
                 costs: CostModel = CostModel(), collar_bps: float = 15.0,
                 strategy_loader=None):
        self.load_strategy = strategy_loader or (
            lambda bot: registry.get(bot.strategy.name, bot.strategy.version)(bot.params))
        self.sessions = sessions
        self.broker = broker
        self.archive = archive
        self.limits = limits
        self.desk = Desk(broker, costs)
        self.collar_bps = collar_bps

    # ----- data -----
    def _load_bars(self, symbols: set[str], interval: Interval, n_bars: int, now: pd.Timestamp,
                   report: TickReport) -> tuple[dict[str, pd.DataFrame], set[str]]:
        start = _history_start(interval, n_bars, now)
        bars, good = {}, set()
        for s in sorted(symbols):
            try:
                df = self.archive.get(s, interval, start=start, end=now)
            except Exception as exc:  # noqa: BLE001 - a symbol failing must not stop the tick
                report.notes.append(f"{s}: data error {exc}")
                continue
            df = drop_incomplete(df, interval, now)
            if df.empty:
                continue
            bars[s] = df
            issues = check_bars(s, df, interval, now)
            if issues:
                report.notes.append(f"{s}: " + ", ".join(i.kind for i in issues))
            else:
                good.add(s)
        return bars, good

    # ----- one bot -----
    def _bot_intents(self, session: Session, bot: Bot, strategy: Strategy, book: Book,
                     bars: Mapping[str, pd.DataFrame], good: set[str], prices: Mapping[str, float],
                     bar_now: pd.Timestamp, now: pd.Timestamp, report: TickReport) -> list[Intent]:
        equity = book.equity(prices)
        session.add(EquitySnapshot(bot_id=bot.id, ts=now.to_pydatetime(), equity=equity,
                                   cash=book.cash, positions=dict(book.qty)))

        allow_new = self._allow_new_risk(session, bot, equity, now, report)

        window = strategy.lookback * FEATURE_WINDOW_MULT
        universe = (set(strategy.symbols) | set(book.qty)) & good
        sym_bars = {s: bars[s].iloc[-window:] for s in universe}
        feats = {s: f for s, df in sym_bars.items() if (f := strategy.features(df)) is not None}
        ctx = make_context(strategy, bar_now, sym_bars, feats, book.weights(prices), equity)
        try:
            targets = strategy.on_bar(ctx)
        except Exception as exc:  # noqa: BLE001 - a broken strategy must not stop the tick
            log.exception("strategy %s failed", strategy.key)
            report.notes.append(f"{bot.slug}: strategy error {exc!r}")
            targets = None

        if targets is not None:
            weights = clean_targets(targets, set(book.qty), self.limits.max_gross, set(sym_bars))
            tq = target_quantities(weights, equity, prices)
            assets = {s: self.broker.asset(s) for s, q in tq.items() if q != book.qty.get(s, 0.0)}
            tq, notes = clamp_targets(tq, book.qty, prices, equity, self.limits, assets)
            report.notes += [f"{bot.slug}: {n}" for n in notes]
            bot.target_qty = {s: q for s, q in tq.items() if q or s in book.qty}

        target = dict(bot.target_qty or {})
        if not allow_new:  # only reductions
            target = {s: (t if abs(t) <= abs(book.qty.get(s, 0.0)) and t * book.qty.get(s, 0.0) >= 0
                          else book.qty.get(s, 0.0)) for s, t in target.items()}
        working = working_qty(session, bot)
        effective = {s: book.qty.get(s, 0.0) + working.get(s, 0.0) for s in set(book.qty) | set(working)}
        tradable = good & set(prices)
        deltas = order_deltas({s: t for s, t in target.items() if s in tradable},
                              {s: q for s, q in effective.items() if s in tradable}, prices)
        return [Intent(bot.id, s, d, {"target": target.get(s, 0.0), "held": book.qty.get(s, 0.0),
                                      "bar": str(bar_now), "strategy": strategy.key})
                for s, d in sorted(deltas.items())]

    def _allow_new_risk(self, session: Session, bot: Bot, equity: float, now: pd.Timestamp,
                        report: TickReport) -> bool:
        if bot.paused_until is not None and _utc(bot.paused_until) > now:
            return False
        sess = session_for(now)
        if sess is None:
            return True
        day_start = sess.market_open - pd.Timedelta(hours=12)
        first = session.scalars(
            select(EquitySnapshot.equity).where(EquitySnapshot.bot_id == bot.id,
                                                EquitySnapshot.ts >= day_start.to_pydatetime())
            .order_by(EquitySnapshot.ts).limit(1)).first()
        if first and equity < first * (1 - self.limits.bot_daily_loss_pct):
            bot.paused_until = (sess.market_close + pd.Timedelta(hours=6)).to_pydatetime()
            msg = f"daily loss limit hit ({equity / first - 1:.1%}); no new risk until next session"
            report.notes.append(f"{bot.slug}: {msg}")
            alert(f"{bot.slug} paused", msg)
            return False
        return True

    # ----- tick -----
    def tick(self, interval: Interval, now: pd.Timestamp | None = None, dry_run: bool = False
             ) -> TickReport:
        now = now or pd.Timestamp.now(tz="UTC")
        report = TickReport(interval, now)
        session = self.sessions()
        try:
            self.desk.sync(session)
            if is_killed(session):
                report.notes.append("kill switch is ON; not trading")
                session.commit()
                return report
            if interval != "1d" and not dry_run:
                self.desk.cancel_working(session, interval)

            all_bots = session.scalars(
                select(Bot).join(StrategyVersion)
                .where(StrategyVersion.interval == interval, Bot.account == self.broker.account)
                .order_by(Bot.id)).all()
            books = {b.id: load_book(session, b) for b in all_bots}
            bots = [b for b in all_bots if b.stage in ACTIVE_STAGES]
            # Eradicated bots still holding positions are closed out at their normal cadence.
            closing = [b for b in all_bots if b.stage in LIQUIDATE_STAGES and books[b.id].qty]
            report.bots = len(bots)
            if not bots and not closing:
                session.commit()
                return report

            strategies = {b.id: self.load_strategy(b) for b in bots}
            symbols = set().union(*(set(s.symbols) for s in strategies.values()),
                                  *(set(books[b.id].qty) for b in bots + closing))
            n_bars = max((s.lookback for s in strategies.values()), default=10) * FEATURE_WINDOW_MULT
            bars, good = self._load_bars(symbols, interval, n_bars, now, report)
            prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items()}
            if not good:
                report.notes.append("no usable bars")
                session.commit()
                return report
            bar_now = max(bars[s].index[-1] for s in good)

            intents: list[Intent] = []
            for bot in bots:
                book = books[bot.id]
                missing = [s for s in book.qty if s not in prices]
                if missing:
                    report.notes.append(f"{bot.slug}: no price for held {missing}; skipped")
                    continue
                intents += self._bot_intents(session, bot, strategies[bot.id], book, bars, good,
                                             prices, bar_now, now, report)
            for bot in closing:
                working = working_qty(session, bot)
                for s, q in books[bot.id].qty.items():
                    remaining = -q - working.get(s, 0.0)
                    if s in prices and remaining:
                        intents.append(Intent(bot.id, s, remaining, {"liquidate": bot.stage.value}))
                report.notes.append(f"{bot.slug}: closing out ({bot.stage.value})")

            plan = net_intents(intents, self.broker.positions())
            report.plan = plan
            if dry_run:
                session.rollback()
                return report
            order_type, tif = ("market", "opg") if interval == "1d" else ("limit", "day")
            self.desk.execute(session, plan, interval, prices, order_type, tif, self.collar_bps,
                              {(i.bot_id, i.symbol): i.reason for i in intents})
            self._check_account_drawdown(session, report)
            session.commit()
            return report
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _check_account_drawdown(self, session: Session, report: TickReport) -> None:
        equity = self.broker.equity()
        hwm = max(get_state(session, ACCOUNT_HWM).get("equity", equity), equity)
        set_state(session, ACCOUNT_HWM, {"equity": hwm})
        if equity < hwm * (1 - self.limits.account_max_drawdown_pct):
            report.notes.append(f"account drawdown {equity / hwm - 1:.1%} breached limit")
            self._kill(session, f"account drawdown {equity / hwm - 1:.1%}", report)

    # ----- kill switch -----
    def kill(self, reason: str) -> TickReport:
        report = TickReport("all", pd.Timestamp.now(tz="UTC"))
        with self.sessions() as session:
            self._kill(session, reason, report)
            session.commit()
        return report

    def _kill(self, session: Session, reason: str, report: TickReport) -> None:
        now = pd.Timestamp.now(tz="UTC")
        set_state(session, KILL_SWITCH, {"on": True, "reason": reason, "at": now.isoformat()})
        self.desk.cancel_working(session)
        intents, prices = [], {}
        for bot in session.scalars(select(Bot)).all():
            bot.target_qty = {}
            for s, q in load_book(session, bot).qty.items():
                intents.append(Intent(bot.id, s, -q, {"kill": reason}))
        positions = self.broker.positions()
        for it in intents:
            prices[it.symbol] = prices.get(it.symbol) or self._last_price(it.symbol)
        plan = net_intents(intents, positions)
        report.plan = plan
        tif = "day" if is_open(now) else "opg"
        self.desk.execute(session, plan, "kill", prices, "market", tif)
        report.notes.append(f"KILL SWITCH ON: {reason}")
        alert("KILL SWITCH ON", f"{reason}. All orders cancelled, flatten orders sent.", priority="urgent")

    def _last_price(self, symbol: str) -> float:
        for interval in ("1m", "5m", "1d"):
            df = self.archive.load(symbol, interval)
            if len(df):
                return float(df["close"].iloc[-1])
        return 0.0

    def unkill(self) -> None:
        with self.sessions() as session:
            set_state(session, KILL_SWITCH, {"on": False})
            session.commit()
