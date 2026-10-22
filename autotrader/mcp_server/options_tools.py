"""MCP tools for the options gauntlet's agents (options_researcher, options_auditor).

Same rules as the arena's tools: every call is logged, agent code only runs in the
sandbox, and nothing here can place orders or touch the risk limits.
"""

import hashlib

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from autotrader.db.models import OPTIONS, Bot, BotStage, StrategyVersion, Submission
from autotrader.options import registry as oreg
from autotrader.options.config import load_gauntlet_config
from autotrader.strategies import lint

OPTIONS_DIR = Path(__file__).parent.parent / "options"


class OptionsToolsMixin:
    # ----- helpers -----
    def _gcfg(self):
        return load_gauntlet_config()

    def _olimits(self):
        from autotrader.config import get_settings
        from autotrader.risk.limits import load_limits

        return load_limits(get_settings().risk_file, verify=False).options

    def _ometa(self, path: Path) -> dict:
        from autotrader.mcp_server.tools import NAME_RE, VERSION_RE, ToolFailure

        source = path.read_text()
        if problems := lint.lint_source(source):
            raise ToolFailure("lint failed:\n- " + "\n- ".join(problems))
        try:
            meta = oreg.static_meta(source)
        except (ValueError, SyntaxError) as exc:
            raise ToolFailure(str(exc)) from None
        universe = set(self._gcfg().universe)
        bad = [u for u in meta["underlyings"] if u not in universe]
        if bad or not meta["underlyings"]:
            raise ToolFailure(f"underlyings must be a non-empty tuple from {sorted(universe)}; not {bad}")
        if not NAME_RE.match(meta["name"]) or not VERSION_RE.match(meta["version"]):
            raise ToolFailure(f"name must match {NAME_RE.pattern} and version be digits")
        if not meta["thesis"].strip():
            raise ToolFailure("set `thesis`: a few sentences on why this makes money")
        return meta

    def _ohistory(self, underlyings, start, end) -> dict:
        from autotrader.options.history import load_history

        return load_history(tuple(underlyings), start, end, self.archive, refresh=False)

    # ----- read-only -----
    def get_options_api(self) -> str:
        """The option strategy contract, helpers, an example, the risk rules and how you are judged.
        Read this first."""
        cfg, lim = self._gcfg(), self._olimits()
        contract = (OPTIONS_DIR / "strategy.py").read_text()
        helpers = (OPTIONS_DIR / "helpers.py").read_text()
        example = (OPTIONS_DIR / "builtin" / "vix1d_gamma_switch.py").read_text()
        return f"""# Writing an options gauntlet bot

Write ONE file in your workspace ({self.workspace}), e.g. `strategy.py`, with exactly one
`class X(OptionStrategy)`. `name`, `version`, `underlyings`, `description` and `thesis` must be
plain literals (read without running your code). Underlyings must come from: {list(cfg.universe)}.
Allowed imports: {sorted(lint.ALLOWED_IMPORTS)}. No I/O. Your code runs in a sandbox; on_tick must
answer within 5 seconds. It is called every {cfg.tick_minutes} minutes from {cfg.first_tick_et} ET.

## The competition
- {cfg.slots} bots, ${cfg.capital:,.0f} each, reset to ${cfg.capital:,.0f} every Monday. After the week's last session the
  week's return is ranked; the bottom {cfg.cull} are culled and replaced by new research (you).
- Only the week's return counts. Think big asymmetric payoffs with capped losses; a bot that
  quietly makes 0.5% a week will be culled by one that makes 40%.
- A bot that joins mid-week with fewer than {cfg.min_sessions_to_cull} sessions is exempt that week.

## Hard rules (enforced by the engine whatever your code says)
- Defined risk only: no uncovered short calls; every structure's worst case is computed and a new
  structure may risk at most {lim.max_structure_risk_pct:.0%} of equity; the bot can never be able
  to lose more than its equity. At most {lim.max_open_structures} open structures, {lim.max_legs} legs,
  {lim.max_contracts_per_order} contracts per leg per order.
- Same-day expiries: no new positions {lim.open_cutoff_min} min before Alpaca's cutoff
  ({cfg.cutoff_et} ET); they are force-closed {lim.flatten_cutoff_min} min before it.
- Everything is closed at {lim.friday_flat_et} ET on the week's last session (before the scoring).
- Below ${lim.bust_equity:,.0f} the bot is frozen for the rest of the week (busted, lands last).
- Two bots may not hold opposite sides of the same contract; such an open is refused (pick
  a neighbouring strike).

## Data and fills
Chains are CBOE's free delayed quotes (~15 min old) repriced at the live underlying price
(`mark`, `est_bid`, `est_ask`; IV and greeks recomputed). Orders start at the mid and are
chased toward the far side for ~2 minutes, then cancelled (you can ask again next tick).
Backtests use SYNTHETIC chains (VIX/VIX1D-based IV with skew, pessimistic spreads, fills at the
far side), so treat backtest profits with suspicion; use them to check behaviour.
CBOE volume / open interest are real live, but NaN in synthetic backtests.

## Entry test (submit_options_strategy)
Lint, then a replay of the last {cfg.smoke.sessions} sessions: it must open at least
{cfg.smoke.min_trades} structures, raise no errors, and never be able to lose more than its equity.
No profit bar. Then a second model audits the code (bounded risk, no lookahead, sane sizing,
not a clone of a survivor). Passing puts the bot straight into the gauntlet.

## Contract (autotrader/options/strategy.py)
```python
{contract}
```

## Helpers (autotrader/options/helpers.py)
```python
{helpers}
```

## Example (a seed bot)
```python
{example}
```
"""

    def get_gauntlet(self) -> str:
        """This week's live standings (return, trades, open positions, status) and the cut line."""
        from autotrader.options import views

        with self.sessions() as s:
            st = views.standings(s, self.now)
        for r in st["rows"]:
            r.pop("last_plan", None)
            r["positions"] = {t: {"what": views.describe(p["legs"]), "qty": p["qty"], "pnl": p["pnl"]}
                              for t, p in (r["positions"] or {}).items()}
        return json.dumps(st, indent=1, default=str)

    def get_survivors(self) -> str:
        """Every bot still in the gauntlet: strategy, thesis, record so far, and its full source."""
        out = []
        with self.sessions() as s:
            bots = s.scalars(select(Bot).where(Bot.group == OPTIONS, Bot.stage.in_(
                (BotStage.PAPER_PROBATION, BotStage.PAPER_PROVEN))).order_by(Bot.id)).all()
            from autotrader.options import views

            history = {r["slug"]: r for r in views.all_time(s)}
            for b in bots:
                try:
                    cls = oreg.get(b.strategy.name, b.strategy.version)
                    source = oreg.source_of(cls)
                except KeyError:
                    source = "(source unavailable)"
                out.append(f"## {b.slug} ({b.strategy.name}@{b.strategy.version}, by {b.strategy.author})\n"
                           f"Record: {json.dumps(history.get(b.slug, {}), default=str)}\n\n```python\n{source}\n```")
        return "\n\n".join(out) or "No bots in the gauntlet yet."

    def get_options_graveyard(self, limit: int = 10) -> str:
        """Culled gauntlet bots with full autopsies (trade logs, what the winners did) and their code."""
        from autotrader.options import views

        with self.sessions() as s:
            rows = views.graveyard(s, limit=min(int(limit), 30))
            for r in rows:
                bot = s.scalars(select(Bot).where(Bot.slug == r["slug"])).first()
                try:
                    r["source"] = oreg.source_of(oreg.get(bot.strategy.name, bot.strategy.version))[:6000]
                except KeyError:
                    r["source"] = None
        return json.dumps(rows, indent=1, default=str) if rows else "The gauntlet graveyard is empty."

    def get_weekly_results(self) -> str:
        """Every scored week: standings, who was culled, and the all-time leaderboard."""
        from autotrader.options import views

        with self.sessions() as s:
            weeks = [dict(w, standings=views.week_detail(s, w["week"])) for w in views.weeks(s)[:12]]
            for w in weeks:
                for r in w["standings"]:
                    r.pop("autopsy", None)
            return json.dumps({"weeks": weeks, "all_time": views.all_time(s)}, indent=1, default=str)

    def get_chain(self, underlying: str, expiry_index: int = 0, strikes: int = 8) -> str:
        """The latest recorded (real, ~15 min delayed) chain for one underlying around the money."""
        from autotrader.mcp_server.tools import ToolFailure
        from autotrader.options.archive import ChainArchive
        from autotrader.options.helpers import Chain
        from autotrader.options.pricing import nowcast

        snap = ChainArchive().latest(underlying.upper())
        if snap is None:
            raise ToolFailure(f"no recorded chain for {underlying}")
        df = nowcast(snap.chain, snap.spot, snap.spot, snap.quote_time, snap.quote_time)
        c = Chain(snap.underlying, snap.spot, snap.quote_time, df, snap.quote_time.tz_convert("America/New_York").date())
        e = c.expiry(int(expiry_index))
        if e is None:
            raise ToolFailure("no such expiry")
        side = c.df[c.df["expiry"] == e]
        near = side.iloc[(side["strike"] - c.spot).abs().argsort()[: 2 * max(1, min(int(strikes), 30))]]
        near = near.sort_values(["right", "strike"])
        cols = ["symbol", "right", "strike", "bid", "ask", "iv", "delta", "gamma", "theta", "oi", "volume"]
        return (f"{c.underlying} spot {c.spot} quotes as of {snap.quote_time} | expiries "
                f"{[str(x) for x in c.expiries()[:10]]}\n" + near[cols].round(4).to_csv(index=False))

    def fetch_underlying(self, symbol: str, interval: str = "5m", last_n: int = 100) -> str:
        """Bars for an underlying or ^VIX / ^VIX1D (5m: ~60 days, 1d: years): summary and a CSV tail."""
        from autotrader.mcp_server.tools import ToolFailure

        if interval not in ("1m", "5m", "15m", "1h", "1d"):
            raise ToolFailure("interval must be 1m, 5m, 15m, 1h or 1d")
        try:
            df = self.archive.get(symbol.upper(), interval, refresh=False)
        except Exception as exc:  # noqa: BLE001
            raise ToolFailure(f"no data for {symbol}: {exc}") from None
        if df.empty:
            raise ToolFailure(f"no {interval} data for {symbol}")
        tail = df.tail(max(1, min(int(last_n), 500))).round(4)
        return f"{symbol.upper()} {interval}: {len(df)} bars, {df.index[0]} .. {df.index[-1]}\n" + tail.to_csv()

    # ----- research (sandboxed) -----
    def run_options_backtest(self, path: str, sessions: int = 20, params: dict | None = None) -> str:
        """Replay your strategy file over the last N sessions (max 40) on synthetic chains, with the
        engine's real rules and weekly $5k resets. Returns weekly returns, trades and rejections."""
        from autotrader.options.smoke import simulate_file, window

        p = self._path(path)
        meta = self._ometa(p)
        self._spend_trials(1)
        self._trials = (1, [])
        start, end = window(max(5, min(int(sessions), 40)), self.now)
        out = simulate_file(p, meta["underlyings"], self._gcfg(), self._olimits(), start, end,
                            self._ohistory(meta["underlyings"], start, end), params)
        return json.dumps({"window": [str(start.date()), str(end.date())], **out["summary"],
                           "last_trades": out["trades"][-15:]}, indent=1, default=str)

    def submit_options_strategy(self, path: str, rationale: str) -> str:
        """Submit your finished bot: lint, smoke test (it must trade, risk must be bounded), then audit.
        `rationale`: why it should win a week, and what you learned from the culled bots."""
        from autotrader.mcp_server.tools import ToolFailure
        from autotrader.options.smoke import judge, simulate_file, window

        p = self._path(path)
        meta = self._ometa(p)
        with self.sessions() as s:
            count = s.scalar(select(func.count()).select_from(Submission).where(Submission.job_id == self.job_id))
            if count >= self.cfg.per_job.submissions:
                raise ToolFailure(f"submission limit ({self.cfg.per_job.submissions}) reached")
            taken = s.scalars(select(StrategyVersion).where(
                StrategyVersion.name == meta["name"], StrategyVersion.version == meta["version"])).first()
            pending = s.scalars(select(Submission).where(
                Submission.name == meta["name"], Submission.version == meta["version"],
                Submission.status.in_(("audit_pending", "approved", "deployed")))).first()
            if taken or pending:
                raise ToolFailure(f"{meta['name']}@{meta['version']} already exists; use a new name")
        cfg = self._gcfg()
        start, end = window(cfg.smoke.sessions, self.now)
        try:
            out = simulate_file(p, meta["underlyings"], cfg, self._olimits(), start, end,
                                self._ohistory(meta["underlyings"], start, end))
            res = judge(out, cfg)
        except Exception as exc:  # noqa: BLE001
            res = None
            reasons, summary = [f"simulation failed: {str(exc)[:500]}"], {}
        if res is not None:
            reasons, summary = res.reasons, res.summary
        passed = res is not None and res.passed
        source = p.read_text()
        sub = Submission(job_id=self.job_id, name=meta["name"], version=meta["version"], interval="opt",
                         draft_path=str(p), code_sha=hashlib.sha256(source.encode()).hexdigest(),
                         source=source, rationale=rationale[:4000], params={},
                         status="audit_pending" if passed else "gate_failed", reasons=reasons,
                         gate_metrics=json.loads(json.dumps(summary, default=str)), holdout_metrics={},
                         n_trials=1)
        with self.sessions() as s:
            s.add(sub)
            s.commit()
        return json.dumps({"submission_id": sub.id,
                           "verdict": "PASSED the entry test; queued for audit" if passed else "FAILED the entry test",
                           "reasons": reasons, "smoke": summary}, indent=1, default=str)

    def _options_submission_view(self, sub) -> dict:
        return {"submission_id": sub.id, "strategy": f"{sub.name}@{sub.version}", "kind": "options gauntlet bot",
                "rationale": sub.rationale, "smoke_test": sub.gate_metrics}



