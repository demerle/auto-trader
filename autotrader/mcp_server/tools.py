"""The tools agents get, bound to one job and one role.

Everything an agent can do to the arena goes through here: it never gets a
shell, the database or the broker. Every call is logged as a ToolCall; backtest
calls also record their Sharpe ratios so the gate can deflate for the whole search.
Agent code only ever runs inside the sandbox.
"""

import functools
import inspect
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from autotrader.config.universes import UNIVERSES, resolve
from autotrader.data.intervals import PERIODS_PER_YEAR
from autotrader.db.models import (
    AgentJob, AgentNote, Bot, BotStage, Evaluation, StrategyVersion, Submission, ToolCall,
)
from autotrader.evaluator.gate import split_holdout, universe_benchmark
from autotrader.evaluator.policy import Policy
from autotrader.orchestrator.config import AgentsConfig
from autotrader.sandbox import SandboxError, run_once, static_meta
from autotrader.strategies import lint

CONFIG_DIR = Path(__file__).parent.parent / "config"
STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
VERSION_RE = re.compile(r"^\d{1,4}$")

ROLE_TOOLS = {
    "researcher": ("get_strategy_api", "list_bots", "get_bot_stats", "get_graveyard", "get_catalog",
                   "get_universes", "fetch_bars", "run_backtest", "run_walkforward",
                   "submit_strategy", "write_note"),
    "reviewer": ("get_strategy_api", "list_bots", "get_bot_stats", "get_graveyard", "get_catalog",
                 "get_universes", "fetch_bars", "run_backtest", "run_walkforward",
                 "propose_version", "write_note"),
    "auditor": ("get_strategy_api", "list_bots", "get_bot_stats", "get_submission",
                "audit_verdict", "write_note"),
}


class ToolFailure(Exception):
    """A user-facing error: the agent did something not allowed or not valid."""


def _r(x, nd: int = 4):
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, nd)
    return x


def _summary(m: dict, keys=("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "psr", "dsr",
                            "closing_trades", "win_rate", "profit_factor", "n_periods")) -> dict:
    return {k: _r(m[k]) for k in keys if k in m}


class AgentTools:
    def __init__(self, job_id: int, role: str, workspace: Path, sessions: sessionmaker[Session],
                 archive, cfg: AgentsConfig, policy: Policy, now: pd.Timestamp | None = None):
        if role not in ROLE_TOOLS:
            raise ValueError(f"unknown role {role}")
        self.job_id, self.role = job_id, role
        self.workspace = Path(workspace).resolve()
        self.sessions = sessions
        self.archive = archive
        self.cfg = cfg
        self.policy = policy
        self._now = now

    @property
    def now(self) -> pd.Timestamp:
        return self._now or pd.Timestamp.now(tz="UTC")

    # ----- plumbing -----
    def functions(self) -> list[Callable]:
        """The logged tool callables this role may use (names and docstrings are the API)."""
        return [self._logged(getattr(self, name)) for name in ROLE_TOOLS[self.role]]

    def call(self, name: str, **kwargs) -> str:
        if name not in ROLE_TOOLS[self.role]:
            raise ToolFailure(f"tool {name} is not available to the {self.role}")
        return self._logged(getattr(self, name))(**kwargs)

    def _logged(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            self._trials: tuple[int, list[float]] = (0, [])
            ok = True
            try:
                return fn(*args, **kwargs)
            except (ToolFailure, SandboxError) as exc:
                ok = False
                raise ToolFailure(str(exc)) from None
            except Exception as exc:
                ok = False
                raise ToolFailure(f"internal error: {type(exc).__name__}: {exc}") from None
            finally:
                trials, sharpes = self._trials
                with self.sessions() as s:
                    s.add(ToolCall(job_id=self.job_id, tool=fn.__name__,
                                   args=json.loads(json.dumps(kwargs, default=str)),
                                   ok=ok, trials=trials, sharpes=sharpes,
                                   duration_ms=int((time.monotonic() - t0) * 1000)))
                    s.commit()

        wrapper.__signature__ = inspect.signature(fn)
        return wrapper

    def _job(self, s: Session) -> AgentJob:
        return s.get(AgentJob, self.job_id)

    def _path(self, path: str) -> Path:
        p = (self.workspace / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not p.is_relative_to(self.workspace):
            raise ToolFailure(f"{path} is outside your workspace {self.workspace}")
        if p.suffix != ".py" or not p.is_file():
            raise ToolFailure(f"{path} is not an existing .py file in your workspace")
        return p

    def _meta(self, path: Path) -> dict:
        source = path.read_text()
        if problems := lint.lint_source(source):
            raise ToolFailure("lint failed:\n- " + "\n- ".join(problems))
        try:
            meta = static_meta(source)
        except (ValueError, SyntaxError) as exc:
            raise ToolFailure(str(exc)) from None
        if meta["interval"] not in PERIODS_PER_YEAR:
            raise ToolFailure(f"interval must be one of {sorted(PERIODS_PER_YEAR)}")
        symbols = resolve(meta["universe"])
        cap = self.cfg.max_universe.get(meta["interval"], 20)
        if not symbols or len(symbols) > cap:
            raise ToolFailure(f"universe must have 1..{cap} symbols for {meta['interval']}")
        meta["symbols"] = symbols
        return meta

    def _bars(self, symbols, interval: str, sealed: bool) -> dict[str, pd.DataFrame]:
        start = self.cfg.history_start(interval, self.now)
        out = {}
        for sym in symbols:
            try:
                df = self.archive.get(sym, interval, start=start)
            except Exception:  # noqa: BLE001 - unknown/delisted symbols are just skipped
                continue
            if len(df):
                out[sym] = df
        if not out:
            raise ToolFailure(f"no {interval} data for {list(symbols)[:5]}...")
        return split_holdout(out, self.cfg.holdout_start(interval, self.now)) if sealed else out

    def _trials_so_far(self, s: Session) -> tuple[int, list[float]]:
        rows = s.execute(select(ToolCall.trials, ToolCall.sharpes)
                         .where(ToolCall.job_id == self.job_id)).all()
        return sum(r[0] for r in rows), [x for r in rows for x in (r[1] or [])]

    def _spend_trials(self, n: int) -> int:
        with self.sessions() as s:
            used, _ = self._trials_so_far(s)
        if used + n > self.cfg.per_job.backtests:
            raise ToolFailure(f"trial budget exhausted: {used} of {self.cfg.per_job.backtests} "
                              f"backtests used, this call needs {n}. Submit your best version.")
        return used

    def _run(self, path: Path, op: str, **kwargs) -> dict:
        return run_once(path, op, memory_mb=self.cfg.sandbox_memory_mb, **kwargs)

    # ----- read-only tools -----
    def get_strategy_api(self) -> str:
        """The strategy contract, indicators, import rules, an example, and how you are judged.
        Read this first."""
        from autotrader.indicators import core

        base = (STRATEGIES_DIR / "base.py").read_text()
        example = (STRATEGIES_DIR / "builtin" / "rsi2_reversion.py").read_text()
        indicators = "\n".join(
            f"- {name}{inspect.signature(fn)}: {(fn.__doc__ or '').strip().splitlines()[0] if fn.__doc__ else ''}"
            for name, fn in inspect.getmembers(core, inspect.isfunction) if fn.__module__ == core.__name__)
        g = self.policy.gate
        return f"""# Writing a strategy

Write ONE file in your workspace ({self.workspace}), e.g. `strategy.py`, with exactly one
`class X(Strategy)`. Its `name`, `version`, `interval`, `universe` and `description` must be
plain literals (they are read without running your code). `name` matches {NAME_RE.pattern};
`version` is a string of digits. `universe` is a universe name (see get_universes) or a tuple
of tickers (max {self.cfg.max_universe}). Strategies are pure: bars in, target weights out.

Allowed imports: {sorted(lint.ALLOWED_IMPORTS)}. No I/O, no eval/exec, no dunder attributes,
no globals. Your code always runs in a sandbox with memory and time limits; on_bar must answer
within 5s, features within 60s.

## Contract (autotrader/strategies/base.py)
```python
{base}
```

## Indicators (from autotrader.indicators import ...)
{indicators}

## Example (a builtin strategy)
```python
{example}
```

## Execution model
Targets are sized into whole shares at the close of bar t and filled at the OPEN of bar t+1,
with 5 bps slippage and SEC/TAF fees (shorts pay 3%/yr borrow). Daily bots trade market-on-open;
intraday bots trade marketable limits that may not fill on gaps. No intrabar stops exist:
exits happen on bar closes. Gross exposure is capped at 1.0 of the bot's equity.

## How you are judged
- Walk-forward: tuned on rolling train windows, scored only on the following test windows.
  Pass needs OOS Sharpe >= {g.min_oos_sharpe}, deflated Sharpe >= {g.min_dsr}, max drawdown within
  {g.max_drawdown:.0%}, at least {g.min_trades} OOS closing trades, and a Sharpe at least
  {g.min_excess_sharpe} above equal-weight buy-and-hold of YOUR OWN universe (so picking
  today's winners as a universe earns nothing).
- Deflated Sharpe counts EVERY backtest you run in this job (budget {self.cfg.per_job.backtests}).
  Fishing through many variants lowers your score.
- A sealed holdout (the most recent {self.cfg.holdout_days} days per interval) is hidden from
  all your tools. The final gate also requires a positive Sharpe there. Overfitting fails here.
- Then a different model audits your code for lookahead, leakage and overfitting.
- Passing earns a paper-trading slot with ${self.policy.base_capital:,.0f}; the evaluator then
  eradicates it if paper results disappoint.
"""

    def list_bots(self) -> str:
        """Every bot in the arena with its strategy, interval, stage, capital and latest verdict."""
        with self.sessions() as s:
            rows = []
            for bot in s.scalars(select(Bot).order_by(Bot.id)).all():
                ev = s.scalars(select(Evaluation).where(Evaluation.bot_id == bot.id)
                               .order_by(Evaluation.id.desc()).limit(1)).first()
                rows.append({"slug": bot.slug, "strategy": f"{bot.strategy.name}@{bot.strategy.version}",
                             "interval": bot.strategy.interval, "stage": bot.stage.value,
                             "capital": round(bot.capital), "since": str(bot.stage_since)[:10],
                             "latest": f"{ev.decision}: {ev.rationale}" if ev else None})
        return json.dumps(rows, indent=1)

    def get_bot_stats(self, slug: str) -> str:
        """Paper record, recent evaluations, backtest expectation and source code of one bot."""
        from autotrader.evaluator.record import build_record

        with self.sessions() as s:
            bot = s.scalars(select(Bot).where(Bot.slug == slug)).first()
            if bot is None:
                raise ToolFailure(f"no bot {slug}")
            rec = build_record(s, bot)
            evals = s.scalars(select(Evaluation).where(Evaluation.bot_id == bot.id)
                              .order_by(Evaluation.id.desc()).limit(5)).all()
            sv = bot.strategy
            out = {
                "slug": bot.slug, "strategy": f"{sv.name}@{sv.version}", "interval": sv.interval,
                "params": bot.params, "stage": bot.stage.value, "capital": bot.capital,
                "paper_days": rec.days, "paper_trades": rec.trades,
                "paper_metrics": _summary(rec.metrics) | {k: _r(rec.metrics.get(k)) for k in
                                                          ("alpha", "excess_return", "backtest_z")},
                "backtest_expectation": _summary(sv.backtest_metrics or {}),
                "evaluations": [{"ts": str(e.ts)[:10], "decision": e.decision,
                                 "rationale": e.rationale} for e in evals],
                "notes": bot.notes[-2000:],
            }
        return json.dumps(out, indent=1) + "\n\n## Source\n```python\n" + self._source(sv) + "\n```"

    def _source(self, sv: StrategyVersion) -> str:
        from autotrader.strategies import registry

        try:
            cls = registry.get(sv.name, sv.version)
        except KeyError:
            return "(source not found)"
        path = getattr(cls, "source_path", None) or inspect.getfile(cls)
        return Path(path).read_text()

    def get_graveyard(self, limit: int = 20) -> str:
        """Eradicated bots and why they died. Do not resubmit these ideas unchanged."""
        with self.sessions() as s:
            bots = s.scalars(select(Bot).where(Bot.stage == BotStage.ERADICATED)
                             .order_by(Bot.stage_since.desc()).limit(min(limit, 50))).all()
            out = [{"slug": b.slug, "strategy": f"{b.strategy.name}@{b.strategy.version}",
                    "interval": b.strategy.interval, "params": b.params,
                    "died": str(b.stage_since)[:10], "autopsy": b.notes[-1500:]} for b in bots]
        return json.dumps(out, indent=1) if out else "The graveyard is empty."

    def get_catalog(self) -> str:
        """Known strategy families, every strategy version ever registered, every past agent
        submission with its outcome, and recent agent notes. Use it to avoid repeating work."""
        catalog = yaml.safe_load((CONFIG_DIR / "catalog.yaml").read_text())
        with self.sessions() as s:
            versions = [{"strategy": f"{v.name}@{v.version}", "interval": v.interval,
                         "author": v.author, "description": v.description,
                         "gate_sharpe": _r((v.backtest_metrics or {}).get("sharpe"))}
                        for v in s.scalars(select(StrategyVersion).order_by(StrategyVersion.id))]
            subs = [{"strategy": f"{x.name}@{x.version}", "interval": x.interval, "status": x.status,
                     "reasons": x.reasons[:4], "oos_sharpe": _r(x.gate_metrics.get("sharpe")),
                     "rationale": x.rationale[:300]}
                    for x in s.scalars(select(Submission).order_by(Submission.id.desc()).limit(60))]
            notes = [{"topic": n.topic, "body": n.body[:600], "ts": str(n.created_at)[:10]}
                     for n in s.scalars(select(AgentNote).order_by(AgentNote.id.desc()).limit(30))]
        return json.dumps({"families": catalog, "registered_versions": versions,
                           "past_submissions": subs, "agent_notes": notes}, indent=1)

    def get_universes(self) -> str:
        """Named universes you can use as a strategy's `universe` (or give a tuple of tickers)."""
        return json.dumps({k: list(v) for k, v in UNIVERSES.items()}, indent=1)

    def fetch_bars(self, symbol: str, interval: str = "1d", last_n: int = 100) -> str:
        """OHLCV for one symbol (history before the sealed holdout only): summary and a CSV tail."""
        if interval not in PERIODS_PER_YEAR:
            raise ToolFailure(f"interval must be one of {sorted(PERIODS_PER_YEAR)}")
        df = self._bars([symbol.upper()], interval, sealed=True)[symbol.upper()]
        tail = df.tail(max(1, min(int(last_n), 500))).round(4)
        rets = df["close"].pct_change().dropna()
        info = {"symbol": symbol.upper(), "interval": interval, "bars": len(df),
                "first": str(df.index[0]), "last_visible": str(df.index[-1]),
                "sealed_from": str(self.cfg.holdout_start(interval, self.now)),
                "ann_vol": _r(float(rets.std() * np.sqrt(PERIODS_PER_YEAR[interval]))),
                "total_return": _r(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1))}
        return json.dumps(info, indent=1) + "\n\n" + tail.to_csv()

    # ----- research tools (sandboxed) -----
    def run_backtest(self, path: str, params: dict | None = None) -> str:
        """Backtest your strategy file over the visible history (1 trial)."""
        p = self._path(path)
        meta = self._meta(p)
        used = self._spend_trials(1)
        bars = self._bars(meta["symbols"], meta["interval"], sealed=True)
        res = self._run(p, "backtest", params=params or {}, bars=bars)
        m = res["metrics"]
        self._trials = (1, [m["sharpe"] / math.sqrt(PERIODS_PER_YEAR[meta["interval"]])])
        bench = universe_benchmark(bars, pd.DatetimeIndex(sorted(set().union(
            *(df.index for df in bars.values())))))
        out = {"strategy": f"{meta['name']}@{meta['version']}", "params": res["params"],
               "from": res["start"], "to": res["end"], "fills": res["trades"],
               "metrics": _summary(m),
               "universe_buy_and_hold_sharpe": _r(float(
                   bench.mean() / bench.std() * np.sqrt(PERIODS_PER_YEAR[meta["interval"]]))
                   if bench.std() else 0.0),
               "trials_used": used + 1, "trial_budget": self.cfg.per_job.backtests}
        return json.dumps(out, indent=1)

    def run_walkforward(self, path: str, grid: dict | None = None, params: dict | None = None) -> str:
        """Walk-forward test as the gate runs it (on visible history). `grid` maps param names to
        candidate lists; each combination costs one trial. Keep grids small (<= 3 params)."""
        from autotrader.backtest.walkforward import param_grid

        p = self._path(path)
        meta = self._meta(p)
        n = len(param_grid(grid or {}))
        used = self._spend_trials(n)
        bars = self._bars(meta["symbols"], meta["interval"], sealed=True)
        n_bars = len(set().union(*(df.index for df in bars.values())))
        res = self._run(p, "walkforward", params=params or {}, bars=bars, grid=grid or {},
                        train_bars=max(int(n_bars * 0.4), 50), test_bars=max(int(n_bars * 0.15), 20))
        self._trials = (n, [x for x in res["trial_sharpes"] if x is not None and math.isfinite(x)])
        out = {"strategy": f"{meta['name']}@{meta['version']}", "oos_metrics": _summary(res["metrics"]),
               "folds": [{k: _r(v) for k, v in f.items()} for f in res["folds"]],
               "trials_used": used + n, "trial_budget": self.cfg.per_job.backtests,
               "note": "dsr here counts only this grid; the gate counts all your trials"}
        return json.dumps(out, indent=1)

    # ----- submissions -----
    def _gate_and_record(self, p: Path, meta: dict, rationale: str, params: dict | None,
                         parent: StrategyVersion | None) -> str:
        with self.sessions() as s:
            count = s.scalar(select(func.count()).select_from(Submission)
                             .where(Submission.job_id == self.job_id))
            if count >= self.cfg.per_job.submissions:
                raise ToolFailure(f"submission limit ({self.cfg.per_job.submissions}) reached")
            taken = s.scalars(select(StrategyVersion).where(
                StrategyVersion.name == meta["name"], StrategyVersion.version == meta["version"])).first()
            pending = s.scalars(select(Submission).where(
                Submission.name == meta["name"], Submission.version == meta["version"],
                Submission.status.in_(("audit_pending", "approved", "deployed")))).first()
            if taken or pending:
                raise ToolFailure(f"{meta['name']}@{meta['version']} already exists; "
                                  "use a new name (or bump the version)")
            _, prior = self._trials_so_far(s)
        source = p.read_text()
        bars = self._bars(meta["symbols"], meta["interval"], sealed=False)
        res = self._run(p, "gate", params=params or {}, bars=bars, rules=self.policy.gate.model_dump(),
                        grid={}, holdout_start=self.cfg.holdout_start(meta["interval"], self.now),
                        prior_trial_sharpes=prior)
        import hashlib

        sub = Submission(job_id=self.job_id, parent_strategy_id=parent.id if parent else None,
                         name=meta["name"], version=meta["version"], interval=meta["interval"],
                         draft_path=str(p), code_sha=hashlib.sha256(source.encode()).hexdigest(),
                         source=source, rationale=rationale[:4000], params=params or {},
                         status="audit_pending" if res["passed"] else "gate_failed",
                         reasons=res["reasons"], gate_metrics=_clean(res["metrics"]),
                         holdout_metrics=_clean(res["holdout"]),
                         n_trials=int(res["metrics"].get("n_trials", 1)))
        with self.sessions() as s:
            s.add(sub)
            s.commit()
        # Holdout numbers are never shown: an agent that sees them could fit to them.
        shown = [r if not r.startswith("holdout") else "failed the sealed holdout (details hidden)"
                 for r in res["reasons"]]
        verdict = "PASSED the gate; queued for audit" if res["passed"] else "FAILED the gate"
        return json.dumps({"submission_id": sub.id, "verdict": verdict, "reasons": shown,
                           "oos_metrics": _summary(res["metrics"]),
                           "benchmark_sharpe": _r(res["metrics"].get("benchmark_sharpe"))}, indent=1)

    def submit_strategy(self, path: str, rationale: str, params: dict | None = None) -> str:
        """Submit your finished strategy for the gate (walk-forward + sealed holdout). `rationale`:
        the economic reason the edge should exist, and what you tried. Limited submissions."""
        p = self._path(path)
        meta = self._meta(p)
        if not NAME_RE.match(meta["name"]) or not VERSION_RE.match(meta["version"]):
            raise ToolFailure(f"name must match {NAME_RE.pattern} and version be digits")
        with self.sessions() as s:
            target = (self._job(s).target or {}).get("interval")
        if target and meta["interval"] != target:
            raise ToolFailure(f"this slot is for a {target} strategy, not {meta['interval']}")
        return self._gate_and_record(p, meta, rationale, params, None)

    def propose_version(self, slug: str, path: str, rationale: str, params: dict | None = None) -> str:
        """Submit an improved new version of an existing bot's strategy (same name and interval,
        higher version). The current version keeps trading; the new one must earn its own slot."""
        p = self._path(path)
        meta = self._meta(p)
        with self.sessions() as s:
            bot = s.scalars(select(Bot).where(Bot.slug == slug)).first()
            if bot is None:
                raise ToolFailure(f"no bot {slug}")
            parent = bot.strategy
        if meta["name"] != parent.name or meta["interval"] != parent.interval:
            raise ToolFailure(f"a new version must keep name {parent.name} and interval {parent.interval}")
        if not VERSION_RE.match(meta["version"]) or not parent.version.isdigit() \
                or int(meta["version"]) <= int(parent.version):
            raise ToolFailure(f"version must be digits and greater than {parent.version}")
        return self._gate_and_record(p, meta, rationale, params, parent)

    # ----- auditor -----
    def _audit_target(self, s: Session, submission_id: int) -> Submission:
        job = self._job(s)
        if job.submission_id != submission_id:
            raise ToolFailure(f"you are auditing submission {job.submission_id}, not {submission_id}")
        return s.get(Submission, submission_id)

    def get_submission(self, submission_id: int) -> str:
        """The submission under audit: source, rationale, gate report, and the research trail."""
        with self.sessions() as s:
            sub = self._audit_target(s, submission_id)
            calls = s.scalars(select(ToolCall).where(ToolCall.job_id == sub.job_id,
                                                     ToolCall.tool.in_(("run_backtest", "run_walkforward",
                                                                        "submit_strategy")))
                              .order_by(ToolCall.id)).all()
            trail = [{"tool": c.tool, "args": {k: v for k, v in c.args.items() if k != "rationale"},
                      "trials": c.trials, "ok": c.ok} for c in calls]
            h = sub.holdout_metrics
            out = {"submission_id": sub.id, "strategy": f"{sub.name}@{sub.version}",
                   "interval": sub.interval, "rationale": sub.rationale,
                   "walkforward_oos": _summary(sub.gate_metrics) | {
                       "benchmark_sharpe": _r(sub.gate_metrics.get("benchmark_sharpe")),
                       "n_trials": sub.n_trials},
                   "holdout": _summary(h) | {"benchmark_sharpe": _r(h.get("benchmark_sharpe"))},
                   "research_trail": trail}
        return json.dumps(out, indent=1) + "\n\n## Source\n```python\n" + sub.source + "\n```"

    def audit_verdict(self, submission_id: int, approve: bool, findings: str) -> str:
        """Record your verdict. `findings`: concrete issues (with line refs) or why it is sound."""
        with self.sessions() as s:
            sub = self._audit_target(s, submission_id)
            sub.audit = {"approve": bool(approve), "findings": findings[:6000], "job_id": self.job_id}
            s.commit()
        return f"recorded: {'APPROVE' if approve else 'REJECT'} submission {submission_id}"

    # ----- notes -----
    def write_note(self, topic: str, body: str) -> str:
        """Leave a note for future agents (what you tried, what failed and why, ideas)."""
        with self.sessions() as s:
            job = self._job(s)
            s.add(AgentNote(job_id=self.job_id, bot_id=job.bot_id, topic=topic[:120], body=body[:6000]))
            s.commit()
        return "noted"


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}
