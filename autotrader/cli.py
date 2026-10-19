import json
import logging
from pathlib import Path

import pandas as pd
import typer

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.backtest.walkforward import walk_forward
from autotrader.config.universes import resolve
from autotrader.data.archive import BarArchive
from autotrader.strategies import registry

app = typer.Typer(help="Autotrader command line.", no_args_is_help=True)


def _load(symbols: tuple[str, ...], interval, years: float, refresh: bool) -> dict[str, pd.DataFrame]:
    archive = BarArchive()
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(365 * years))
    return {s: archive.get(s, interval, start=start, refresh=refresh) for s in symbols}


def _print(metrics: dict) -> None:
    typer.echo(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))


@app.command()
def strategies() -> None:
    """List available strategies."""
    for key, cls in sorted(registry.discover().items()):
        typer.echo(f"{key:32} {cls.interval:4} {cls.description}")


@app.command()
def backtest(
    name: str,
    symbols: str = typer.Option(None, help="Comma-separated; defaults to the strategy universe"),
    years: float = 5.0,
    cash: float = 10_000.0,
    refresh: bool = True,
    params: str = typer.Option("{}", help="JSON params"),
) -> None:
    """Backtest a strategy on archived (and freshly fetched) bars."""
    cls = registry.get(name)
    strat = cls(json.loads(params))
    syms = tuple(symbols.split(",")) if symbols else strat.symbols
    bars = _load(syms, cls.interval, years, refresh)
    bench = _load(("SPY",), cls.interval, years, refresh)["SPY"]["close"].pct_change().dropna()
    res = run_backtest(strat, bars, BacktestConfig(initial_cash=cash))
    typer.echo(f"{res.strategy} on {len(syms)} symbols, {len(res.equity)} bars, {len(res.trades)} fills")
    _print(res.metrics(benchmark_returns=bench))


@app.command()
def walkforward(
    name: str,
    grid: str = typer.Option("{}", help='JSON grid, e.g. {"k_period": [14, 21]}'),
    years: float = 8.0,
    train: int = 504,
    test: int = 126,
    refresh: bool = True,
) -> None:
    """Walk-forward tune and score out of sample."""
    cls = registry.get(name)
    bars = _load(resolve(cls.universe), cls.interval, years, refresh)
    res = walk_forward(cls, bars, json.loads(grid), train_bars=train, test_bars=test)
    typer.echo(f"{res.strategy}: {len(res.folds)} folds, {res.n_trials} trials per fold")
    _print(res.metrics)


@app.command()
def collect() -> None:
    """Run the bar collector daemon."""
    from autotrader.data.collector import run_forever

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_forever()


bots_app = typer.Typer(help="Manage the bot roster.", no_args_is_help=True)
risk_app = typer.Typer(help="Human-only risk config.", no_args_is_help=True)
app.add_typer(bots_app, name="bots")
app.add_typer(risk_app, name="risk")


def _arena():
    from autotrader.broker.alpaca import AlpacaBroker
    from autotrader.config import get_settings
    from autotrader.db.session import session_factory
    from autotrader.engine.arena import Arena
    from autotrader.risk.limits import load_limits

    s = get_settings()
    return Arena(session_factory(), AlpacaBroker.paper_from_settings(), BarArchive(),
                 load_limits(s.risk_file), collar_bps=s.collar_bps)


@bots_app.command("sync")
def bots_sync() -> None:
    """Create bots listed in config/bots.yaml that don't exist yet."""
    from autotrader.config import get_settings
    from autotrader.db.session import session_factory
    from autotrader.engine.roster import sync_roster

    with session_factory()() as session:
        for line in sync_roster(session, get_settings().bots_file):
            typer.echo(line)
        session.commit()


@bots_app.command("list")
def bots_list() -> None:
    """Show each bot's stage, equity (at last tick) and positions."""
    from sqlalchemy import select

    from autotrader.db.models import Bot, EquitySnapshot
    from autotrader.db.session import session_factory

    with session_factory()() as session:
        for bot in session.scalars(select(Bot).order_by(Bot.id)).all():
            snap = session.scalars(select(EquitySnapshot).where(EquitySnapshot.bot_id == bot.id)
                                   .order_by(EquitySnapshot.ts.desc()).limit(1)).first()
            eq = f"${snap.equity:,.2f}" if snap else "-"
            pos = {k: v for k, v in (snap.positions if snap else {}).items()}
            typer.echo(f"{bot.slug:18} {bot.strategy.name}@{bot.strategy.version:3} "
                       f"{bot.stage.value:16} equity {eq:>12}  {pos}")


@app.command()
def engine(
    once: str = typer.Option(None, help="Run a single tick for this interval and exit"),
    dry_run: bool = typer.Option(False, help="With --once: print intended orders, change nothing"),
) -> None:
    """Run the trading engine (paper account)."""
    from autotrader.engine.loop import run_forever

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    arena = _arena()
    if once:
        typer.echo(str(arena.tick(once, dry_run=dry_run)))
        return
    run_forever(arena)


@app.command()
def kill(reason: str = typer.Option("manual", help="Why")) -> None:
    """KILL SWITCH: cancel all orders, flatten every bot, stop trading."""
    typer.echo(str(_arena().kill(reason)))


@app.command()
def unkill() -> None:
    """Turn the kill switch off so the engine trades again."""
    _arena().unkill()
    typer.echo("kill switch off")


@app.command("reconcile")
def reconcile_cmd() -> None:
    """Compare summed bot ledgers with the broker's positions."""
    from autotrader.broker.reconcile import reconcile

    arena = _arena()
    with arena.sessions() as session:
        arena.desk.sync(session)
        session.commit()
        drift = reconcile(session, arena.broker)
    typer.echo("no drift" if not drift else "\n".join(map(str, drift)))


def _sessions():
    from autotrader.db.session import session_factory

    return session_factory()


@app.command()
def evaluate(dry_run: bool = typer.Option(False, help="Show verdicts without applying them"),
             allocate: bool = typer.Option(False, help="Also run the weekly capital tilt")) -> None:
    """Score every paper bot now: keep / eradicate / flag live-eligible."""
    from autotrader.evaluator.jobs import run_daily
    from autotrader.evaluator.policy import load_policy

    for line in run_daily(_sessions(), BarArchive(), load_policy(), allocate=allocate,
                          apply=not dry_run):
        typer.echo(line)


@app.command()
def league() -> None:
    """Rank active bots by their paper record."""
    from sqlalchemy import select

    from autotrader.db.models import Bot
    from autotrader.evaluator.jobs import spy_daily_returns
    from autotrader.evaluator.record import build_record
    from autotrader.evaluator.run import EVALUATED_STAGES

    spy = spy_daily_returns(BarArchive(), pd.Timestamp.now(tz="UTC"))
    with _sessions()() as session:
        rows = []
        for bot in session.scalars(select(Bot).where(Bot.stage.in_(EVALUATED_STAGES))).all():
            rec = build_record(session, bot, spy)
            rows.append((rec.metrics.get("sharpe", 0.0), bot, rec))
    typer.echo(f"{'bot':18} {'stage':16} {'days':>4} {'trades':>6} {'return':>8} {'vsSPY':>8} "
               f"{'sharpe':>7} {'psr':>5} {'maxDD':>7} {'capital':>9}")
    for _, bot, rec in sorted(rows, key=lambda r: r[0], reverse=True):
        m = rec.metrics
        typer.echo(f"{bot.slug:18} {bot.stage.value:16} {rec.days:4d} {rec.trades:6d} "
                   f"{m.get('total_return', 0):8.2%} {m.get('excess_return', 0):8.2%} "
                   f"{m.get('sharpe', 0):7.2f} {m.get('psr', 0):5.2f} {m.get('max_drawdown', 0):7.2%} "
                   f"{bot.capital:9,.0f}")


@app.command()
def graveyard() -> None:
    """Eradicated bots and why they died."""
    from sqlalchemy import select

    from autotrader.db.models import Bot, BotStage

    with _sessions()() as session:
        for bot in session.scalars(select(Bot).where(Bot.stage == BotStage.ERADICATED)).all():
            typer.echo(f"{bot.stage_since:%Y-%m-%d} {bot.slug}: {bot.notes}\n")


@app.command()
def allocate(dry_run: bool = typer.Option(True, help="Show moves without applying")) -> None:
    """Weekly capital tilt toward better bots (normally run by the engine on Fridays)."""
    from autotrader.evaluator.allocator import apply_allocation, plan_allocation
    from autotrader.evaluator.policy import load_policy

    policy = load_policy()
    with _sessions()() as session:
        moves = plan_allocation(session, policy)
        if not dry_run:
            moves = apply_allocation(session, policy, moves)
            session.commit()
    for mv in moves:
        typer.echo(f"{mv.bot.slug:18} ${mv.current:,.0f} -> ${mv.target:,.0f} (shrunk Sharpe {mv.score:+.2f})")


@app.command()
def gate(
    name: str,
    years: float = 10.0,
    grid: str = typer.Option("{}", help="JSON param grid tried in each walk-forward fold"),
    save: bool = typer.Option(False, help="Store results as the strategy's backtest expectation"),
    refresh: bool = True,
) -> None:
    """Walk-forward gate a strategy must pass to earn a paper slot."""
    from sqlalchemy import select

    from autotrader.db.models import StrategyVersion
    from autotrader.evaluator.gate import run_gate
    from autotrader.evaluator.policy import load_policy
    from autotrader.evaluator.run import clean_json

    cls = registry.get(name)
    bars = _load(resolve(cls.universe), cls.interval, years, refresh)
    res = run_gate(cls, bars, load_policy().gate, json.loads(grid))
    typer.echo(f"{res.strategy}: {'PASS' if res.passed else 'FAIL'}")
    for r in res.reasons:
        typer.echo(f"  - {r}")
    _print({k: v for k, v in res.metrics.items() if isinstance(v, float)})
    if save:
        with _sessions()() as session:
            sv = session.scalars(select(StrategyVersion).where(
                StrategyVersion.name == cls.name, StrategyVersion.version == cls.version)).first()
            if sv is None:
                raise typer.BadParameter(f"{res.strategy} is not registered (run `bots sync`)")
            sv.backtest_metrics = clean_json(res.metrics | {"gate_passed": float(res.passed)})
            session.commit()
        typer.echo("saved as backtest expectation")


agents_app = typer.Typer(help="Strategy-research agents (opencode on OpenCode Go).",
                         no_args_is_help=True)
app.add_typer(agents_app, name="agents")


def _agents_ctx():
    from autotrader.evaluator.policy import load_policy
    from autotrader.orchestrator.config import load_agents_config

    return _agent_sessions(), BarArchive(), load_agents_config(), load_policy()


def _agent_sessions():
    return _sessions()


@app.command()
def orchestrator(
    once: bool = typer.Option(False, help="Run one orchestrator tick (may run one agent job)"),
    dry_run: bool = typer.Option(False, help="Show what would run; change nothing"),
) -> None:
    """Run the agent orchestrator: fill slots, run agent jobs, audit and deploy strategies."""
    from autotrader.orchestrator.loop import run_forever, tick

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sessions, archive, cfg, policy = _agents_ctx()
    if once or dry_run:
        for line in tick(sessions, archive, cfg, policy, dry_run=dry_run):
            typer.echo(line)
        return
    run_forever(sessions, archive, cfg, policy)


@agents_app.command("status")
def agents_status() -> None:
    """Slots, budget and the job queue at a glance."""
    from sqlalchemy import func, select

    from autotrader.db.models import AgentJob
    from autotrader.orchestrator import budget, slots

    sessions, archive, cfg, policy = _agents_ctx()
    from autotrader.engine.state import get_state
    from autotrader.orchestrator.loop import PAUSED

    with sessions() as s:
        if paused := get_state(s, PAUSED):
            typer.echo(f"PAUSED: {paused.get('reason')}")
        active = slots.active_bots(s)
        typer.echo(f"slots: {len(active)}/{policy.slots} bots, {slots.in_flight(s)} in research, "
                   f"{slots.open_slots(s, policy)} open")
        chk = budget.check(s, "researcher", cfg)
        typer.echo(f"budget: {chk}  (next researcher ~${chk.estimate:.2f})")
        counts = s.execute(select(AgentJob.agent, AgentJob.status, func.count())
                           .group_by(AgentJob.agent, AgentJob.status)).all()
        for agent, status, n in sorted(counts):
            typer.echo(f"  {agent:10} {status:10} {n}")


@agents_app.command("jobs")
def agents_jobs(limit: int = 20) -> None:
    """Recent agent jobs."""
    from sqlalchemy import select

    from autotrader.db.models import AgentJob

    with _agent_sessions()() as s:
        for j in s.scalars(select(AgentJob).order_by(AgentJob.id.desc()).limit(limit)):
            when = f"{j.started_at:%m-%d %H:%M}" if j.started_at else "-"
            err = (j.result or {}).get("error") or ""
            typer.echo(f"#{j.id:<4} {j.agent:10} {j.status:9} {when:11} {j.model or '-':30} "
                       f"${j.cost_usd:6.3f}  {json.dumps(j.target)} {err[:80]}")


@agents_app.command("submissions")
def agents_submissions(limit: int = 20) -> None:
    """Strategies agents submitted, with gate and audit outcomes."""
    from sqlalchemy import select

    from autotrader.db.models import Submission

    with _agent_sessions()() as s:
        for x in s.scalars(select(Submission).order_by(Submission.id.desc()).limit(limit)):
            m = x.gate_metrics or {}
            typer.echo(f"#{x.id:<4} job {x.job_id:<4} {x.name}@{x.version:3} {x.interval:4} "
                       f"{x.status:13} OOS Sharpe {m.get('sharpe') or 0:5.2f}  "
                       f"{'; '.join(x.reasons or [])[:100]}")
            if x.audit:
                typer.echo(f"      audit: {'APPROVE' if x.audit.get('approve') else 'REJECT'} "
                           f"{x.audit.get('findings', '')[:200]}")


@agents_app.command("queue")
def agents_queue(role: str = typer.Argument("researcher"),
                 interval: str = typer.Option("1d", help="Target interval (researcher)"),
                 slug: str = typer.Option(None, help="Bot to review (reviewer)"),
                 note: str = typer.Option("", help="Extra instructions for the agent")) -> None:
    """Queue a researcher (or reviewer) job by hand."""
    from autotrader.db.models import AgentJob
    from autotrader.orchestrator.slots import research_prompt

    with _agent_sessions()() as s:
        if role == "researcher":
            job = AgentJob(agent=role, model="", target={"interval": interval, "reason": "queued by hand"},
                           prompt=research_prompt(interval, "queued by hand", note))
        elif role == "reviewer" and slug:
            from sqlalchemy import select

            from autotrader.db.models import Bot

            bot = s.scalars(select(Bot).where(Bot.slug == slug)).one()
            job = AgentJob(agent=role, model="", bot_id=bot.id, target={"slug": slug},
                           prompt=f"Review of {slug} requested by hand. {note}")
        else:
            raise typer.BadParameter("role must be researcher, or reviewer with --slug")
        s.add(job)
        s.commit()
        typer.echo(f"queued job {job.id}")


@agents_app.command("auth-export")
def agents_auth_export(out: Path = typer.Option(Path("secrets/opencode-go.json"),
                                                help="Where the agents container reads it from")) -> None:
    """Export ONLY your OpenCode Go credential for the agents container (run on the host
    after `opencode auth login`). Other providers' keys are never exported."""
    import os
    import subprocess

    raw = subprocess.run(["opencode", "auth", "export"], capture_output=True, text=True, timeout=60,
                         check=True).stdout
    keep = [c for c in json.loads(raw) if c.get("integrationID") == "opencode-go"]
    if not keep:
        raise typer.BadParameter("no opencode-go credential found; run `opencode auth login` first")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(keep, f)
    os.chmod(out, 0o600)
    typer.echo(f"wrote {len(keep)} OpenCode Go credential(s) to {out} (mode 600); "
               "the agents container picks it up before its next job")


@agents_app.command("resume")
def agents_resume() -> None:
    """Resume the orchestrator after it paused itself (e.g. an expired OpenCode key)."""
    from autotrader.db.models import SystemState
    from autotrader.orchestrator.loop import PAUSED

    with _agent_sessions()() as s:
        row = s.get(SystemState, PAUSED)
        if row is None:
            typer.echo("not paused")
            return
        s.delete(row)
        s.commit()
    typer.echo("resumed")


@agents_app.command("cancel")
def agents_cancel(job_id: int) -> None:
    """Cancel a queued job."""
    from autotrader.db.models import AgentJob

    with _agent_sessions()() as s:
        job = s.get(AgentJob, job_id)
        if job is None or job.status != "queued":
            raise typer.BadParameter("only queued jobs can be cancelled")
        job.status = "cancelled"
        s.commit()
    typer.echo(f"cancelled job {job_id}")


@agents_app.command("logs")
def agents_logs(job_id: int, raw: bool = typer.Option(False, help="Print raw opencode events")) -> None:
    """What an agent did: its tool calls and its final text."""
    from sqlalchemy import select

    from autotrader.db.models import AgentJob, ToolCall

    with _agent_sessions()() as s:
        job = s.get(AgentJob, job_id)
        calls = s.scalars(select(ToolCall).where(ToolCall.job_id == job_id).order_by(ToolCall.id)).all()
    typer.echo(f"job {job.id} {job.agent} {job.status} on {job.model}, ${job.cost_usd:.3f}")
    for c in calls:
        args = {k: (v if len(str(v)) < 80 else str(v)[:77] + "...") for k, v in c.args.items()}
        typer.echo(f"  {c.ts:%H:%M:%S} {c.tool:16} {'ok ' if c.ok else 'ERR'} {c.duration_ms:6d}ms {args}")
    if job.transcript_path and Path(job.transcript_path).exists():
        text = Path(job.transcript_path).read_text(errors="replace")
        if raw:
            typer.echo(text)
        else:
            from autotrader.orchestrator.runner import final_text

            typer.echo("\n" + final_text(text.splitlines()))


@risk_app.command("sign")
def risk_sign() -> None:
    """Record the hash of config/risk.yaml after a human has reviewed it."""
    from autotrader.config import get_settings
    from autotrader.risk.limits import load_limits, sign

    path = get_settings().risk_file
    load_limits(path, verify=False)  # validate before signing
    typer.echo(f"signed {path}: {sign(path)}")


if __name__ == "__main__":
    app()
