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
    from autotrader.views import roster

    with _sessions()() as session:
        rows = roster(session)
    for r in rows:
        eq = f"${r['equity']:,.2f}" if r["equity"] is not None else "-"
        typer.echo(f"{r['slug']:18} {r['strategy']:24} {r['stage']:16} equity {eq:>12}  {r['positions']}")


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
    from autotrader.evaluator.jobs import spy_daily_returns
    from autotrader.views import league as league_rows

    spy = spy_daily_returns(BarArchive(), pd.Timestamp.now(tz="UTC"))
    with _sessions()() as session:
        rows = league_rows(session, spy)
    typer.echo(f"{'bot':18} {'stage':16} {'days':>4} {'trades':>6} {'return':>8} {'vsSPY':>8} "
               f"{'sharpe':>7} {'psr':>5} {'maxDD':>7} {'capital':>9}")
    for r in rows:
        v = {k: r[k] or 0.0 for k in ("total_return", "excess_return", "sharpe", "psr", "max_drawdown")}
        typer.echo(f"{r['slug']:18} {r['stage']:16} {r['days']:4d} {r['trades']:6d} "
                   f"{v['total_return']:8.2%} {v['excess_return']:8.2%} {v['sharpe']:7.2f} "
                   f"{v['psr']:5.2f} {v['max_drawdown']:7.2%} {r['capital']:9,.0f}")


@app.command()
def graveyard() -> None:
    """Eradicated bots and why they died."""
    from autotrader.views import graveyard as graveyard_rows

    with _sessions()() as session:
        for r in graveyard_rows(session):
            typer.echo(f"{r['died'][:10]} {r['slug']}: {r['autopsy']}\n")


@app.command()
def report(date: str = typer.Option(None, help="ET date (YYYY-MM-DD); default today"),
           post: bool = typer.Option(False, help="Post to Discord if not posted yet")) -> None:
    """Build (or rebuild) the daily report and print it."""
    from autotrader.reports.daily import publish_daily_report

    rep = publish_daily_report(_sessions(), BarArchive(), date, post=post)
    typer.echo(rep.body_md)
    if rep.commentary:
        typer.echo(f"\n## Supervisor\n{rep.commentary}")


alerts_app = typer.Typer(help="Alert delivery (Discord / ntfy).", no_args_is_help=True)
app.add_typer(alerts_app, name="alerts")


@alerts_app.command("test")
def alerts_test() -> None:
    """Send a test alert to every configured channel."""
    from autotrader.alerts import alert
    from autotrader.config import get_settings

    s = get_settings()
    dm = bool(s.discord_bot_token and s.discord_user_id)
    channels = [n for n, v in (("Discord DM", dm), ("Discord webhook", s.discord_webhook and not dm),
                               ("ntfy", s.ntfy_url)) if v]
    if not channels:
        raise typer.BadParameter("nothing configured: set AT_DISCORD_BOT_TOKEN and AT_DISCORD_USER_ID "
                                 "(or AT_DISCORD_WEBHOOK) in .env")
    if dm:  # deliver directly so a failure is shown here instead of only logged
        from autotrader.alerts import send_discord

        send_discord({"content": "**Autotrader test**\nIf you can read this, DMs reach you."})
        channels.remove("Discord DM")
        typer.echo("Discord DM sent")
        if not channels:
            return
    alert("Autotrader test alert", "If you can read this, alerts reach you.")
    typer.echo(f"sent to {', '.join(channels)}")


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
    from autotrader.views import agents_overview

    sessions, archive, cfg, policy = _agents_ctx()
    with sessions() as s:
        o = agents_overview(s, cfg, policy)
    if o["paused"]:
        typer.echo(f"PAUSED: {o['paused'].get('reason')}")
    sl = o["slots"]
    typer.echo(f"model: {o['model']}")
    typer.echo(f"slots: {sl['bots']}/{sl['total']} bots, {sl['in_flight']} in research, {sl['open']} open")
    typer.echo(f"budget: {o['budget']['text']}  (next researcher ~${o['budget']['estimate']:.2f})")
    for c in o["counts"]:
        typer.echo(f"  {c['agent']:10} {c['status']:10} {c['n']}")


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
    from autotrader.orchestrator import control

    with _agent_sessions()() as s:
        try:
            if role == "researcher":
                job = control.queue_research(s, interval, note)
            elif role == "reviewer" and slug:
                job = control.queue_review(s, slug, note)
            else:
                raise typer.BadParameter("role must be researcher, or reviewer with --slug")
        except control.ControlError as exc:
            raise typer.BadParameter(str(exc)) from None
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
    """Resume the orchestrator after it paused (e.g. an expired OpenCode key, or by hand)."""
    from autotrader.orchestrator import control

    with _agent_sessions()() as s:
        resumed = control.resume(s)
        s.commit()
    typer.echo("resumed" if resumed else "not paused")


@agents_app.command("pause")
def agents_pause(reason: str = typer.Option("paused by hand", help="Shown in status")) -> None:
    """Stop starting new agent jobs (a running job finishes)."""
    from autotrader.orchestrator import control

    with _agent_sessions()() as s:
        control.pause(s, reason)
        s.commit()
    typer.echo("paused")


@agents_app.command("cancel")
def agents_cancel(job_id: int) -> None:
    """Cancel a queued job."""
    from autotrader.orchestrator import control

    with _agent_sessions()() as s:
        try:
            control.cancel_job(s, job_id)
        except control.ControlError as exc:
            raise typer.BadParameter(str(exc)) from None
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
