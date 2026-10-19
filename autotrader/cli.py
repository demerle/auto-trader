import json
import logging

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
