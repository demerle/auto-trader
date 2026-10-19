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
    run_forever()


if __name__ == "__main__":
    app()
