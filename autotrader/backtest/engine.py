"""Event-driven bar replay using the same Strategy.on_bar, sizing and Book as live trading.

Timing model, per bar t of the union timeline:
  1. orders sized at the close of bar t-1 fill at the *open* of bar t
     (with adverse slippage and fees); an order for a symbol with no bar at t
     stays pending until that symbol trades;
  2. positions are marked at the close of bar t;
  3. the strategy sees bars up to and including t and returns target weights,
     which are sized into share quantities at bar t's close (as a live order would be).
A strategy can therefore never trade on the bar that produced its signal.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.backtest.costs import CostModel
from autotrader.backtest.metrics import compute_metrics, returns_from_equity
from autotrader.broker.sizing import clean_targets, order_deltas, target_quantities
from autotrader.data.intervals import PERIODS_PER_YEAR
from autotrader.ledger.book import Book
from autotrader.strategies.base import Strategy, make_context


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    fractional: bool = False
    max_gross: float = 1.0
    min_trade_value: float = 1.0


@dataclass(frozen=True)
class Trade:
    ts: pd.Timestamp
    symbol: str
    qty: float
    price: float
    fees: float
    realized_pnl: float | None  # set when the fill reduces/closes a position


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    interval: str
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    final_positions: dict[str, float] = field(default_factory=dict)
    turnover: float = 0.0

    @property
    def returns(self) -> pd.Series:
        return returns_from_equity(self.equity)

    def metrics(self, benchmark_returns: pd.Series | None = None, n_trials: int = 1,
                sr_variance: float = 0.0) -> dict[str, float]:
        realized = [t.realized_pnl for t in self.trades if t.realized_pnl is not None]
        return compute_metrics(
            self.equity, PERIODS_PER_YEAR[self.interval], realized, self.turnover,
            benchmark_returns, n_trials, sr_variance,
        )


def run_backtest(
    strategy: Strategy,
    bars: Mapping[str, pd.DataFrame],
    config: BacktestConfig = BacktestConfig(),
    costs: CostModel = CostModel(),
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Replay `bars` (canonical frames at strategy.interval) through `strategy`.

    History before `start` is used only as warm-up; equity is recorded from `start`.
    """
    bars = {s: df for s, df in bars.items() if not df.empty}
    if end is not None:
        bars = {s: df[df.index <= end] for s, df in bars.items()}
    timeline = pd.DatetimeIndex(sorted(set().union(*(df.index for df in bars.values())))) \
        if bars else pd.DatetimeIndex([], tz="UTC")
    ppy = PERIODS_PER_YEAR[strategy.interval]
    feats = {s: f for s, df in bars.items() if (f := strategy.features(df)) is not None}

    # Per symbol: row position for each timeline step (-1 when no bar), and bars-so-far count.
    rows: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}
    for s, df in bars.items():
        rows[s] = df.index.get_indexer(timeline)
        counts[s] = np.searchsorted(df.index.values, timeline.values, side="right")
        opens[s] = df["open"].to_numpy()
        closes[s] = df["close"].to_numpy()

    book = Book(config.initial_cash)
    last_price: dict[str, float] = {}
    pending: dict[str, float] = {}  # symbol -> signed share delta awaiting a fill
    trades: list[Trade] = []
    equity_ts: list[pd.Timestamp] = []
    equity_vals: list[float] = []
    traded_notional = 0.0
    borrow_per_bar = costs.borrow_rate_annual / ppy

    for i, ts in enumerate(timeline):
        here = {s for s in bars if rows[s][i] >= 0}

        # 1. Fill pending orders at this bar's open (sells first).
        for s, delta in sorted(((s, d) for s, d in pending.items() if s in here), key=lambda o: o[1]):
            del pending[s]
            fill_px = costs.fill_price(opens[s][rows[s][i]], delta)
            fees = costs.fees(delta, fill_px)
            realized = book.fill(s, delta, fill_px, fees)
            trades.append(Trade(ts, s, delta, fill_px, fees, realized))
            traded_notional += abs(delta * fill_px)

        # 2. Mark to market at the close; accrue borrow on shorts.
        for s in here:
            last_price[s] = closes[s][rows[s][i]]
        short_value = sum(-q * last_price[s] for s, q in book.qty.items() if q < 0)
        book.cash -= short_value * borrow_per_bar
        equity = book.equity(last_price)

        if start is not None and ts < start:
            continue
        equity_ts.append(ts)
        equity_vals.append(equity)

        # 3. Ask the strategy for targets and size them at this close.
        if equity <= 0:
            break
        upto = {s: df.iloc[:counts[s][i]] for s, df in bars.items()}
        ctx = make_context(strategy, ts, upto, {s: f.iloc[:counts[s][i]] for s, f in feats.items()},
                           book.weights(last_price), equity)
        targets = strategy.on_bar(ctx)
        if targets is not None:
            tradable = set(last_price)
            weights = clean_targets(targets, set(book.qty), config.max_gross, tradable)
            target_qty = target_quantities(weights, equity, last_price, config.fractional)
            pending = order_deltas(target_qty, book.qty, last_price, config.min_trade_value)

    equity_series = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_ts, name="ts"), name="equity")
    avg_equity = float(equity_series.mean()) if len(equity_series) else config.initial_cash
    years = len(equity_series) / ppy if ppy else 0
    return BacktestResult(
        strategy=strategy.key,
        params=strategy.params.model_dump(),
        interval=strategy.interval,
        equity=equity_series,
        trades=trades,
        final_positions=dict(book.qty),
        turnover=(traded_notional / avg_equity / years) if years and avg_equity else 0.0,
    )
