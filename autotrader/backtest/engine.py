"""Event-driven bar replay using the same Strategy.on_bar as live trading.

Timing model, per bar t of the union timeline:
  1. targets decided at the close of bar t-1 fill at the *open* of bar t
     (with adverse slippage and fees);
  2. positions are marked at the close of bar t;
  3. the strategy sees bars up to and including t and returns new targets.
A strategy can therefore never trade on the bar that produced its signal.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.backtest.costs import CostModel
from autotrader.backtest.metrics import compute_metrics, returns_from_equity
from autotrader.data.intervals import PERIODS_PER_YEAR
from autotrader.strategies.base import Strategy, StrategyContext


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


class _Book:
    """Cash, signed positions and average cost, with realized P&L on reductions."""

    def __init__(self, cash: float):
        self.cash = cash
        self.qty: dict[str, float] = {}
        self.avg: dict[str, float] = {}

    def fill(self, symbol: str, qty: float, price: float, fees: float) -> float | None:
        q = self.qty.get(symbol, 0.0)
        a = self.avg.get(symbol, 0.0)
        self.cash -= qty * price + fees
        realized = None
        if q == 0 or math.copysign(1, q) == math.copysign(1, qty):
            new_q = q + qty
            self.avg[symbol] = (q * a + qty * price) / new_q
        else:
            closed = min(abs(qty), abs(q))
            realized = closed * (price - a) * math.copysign(1, q) - fees
            new_q = q + qty
            if new_q != 0 and math.copysign(1, new_q) != math.copysign(1, q):
                self.avg[symbol] = price  # flipped: remainder opened at this price
        if abs(new_q) < 1e-9:
            self.qty.pop(symbol, None)
            self.avg.pop(symbol, None)
        else:
            self.qty[symbol] = new_q
        return realized

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.qty.items())


def _clean_targets(targets: dict[str, float], held: set[str], max_gross: float,
                   tradable: set[str]) -> dict[str, float]:
    out = {s: float(w) for s, w in targets.items() if s in tradable and w == w}
    gross = sum(abs(w) for w in out.values())
    if gross > max_gross:
        out = {s: w * max_gross / gross for s, w in out.items()}
    for s in held:
        out.setdefault(s, 0.0)
    return out


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
    lookback = strategy.lookback
    ppy = PERIODS_PER_YEAR[strategy.interval]

    # Per symbol: row position for each timeline step (-1 when no bar), and bars-so-far count.
    feats = {s: f for s, df in bars.items() if (f := strategy.features(df)) is not None}

    rows: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}
    for s, df in bars.items():
        pos = df.index.get_indexer(timeline)
        rows[s] = pos
        counts[s] = np.searchsorted(df.index.values, timeline.values, side="right")
        opens[s] = df["open"].to_numpy()
        closes[s] = df["close"].to_numpy()

    book = _Book(config.initial_cash)
    last_price: dict[str, float] = {}
    pending: dict[str, float] = {}
    trades: list[Trade] = []
    equity_ts: list[pd.Timestamp] = []
    equity_vals: list[float] = []
    traded_notional = 0.0
    borrow_per_bar = costs.borrow_rate_annual / ppy

    for i, ts in enumerate(timeline):
        live = start is None or ts >= start
        here = {s for s in bars if rows[s][i] >= 0}

        # 1. Fill pending targets at this bar's open.
        if pending:
            open_px = {s: opens[s][rows[s][i]] for s in here}
            marks = {**last_price, **open_px}
            equity_open = book.equity(marks)
            orders = []
            for s, w in list(pending.items()):
                if s not in here:
                    continue  # stays pending until the symbol trades
                del pending[s]
                px = open_px[s]
                target = w * equity_open / px
                if not config.fractional:
                    target = float(math.trunc(target))
                delta = target - book.qty.get(s, 0.0)
                if abs(delta * px) >= config.min_trade_value:
                    orders.append((s, delta, px))
            for s, delta, px in sorted(orders, key=lambda o: o[1]):  # sells first
                fill_px = costs.fill_price(px, delta)
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

        if not live:
            continue
        equity_ts.append(ts)
        equity_vals.append(equity)

        # 3. Ask the strategy for new targets.
        if equity <= 0:
            break
        ctx_bars, ctx_feats = {}, {}
        for s, df in bars.items():
            k = counts[s][i]
            if k > 0:
                ctx_bars[s] = df.iloc[max(0, k - lookback):k]
                if s in feats:
                    ctx_feats[s] = feats[s].iloc[max(0, k - lookback):k]
        weights = {s: q * last_price[s] / equity for s, q in book.qty.items()}
        targets = strategy.on_bar(StrategyContext(
            now=ts, bars=ctx_bars, features=ctx_feats, weights=weights, equity=equity))
        if targets is not None:
            pending = _clean_targets(targets, set(book.qty), config.max_gross, set(bars))

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
