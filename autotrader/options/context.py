"""Build an OptionContext and value positions; shared by the simulator and the live engine."""

from collections.abc import Mapping

import pandas as pd

from autotrader.options.book import OptionBook
from autotrader.options.helpers import Chain
from autotrader.options.occ import parse
from autotrader.options.strategy import OptionContext


def intrinsic(symbol: str, spot: float) -> float:
    c = parse(symbol)
    return max(spot - c.strike, 0.0) if c.right == "C" else max(c.strike - spot, 0.0)


def marks_for(symbols, chains: Mapping[str, Chain], spot: Mapping[str, float],
              last: Mapping[str, float] | None = None) -> dict[str, float]:
    """Mark per share for each symbol: chain mark, else the last known mark, else intrinsic."""
    out = {}
    for s in symbols:
        c = parse(s)
        chain = chains.get(c.underlying)
        m = chain.mark(s) if chain is not None else None
        if m is None:
            m = (last or {}).get(s)
        if m is None:
            m = intrinsic(s, spot.get(c.underlying, 0.0))
        out[s] = float(m)
    return out


def build_context(now: pd.Timestamp, chains: Mapping[str, Chain], spot: Mapping[str, float],
                  bars: Mapping[str, pd.DataFrame], daily: Mapping[str, pd.DataFrame],
                  vix: float | None, vix1d: float | None, book: OptionBook, marks: Mapping[str, float],
                  week_start_equity: float, session_open: pd.Timestamp, session_close: pd.Timestamp,
                  week_last_session: bool) -> OptionContext:
    closed = tuple(c for c in book.closed if c.closed_at is not None and c.closed_at >= session_open)
    return OptionContext(
        now=now, chains=dict(chains), spot=dict(spot), bars=dict(bars), daily=dict(daily),
        vix=vix, vix1d=vix1d, positions=book.snapshot(marks), closed_today=closed,
        equity=book.equity(marks), cash=book.cash, week_start_equity=week_start_equity,
        session_open=session_open, session_close=session_close, week_last_session=week_last_session,
    )
