"""Helpers for option strategies (pure; strategies may import this module).

A `Chain` wraps one underlying's nowcast chain. Prices are per share; one contract
is 100 shares. Columns: symbol, underlying, expiry (date), right ("C"/"P"), strike,
bid, ask (CBOE, ~15 min old), mark, est_bid, est_ask (repriced at the live spot),
iv, delta, gamma, theta, vega, oi, volume.
"""

from datetime import date

import numpy as np
import pandas as pd

MULTIPLIER = 100


class Chain:
    def __init__(self, underlying: str, spot: float, quote_time: pd.Timestamp, df: pd.DataFrame,
                 today: date, index: dict[str, int] | None = None):
        self.underlying = underlying
        self.spot = spot
        self.quote_time = quote_time
        self.df = df.reset_index(drop=True)
        self.today = today
        self._by_symbol = index if index is not None else \
            ({s: i for i, s in enumerate(self.df["symbol"])} if len(self.df) else {})
        self._exps: list[date] | None = None

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Chain({self.underlying} @ {self.spot:.2f}, {len(self.df)} contracts)"

    # ----- selection -----
    def expiries(self) -> list[date]:
        if self._exps is None:
            self._exps = sorted(set(self.df["expiry"])) if len(self.df) else []
        return self._exps

    def expiry(self, n: int = 0, min_dte: int = 0) -> date | None:
        """The n-th expiry at least `min_dte` calendar days out (n=0, min_dte=0: today's 0DTE if any)."""
        exps = [e for e in self.expiries() if (e - self.today).days >= min_dte]
        return exps[n] if len(exps) > n else None

    def weekly(self) -> date | None:
        """The last expiry of this week (usually Friday); None if it has passed."""
        friday = (pd.Timestamp(self.today) + pd.Timedelta(days=4 - self.today.weekday())).date()
        exps = [e for e in self.expiries() if self.today <= e <= friday]
        return exps[-1] if exps else None

    def dte(self, expiry: date) -> int:
        return (expiry - self.today).days

    def side(self, expiry: date, right: str) -> pd.DataFrame:
        df = self.df
        return df[(df["expiry"] == expiry) & (df["right"] == right)].sort_values("strike")

    def calls(self, expiry: date) -> pd.DataFrame:
        return self.side(expiry, "C")

    def puts(self, expiry: date) -> pd.DataFrame:
        return self.side(expiry, "P")

    def row(self, symbol: str) -> pd.Series | None:
        i = self._by_symbol.get(symbol)
        return None if i is None else self.df.iloc[i]

    def mark(self, symbol: str) -> float | None:
        r = self.row(symbol)
        return None if r is None else float(r["mark"])

    def nearest_strike(self, expiry: date, right: str, price: float, offset: int = 0) -> pd.Series | None:
        """The contract whose strike is closest to `price`; offset moves N strikes up (+) / down (-)."""
        side = self.side(expiry, right)
        if side.empty:
            return None
        strikes = side["strike"].to_numpy()
        i = int(np.argmin(np.abs(strikes - price))) + offset
        return side.iloc[i] if 0 <= i < len(side) else None

    def atm(self, expiry: date, right: str, offset: int = 0) -> pd.Series | None:
        return self.nearest_strike(expiry, right, self.spot, offset)

    def nearest_delta(self, expiry: date, right: str, delta: float) -> pd.Series | None:
        """Contract with |delta| closest to abs(delta) (e.g. 0.30)."""
        side = self.side(expiry, right).dropna(subset=["delta"])
        if side.empty:
            return None
        return side.iloc[int(np.argmin(np.abs(side["delta"].abs().to_numpy() - abs(delta))))]

    def cheapest_above(self, expiry: date, right: str, min_price: float) -> pd.Series | None:
        """The furthest out-of-the-money contract still priced at least `min_price` (lotto tickets)."""
        side = self.side(expiry, right)
        side = side[side["mark"] >= min_price]
        if side.empty:
            return None
        return side.iloc[-1] if right == "C" else side.iloc[0]

    def expected_move(self, expiry: date) -> float | None:
        """ATM straddle mark: the market's expected +-move to that expiry, in dollars."""
        c, p = self.atm(expiry, "C"), self.atm(expiry, "P")
        if c is None or p is None:
            return None
        return float(c["mark"] + p["mark"])

    def volume_ratio(self, expiry: date | None = None) -> float:
        """Call volume / put volume (today, cumulative per CBOE)."""
        df = self.df if expiry is None else self.df[self.df["expiry"] == expiry]
        calls = df.loc[df["right"] == "C", "volume"].sum()
        puts = df.loc[df["right"] == "P", "volume"].sum()
        return float(calls / puts) if puts else float("inf") if calls else 1.0


# ----- structure builders (return lists of (symbol, ratio); wrap in Leg in the strategy) -----

def _sym(r) -> str:
    return str(r["symbol"])


def single(row, long: bool = True) -> list[tuple[str, int]]:
    return [(_sym(row), 1 if long else -1)]


def vertical(chain: Chain, expiry: date, right: str, long_strike: float,
             short_strike: float) -> list[tuple[str, int]] | None:
    """Long one strike, short another (debit if the long leg is nearer the money)."""
    a = chain.nearest_strike(expiry, right, long_strike)
    b = chain.nearest_strike(expiry, right, short_strike)
    if a is None or b is None or a["symbol"] == b["symbol"]:
        return None
    return [(_sym(a), 1), (_sym(b), -1)]


def straddle(chain: Chain, expiry: date, strike: float | None = None, long: bool = True):
    k = chain.spot if strike is None else strike
    c, p = chain.nearest_strike(expiry, "C", k), chain.nearest_strike(expiry, "P", k)
    if c is None or p is None:
        return None
    s = 1 if long else -1
    return [(_sym(c), s), (_sym(p), s)]


def strangle(chain: Chain, expiry: date, delta: float = 0.25, long: bool = True):
    c, p = chain.nearest_delta(expiry, "C", delta), chain.nearest_delta(expiry, "P", delta)
    if c is None or p is None:
        return None
    s = 1 if long else -1
    return [(_sym(c), s), (_sym(p), s)]


def iron_condor(chain: Chain, expiry: date, short_delta: float = 0.10, wing: float = 5.0):
    """Short a call and a put at ~short_delta, long wings `wing` dollars further out (defined risk)."""
    sc, sp = chain.nearest_delta(expiry, "C", short_delta), chain.nearest_delta(expiry, "P", short_delta)
    if sc is None or sp is None:
        return None
    lc = chain.nearest_strike(expiry, "C", float(sc["strike"]) + wing)
    lp = chain.nearest_strike(expiry, "P", float(sp["strike"]) - wing)
    if lc is None or lp is None or lc["symbol"] == sc["symbol"] or lp["symbol"] == sp["symbol"]:
        return None
    return [(_sym(sc), -1), (_sym(lc), 1), (_sym(sp), -1), (_sym(lp), 1)]


def butterfly(chain: Chain, expiry: date, right: str, center: float, width: float):
    lo = chain.nearest_strike(expiry, right, center - width)
    mid = chain.nearest_strike(expiry, right, center)
    hi = chain.nearest_strike(expiry, right, center + width)
    if lo is None or mid is None or hi is None or len({lo["symbol"], mid["symbol"], hi["symbol"]}) < 3:
        return None
    return [(_sym(lo), 1), (_sym(mid), -2), (_sym(hi), 1)]


def net_mark(chain: Chain, legs) -> float | None:
    """Net value per unit, per share (+ = costs money to buy)."""
    total = 0.0
    for sym, ratio in legs:
        m = chain.mark(sym)
        if m is None:
            return None
        total += ratio * m
    return total


def units_for(budget: float, price_per_share: float) -> int:
    """Whole structure units that `budget` dollars buys at `price_per_share` net debit."""
    cost = abs(price_per_share) * MULTIPLIER
    return int(budget // cost) if cost > 0 else 0


# ----- underlying indicators -----

def realized_vol(closes: pd.Series, periods_per_year: float = 252.0, n: int | None = None) -> float:
    r = np.log(closes.astype(float)).diff().dropna()
    if n:
        r = r.iloc[-n:]
    return float(r.std() * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")


def opening_range(today: pd.DataFrame, minutes: int = 30, bar_minutes: int = 5) -> tuple[float, float] | None:
    n = max(minutes // bar_minutes, 1)
    if len(today) < n:
        return None
    first = today.iloc[:n]
    return float(first["high"].max()), float(first["low"].min())


def vwap(today: pd.DataFrame) -> float | None:
    if today.empty or today["volume"].sum() <= 0:
        return None
    typical = (today["high"] + today["low"] + today["close"]) / 3
    return float((typical * today["volume"]).sum() / today["volume"].sum())


def gap(daily: pd.DataFrame, today_open: float) -> float | None:
    """Today's open vs yesterday's close, as a fraction."""
    if daily is None or daily.empty:
        return None
    return today_open / float(daily["close"].iloc[-1]) - 1
