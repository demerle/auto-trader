"""Cash and signed positions with average cost; shared by backtests and live ledgers."""

import math
from collections.abc import Mapping


class Book:
    def __init__(self, cash: float):
        self.cash = cash
        self.qty: dict[str, float] = {}
        self.avg: dict[str, float] = {}

    def fill(self, symbol: str, qty: float, price: float, fees: float = 0.0) -> float | None:
        """Apply a fill; returns realized P&L (net of fees) if it reduced a position."""
        q = self.qty.get(symbol, 0.0)
        a = self.avg.get(symbol, 0.0)
        self.cash -= qty * price + fees
        realized = None
        new_q = q + qty
        if q == 0 or math.copysign(1, q) == math.copysign(1, qty):
            self.avg[symbol] = (q * a + qty * price) / new_q
        else:
            closed = min(abs(qty), abs(q))
            realized = closed * (price - a) * math.copysign(1, q) - fees
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

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        return {s: q * prices[s] / eq for s, q in self.qty.items()} if eq > 0 else {}
