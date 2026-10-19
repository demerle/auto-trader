"""In-memory broker for tests and simulations."""

from autotrader.broker.base import BrokerOrder, OrderType, TimeInForce
from autotrader.risk.limits import AssetInfo


class FakeBroker:
    def __init__(self, cash: float = 100_000.0, account: str = "paper"):
        self.account = account
        self.cash = cash
        self.orders: dict[str, BrokerOrder] = {}
        self.meta: dict[str, tuple[OrderType, TimeInForce, float | None]] = {}
        self.pos: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.assets: dict[str, AssetInfo] = {}
        self._n = 0

    def submit(self, client_order_id, symbol, qty, order_type, tif, limit_price=None):
        self._n += 1
        o = BrokerOrder(client_order_id, f"fake-{self._n}", symbol, qty, "accepted")
        self.orders[client_order_id] = o
        self.meta[client_order_id] = (order_type, tif, limit_price)
        return o

    def get_order(self, client_order_id):
        return self.orders[client_order_id]

    def cancel(self, client_order_id):
        o = self.orders[client_order_id]
        if not o.terminal:
            o.status = "canceled"

    def positions(self):
        return {s: q for s, q in self.pos.items() if q}

    def equity(self):
        return self.cash + sum(q * self.last.get(s, 0.0) for s, q in self.pos.items())

    def asset(self, symbol):
        return self.assets.get(symbol, AssetInfo())

    # --- simulation helpers ---
    def fill(self, client_order_id: str, price: float, qty: float | None = None) -> None:
        o = self.orders[client_order_id]
        remaining = abs(o.qty) - o.filled_qty
        q = remaining if qty is None else min(qty, remaining)
        if q <= 0 or o.terminal:
            return
        signed = q if o.qty > 0 else -q
        notional = (o.filled_avg_price or 0.0) * o.filled_qty + q * price
        o.filled_qty += q
        o.filled_avg_price = notional / o.filled_qty
        o.status = "filled" if o.filled_qty >= abs(o.qty) else "partially_filled"
        self.cash -= signed * price
        self.pos[o.symbol] = self.pos.get(o.symbol, 0.0) + signed
        self.last[o.symbol] = price

    def fill_all(self, prices: dict[str, float]) -> None:
        """Fill every working order whose symbol has a price (e.g. the next bar's open)."""
        for cid, o in list(self.orders.items()):
            if not o.terminal and o.symbol in prices:
                self.fill(cid, prices[o.symbol])
