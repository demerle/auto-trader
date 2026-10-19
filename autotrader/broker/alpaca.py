"""Alpaca trading adapter (orders and account only; no Alpaca market data is used)."""

from functools import lru_cache

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from autotrader.broker.base import BrokerOrder, OrderType, TimeInForce
from autotrader.config import get_settings
from autotrader.risk.limits import AssetInfo


def _to_order(o) -> BrokerOrder:
    qty = float(o.qty or 0)
    side = o.side.value if hasattr(o.side, "value") else str(o.side)
    return BrokerOrder(
        client_order_id=o.client_order_id,
        broker_order_id=str(o.id),
        symbol=o.symbol,
        qty=qty if side == "buy" else -qty,
        status=o.status.value if hasattr(o.status, "value") else str(o.status),
        filled_qty=float(o.filled_qty or 0),
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
    )


class AlpacaBroker:
    def __init__(self, key_id: str, secret_key: str, paper: bool = True):
        self.account = "paper" if paper else "live"
        self.client = TradingClient(key_id, secret_key, paper=paper)
        self._asset = lru_cache(maxsize=2048)(self._fetch_asset)

    @classmethod
    def paper_from_settings(cls) -> "AlpacaBroker":
        s = get_settings()
        if not (s.alpaca_paper_key_id and s.alpaca_paper_secret_key):
            raise RuntimeError("ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY are not set")
        return cls(s.alpaca_paper_key_id, s.alpaca_paper_secret_key, paper=True)

    def submit(self, client_order_id: str, symbol: str, qty: float, order_type: OrderType,
               tif: TimeInForce, limit_price: float | None = None) -> BrokerOrder:
        common = dict(
            symbol=symbol, qty=abs(qty), side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
            time_in_force=AlpacaTIF(tif), client_order_id=client_order_id,
        )
        if order_type == "limit":
            req = LimitOrderRequest(limit_price=round(limit_price, 2 if limit_price >= 1 else 4),
                                    **common)
        else:
            req = MarketOrderRequest(**common)
        return _to_order(self.client.submit_order(req))

    def get_order(self, client_order_id: str) -> BrokerOrder:
        return _to_order(self.client.get_order_by_client_id(client_order_id))

    def cancel(self, client_order_id: str) -> None:
        o = self.client.get_order_by_client_id(client_order_id)
        try:
            self.client.cancel_order_by_id(o.id)
        except APIError as exc:  # already filled/canceled
            if "422" not in str(exc) and "not cancelable" not in str(exc).lower():
                raise

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def _fetch_asset(self, symbol: str) -> AssetInfo:
        a = self.client.get_asset(symbol)
        return AssetInfo(tradable=a.tradable, shortable=a.shortable,
                         easy_to_borrow=a.easy_to_borrow)

    def asset(self, symbol: str) -> AssetInfo:
        return self._asset(symbol)
