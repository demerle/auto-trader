from dataclasses import dataclass
from typing import Literal, Protocol

from autotrader.risk.limits import AssetInfo

OrderType = Literal["market", "limit"]
TimeInForce = Literal["opg", "day"]
TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced", "stopped", "suspended",
            "done_for_day"}


@dataclass
class BrokerOrder:
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    qty: float  # signed
    status: str
    filled_qty: float = 0.0  # unsigned
    filled_avg_price: float | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


class Broker(Protocol):
    account: str  # "paper" or "live"

    def submit(self, client_order_id: str, symbol: str, qty: float, order_type: OrderType,
               tif: TimeInForce, limit_price: float | None = None) -> BrokerOrder: ...

    def get_order(self, client_order_id: str) -> BrokerOrder: ...

    def cancel(self, client_order_id: str) -> None: ...

    def positions(self) -> dict[str, float]: ...

    def equity(self) -> float: ...

    def asset(self, symbol: str) -> AssetInfo: ...
