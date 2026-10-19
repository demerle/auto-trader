"""Alpaca trading adapter (orders and account only; no Alpaca market data is used)."""

from functools import lru_cache

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide, PositionIntent
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import (
    GetOrderByIdRequest, LimitOrderRequest, MarketOrderRequest, OptionLegRequest,
)

from autotrader.broker.base import BrokerOrder, OrderType, TimeInForce
from autotrader.config import get_settings
from autotrader.risk.limits import AssetInfo


def _val(x) -> str:
    return x.value if hasattr(x, "value") else str(x)


def _to_order(o) -> BrokerOrder:
    qty = float(o.qty or 0)
    side = _val(o.side) if o.side is not None else "buy"
    legs = None
    if o.legs:
        legs = [{"symbol": leg.symbol, "side": _val(leg.side) if leg.side is not None else None,
                 "ratio_qty": float(leg.ratio_qty or 1), "filled_qty": float(leg.filled_qty or 0),
                 "filled_avg_price": float(leg.filled_avg_price) if leg.filled_avg_price else None}
                for leg in o.legs]
    return BrokerOrder(
        client_order_id=o.client_order_id,
        broker_order_id=str(o.id),
        symbol=o.symbol or "",
        qty=qty if side == "buy" else -qty,
        status=_val(o.status),
        filled_qty=float(o.filled_qty or 0),
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        legs=legs,
    )


class AlpacaBroker:
    def __init__(self, key_id: str, secret_key: str, paper: bool = True, account: str | None = None):
        self.account = account or ("paper" if paper else "live")
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
        o = self.client.get_order_by_client_id(client_order_id)
        if _val(o.order_class) == "mleg" and not o.legs:
            o = self.client.get_order_by_id(o.id, GetOrderByIdRequest(nested=True))
        return _to_order(o)

    # ----- options (the gauntlet's account) -----
    @classmethod
    def options_from_settings(cls, account: str = "paper-options") -> "AlpacaBroker":
        s = get_settings()
        if not (s.alpaca_options_key_id and s.alpaca_options_secret_key):
            raise RuntimeError("ALPACA_OPTIONS_KEY_ID / ALPACA_OPTIONS_SECRET_KEY are not set")
        return cls(s.alpaca_options_key_id, s.alpaca_options_secret_key, paper=True, account=account)

    def submit_option(self, client_order_id: str, legs: list[tuple[str, int, str]], qty: int,
                      limit_price: float | None) -> BrokerOrder:
        """legs: (symbol, signed ratio, intent "open"/"close"); qty: structure units (> 0).

        One leg: a plain option order of qty x |ratio| contracts. Several: an mleg order
        whose limit is the net price per unit (+ debit, - credit; see mleg_price below).
        """
        tif = AlpacaTIF.DAY
        if len(legs) == 1:
            sym, ratio, intent = legs[0]
            side = OrderSide.BUY if ratio > 0 else OrderSide.SELL
            common = dict(symbol=sym, qty=abs(ratio) * qty, side=side, time_in_force=tif,
                          client_order_id=client_order_id, position_intent=_intent(ratio, intent))
            req = (MarketOrderRequest(**common) if limit_price is None else
                   LimitOrderRequest(limit_price=option_tick(limit_price, sym), **common))
            return _to_order(self.client.submit_order(req))
        leg_reqs = [OptionLegRequest(symbol=sym, ratio_qty=abs(ratio),
                                     side=OrderSide.BUY if ratio > 0 else OrderSide.SELL,
                                     position_intent=_intent(ratio, intent))
                    for sym, ratio, intent in legs]
        common = dict(qty=qty, order_class=OrderClass.MLEG, legs=leg_reqs, time_in_force=tif,
                      client_order_id=client_order_id)
        req = (MarketOrderRequest(**common) if limit_price is None else
               LimitOrderRequest(limit_price=mleg_price(limit_price), **common))
        return _to_order(self.client.submit_order(req))

    def option_positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()
                if _val(p.asset_class) == AssetClass.US_OPTION.value}

    def stock_positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()
                if _val(p.asset_class) != AssetClass.US_OPTION.value}

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

    def cancel_all(self) -> None:
        self.client.cancel_orders()

    def _fetch_asset(self, symbol: str) -> AssetInfo:
        a = self.client.get_asset(symbol)
        return AssetInfo(tradable=a.tradable, shortable=a.shortable,
                         easy_to_borrow=a.easy_to_borrow)

    def asset(self, symbol: str) -> AssetInfo:
        return self._asset(symbol)


def _intent(ratio: int, intent: str) -> PositionIntent:
    if ratio > 0:
        return PositionIntent.BUY_TO_OPEN if intent == "open" else PositionIntent.BUY_TO_CLOSE
    return PositionIntent.SELL_TO_OPEN if intent == "open" else PositionIntent.SELL_TO_CLOSE


def option_tick(price: float, symbol: str = "") -> float:
    """Penny increments below $3, nickels above for non-penny names (SPY/QQQ/IWM are all pennies)."""
    price = abs(price)
    if price >= 3 and not symbol.startswith(("SPY", "QQQ", "IWM")):
        return max(round(round(price / 0.05) * 0.05, 2), 0.05)
    return max(round(price, 2), 0.01)


def mleg_price(net: float) -> float:
    """Alpaca's mleg limit: + debit / - credit (MLEG_CREDIT_NEGATIVE; verified on paper)."""
    p = round(net, 2)
    if MLEG_CREDIT_NEGATIVE:
        return p if p != 0 else 0.01
    return abs(p) if p != 0 else 0.01


MLEG_CREDIT_NEGATIVE = True
