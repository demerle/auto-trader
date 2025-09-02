from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import os


def purchaseMarketOrder(symbol="SPY", qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY):
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=time_in_force
                        )

    market_order = trading_client.submit_order(
                    order_data=market_order_data
                   )

    print("Order submitted:", market_order)

def closeAllOpenOrders():

    orders = trading_client.get_orders()
    for order in orders:
        print("Hello")
        if not order.canceled_at:
            trading_client.cancel_order_by_id(order.id)
            print(f"Canceled order {order.id}")




load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

#closeAllOpenOrders()