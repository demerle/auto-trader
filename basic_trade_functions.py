from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os
load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "")
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

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



def closeAllOpenOrders(trade_client):

    orders = trade_client.get_orders()
    for order in orders:
        print("Hello")
        if not order.canceled_at:
            trade_client.cancel_order_by_id(order.id)
            print(f"Canceled order {order.id}")