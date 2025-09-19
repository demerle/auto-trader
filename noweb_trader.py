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

def closeAllOpenOrders(trade_client):

    orders = trade_client.get_orders()
    for order in orders:
        print("Hello")
        if not order.canceled_at:
            trade_client.cancel_order_by_id(order.id)
            print(f"Canceled order {order.id}")




load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "")
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)




#--------------------------------------


import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
import alpaca_trade_api as tradeapi
import os

# -------------------------------
# CONFIG
# -------------------------------
LENGTH = 22          # ATR period from Pine Script
MULT = 3.0           # ATR multiplier
SYMBOL = "SPY"
QTY = 10             # number of shares per trade

# Alpaca setup
alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL)

# -------------------------------
# FUNCTION TO GENERATE SIGNALS
# -------------------------------
def chandelier_exit(df: pd.DataFrame):
    """
    df: DataFrame with 5m OHLCV bars ['open','high','low','close','volume']
    Returns: buy/sell signals as indices
    """
    # --- ATR
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=LENGTH).average_true_range() * MULT

    # --- Long and Short Stops
    long_stop = df['high'].rolling(LENGTH).max() - atr
    short_stop = df['low'].rolling(LENGTH).min() + atr

    # --- Initialize direction
    dir = np.ones(len(df))  # 1=long, -1=short
    for i in range(1, len(df)):
        if df['close'].iloc[i] > short_stop.iloc[i-1]:
            dir[i] = 1
        elif df['close'].iloc[i] < long_stop.iloc[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]

    # --- Signals
    buy_signal = (dir[1:] == 1) & (dir[:-1] == -1)
    sell_signal = (dir[1:] == -1) & (dir[:-1] == 1)

    buy_idx = np.where(buy_signal)[0] + 1
    sell_idx = np.where(sell_signal)[0] + 1

    return buy_idx, sell_idx, dir


# Attempting to backtest with the algorithm


df = alpaca.get_bars(SYMBOL, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute), "2025-09-18", "2025-09-18").df
df = df[['open', 'high', 'low', 'close', 'volume']]  # Clean columns

df.index = pd.to_datetime(df.index)
df = df.between_time("9:30", "16:00")

print(df)


buy_idx, sell_idx, dir = chandelier_exit(df)

print("Buy signals at indices:", buy_idx)
print("Sell signals at indices:", sell_idx)

buy = False
sell = False
curr = 0
count = 0.0
for i in range(len(df)):
    if i in buy_idx:
        buy = True
        currBuy = df['close'].iloc[i]
        print("Bought at", currBuy)
    if i in sell_idx:
        if not buy:
            continue
        else:
            currSell = df['close'].iloc[i]
            print("Sold at", currSell)
            count += currSell - currBuy
            buy = False

startPrice = df['close'].iloc[0]
endPrice = startPrice+count
percentIncrease = ((endPrice / startPrice) - 1) * 100

print()
print("Daily Profit:", count)
print("Daily Percent Increase:", str(percentIncrease) + "%")







