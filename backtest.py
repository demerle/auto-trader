import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os
import chandelier_exit

load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "")
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)



SYMBOL = "SPY"
QTY = 10
alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL)


# BACKTEST STARTS HERE

#Getting alpaca bars for 5m intervals
df = alpaca.get_bars(SYMBOL, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute), "2025-09-18", "2025-09-18").df
df = df[['open', 'high', 'low', 'close', 'volume']]  # Clean columns

df.index = pd.to_datetime(df.index)
df = df.between_time("9:30", "16:00")

print(df)

#Giving the df to the algorithm of choice
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







