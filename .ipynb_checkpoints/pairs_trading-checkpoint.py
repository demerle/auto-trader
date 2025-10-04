import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os
from chandelier_exit import chandelier_exit

load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "")
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)


SYMBOL = "SPY"
QTY = 10
alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL)


# BACKTEST STARTS HERE
def getTestableDataFrameOfSingleStock(symbol=SYMBOL, startDate="2025-09-01", endDate="2025-09-30") -> pd.DataFrame:

    #Getting alpaca bars for 5m intervals
    df = alpaca.get_bars(symbol, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute), startDate, endDate).df
    df = df[['open', 'high', 'low', 'close', 'volume']]  # Clean columns

    df.index = pd.to_datetime(df.index)
    df = df.between_time("9:30", "16:00") #Discluding after-hours data

    return df

df1 = getTestableDataFrameOfSingleStock()
df2 = getTestableDataFrameOfSingleStock(symbol="IVV")



d={}
d['IVV'] = 0
d['SPY'] = 0
#print("len1:", len(df1))
#print("len2:", len(df2))
#print(df1)
#print(df2)
j = 0

print("type:", type(df1['close'].iloc[0]))

for i in range(len(df2)):

    while j < len(df1) and df1.index[j] != df2.index[i]:
        print("\ndf1 time:", df1.index[j])
        print("df2 time:", df2.index[i])
        print("\nNot equal\n")
        j+=1

    if j >= len(df1) or i >= len(df1):
        break
    if df1['close'].iloc[j] > df2['close'].iloc[i]:
        d['SPY'] += 1
    elif df1['close'].iloc[j] == df2['close'].iloc[i]:
        continue
    else:
        d['IVV'] += 1
    j+=1

print(d)


