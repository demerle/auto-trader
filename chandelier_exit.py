import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange




def chandelier_exit(df: pd.DataFrame):
    # -------------------------------
    # CONFIG
    # -------------------------------
    LENGTH = 22  # ATR period from Pine Script
    MULT = 3.0  # ATR multiplier
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