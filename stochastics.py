
import pandas as pd
import pandas_ta as ta
import alpaca_trade_api as tradeapi
import datetime as dt


# -----------------------
# Calculate Stochastic Oscillator
# -----------------------
stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
df['stoch_k'] = stoch['STOCHK_14_3_3']
df['stoch_d'] = stoch['STOCHD_14_3_3']

# -----------------------
# Generate signals
# -----------------------
df['long_entry'] = (df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d'])
df['long_exit'] = (df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d'])

# -----------------------
# Backtesting
# -----------------------
balance = INITIAL_BALANCE
position = 0
entry_price = 0
trades = []

for i in range(1, len(df)):
    # Entry signal
    if df['long_entry'].iloc[i] and position == 0:
        position = balance / df['close'].iloc[i]
        entry_price = df['close'].iloc[i]
        balance = 0
        trades.append({'type': 'BUY', 'time': df['time'].iloc[i], 'price': entry_price})

    # Exit signal or stop/take-profit
    elif position > 0:
        current_price = df['close'].iloc[i]
        # Stop-loss
        if current_price <= entry_price * (1 - STOP_LOSS_PCT):
            balance = position * current_price
            trades.append({'type': 'SELL_STOP', 'time': df['time'].iloc[i], 'price': current_price})
            position = 0
        # Take-profit
        elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
            balance = position * current_price
            trades.append({'type': 'SELL_TP', 'time': df['time'].iloc[i], 'price': current_price})
            position = 0
        # Regular exit
        elif df['long_exit'].iloc[i]:
            balance = position * current_price
            trades.append({'type': 'SELL_EXIT', 'time': df['time'].iloc[i], 'price': current_price})
            position = 0

# Final portfolio value
final_balance = balance + (position * df['close'].iloc[-1] if position > 0 else 0)
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Final Balance: ${final_balance:,.2f}")
print(f"Net Profit: ${final_balance - INITIAL_BALANCE:,.2f}")

# -----------------------
# Calculate trade stats
# -----------------------
wins = []
losses = []
for i in range(1, len(trades), 2):  # Pair buy/sell
    buy = trades[i-1]
    sell = trades[i]
    profit_pct = (sell['price'] - buy['price']) / buy['price']
    if profit_pct > 0:
        wins.append(profit_pct)
    else:
        losses.append(profit_pct)

win_rate = len(wins) / max(1, len(wins) + len(losses))
avg_gain = sum(wins)/max(1,len(wins)) if wins else 0
avg_loss = abs(sum(losses)/max(1,len(losses))) if losses else 0
risk_reward = avg_gain / avg_loss if avg_loss > 0 else float('inf')

print(f"Total Trades: {len(wins) + len(losses)}")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Average Gain: {avg_gain*100:.2f}%")
print(f"Average Loss: {avg_loss*100:.2f}%")
print(f"Risk-Reward Ratio: {risk_reward:.2f}")