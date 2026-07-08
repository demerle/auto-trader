from services import get_stochastic_data, get_buy_sell_signals
# pyrefly: ignore [missing-import]
from returns.result import Success, Failure
# pyrefly: ignore [missing-import]

if __name__ == "__main__":
    list_of_tickers = ["SPY","QQQ","AAPL", "NVDA", "TSLA", "INTC", "GOOGL", "SMCI"]

    signal = get_buy_sell_signals(list_of_tickers)
    print(signal)
