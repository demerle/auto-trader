from stochastics import get_fast_stochastic, get_slow_stochastic, get_momentum, get_data
# pyrefly: ignore [missing-import]
from returns.result import Result, Success, Failure

def get_stochastic_data(ticker, data_range="1y", interval="1d") -> Result[tuple, Exception]:
    try:
        data = get_data(ticker, data_range=data_range, interval=interval)
        fast = get_fast_stochastic(data, k_period=21, d_period=5)
        slow = get_slow_stochastic(data, k_period=21, d_period=5, slow_d_period=5)
        mom  = get_momentum(data, period=21)
        return Success((fast, slow, mom))
    except Exception as e:
        return Failure(e)


def get_buy_sell_signals(list_of_tickers, data_range="1y", interval="1d") -> dict:
    signal = { 
        "buy" : [],
        "sell" : []
    }
    for ticker in list_of_tickers:
        res = get_stochastic_data(ticker, data_range=data_range, interval=interval)
        match res:
            case Success((fast, slow, mom)):
                if mom["momentum_pct"] > 0:
                    if fast["fast_k"] > 80 and slow["slow_k"] > 80 and slow["slow_d"] > 75:
                        signal["sell"].append(ticker)
                        print(f"{ticker} SELL SIGNAL: FastK = {fast['fast_k']:.2f}%, SlowK = {slow['slow_k']:.2f}%")
                else:
                    if fast["fast_k"] < 20 and slow["slow_k"] < 20 and slow["slow_d"] < 25:
                        signal["buy"].append(ticker)
                        print(f"{ticker} BUY SIGNAL: FastK = {fast['fast_k']:.2f}%, SlowK = {slow['slow_k']:.2f}%")
            case Failure(err):
                print(f"Error processing {ticker}: {err}")  
    return signal