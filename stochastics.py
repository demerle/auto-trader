#!/usr/bin/env python3
"""
Stochastics and Momentum Indicators from Yahoo Finance.

This module provides functions to query Yahoo Finance for stock price data (both historical
and real-time intraday data) and calculate Fast Stochastic, Slow Stochastic, and Momentum.
It supports calculating indicators as of the current latest data (including midday market-open data)
or as of a specified historical date/timestamp.
"""

import datetime
import json
import sys
import urllib.request


def parse_target_date_or_timestamp(val):
    """
    Parses a target date or timestamp from various input types.
    
    Supported types:
    - None
    - int or float (treated as Unix epoch timestamp)
    - datetime.datetime or datetime.date object
    - str (e.g. 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', or a numeric string)
    
    Returns a datetime.datetime object in UTC or None.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc)
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=datetime.timezone.utc)
        return val
    if isinstance(val, datetime.date):
        return datetime.datetime.combine(val, datetime.time.min, tzinfo=datetime.timezone.utc)
    if isinstance(val, str):
        val_str = val.strip()
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.datetime.strptime(val_str, fmt)
                return dt.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                pass
        # Check if numeric string representing unix timestamp
        try:
            return datetime.datetime.fromtimestamp(float(val_str), tz=datetime.timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Could not parse timestamp/date value: {val}")


def get_data(ticker, data_range='60d', interval='1d'):
    """
    Queries Yahoo Finance chart endpoint for historical and real-time intraday price data.
    
    Cleans any missing/null data points and returns a dictionary of lists:
    {
        'meta': {...},
        'timestamp': [...],
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]
    }
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={data_range}&interval={interval}"
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
    except Exception as e:
        # Fallback to query2
        url2 = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range={data_range}&interval={interval}"
        req2 = urllib.request.Request(
            url2,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
            }
        )
        try:
            with urllib.request.urlopen(req2) as response:
                res_data = json.loads(response.read().decode())
        except Exception as e2:
            raise RuntimeError(f"Failed to fetch data for ticker '{ticker}' from Yahoo Finance: {e2}") from e2

    if not res_data.get('chart') or not res_data['chart'].get('result'):
        error_msg = "Unknown error"
        if res_data.get('chart') and res_data['chart'].get('error'):
            error_msg = res_data['chart']['error'].get('description', error_msg)
        raise ValueError(f"No chart data returned for '{ticker}': {error_msg}")

    result = res_data['chart']['result'][0]
    meta = result.get('meta', {})
    timestamps = result.get('timestamp', [])
    indicators = result.get('indicators', {})
    
    if not timestamps:
        raise ValueError(f"No price timestamps found for ticker '{ticker}'.")

    quote = indicators.get('quote', [{}])[0]
    opens = quote.get('open', [])
    highs = quote.get('high', [])
    lows = quote.get('low', [])
    closes = quote.get('close', [])
    volumes = quote.get('volume', [])
    
    if not closes or not highs or not lows:
        raise ValueError(f"No price quote data (close/high/low) found for ticker '{ticker}'.")
        
    cleaned_timestamps = []
    cleaned_opens = []
    cleaned_highs = []
    cleaned_lows = []
    cleaned_closes = []
    cleaned_volumes = []
    
    for i in range(len(timestamps)):
        if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes):
            break
            
        t = timestamps[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        v = volumes[i] if (volumes and i < len(volumes)) else 0
        
        # Filter out rows with null values in essential fields
        if t is None or o is None or h is None or l is None or c is None:
            continue
            
        cleaned_timestamps.append(t)
        cleaned_opens.append(float(o))
        cleaned_highs.append(float(h))
        cleaned_lows.append(float(l))
        cleaned_closes.append(float(c))
        cleaned_volumes.append(int(v) if v is not None else 0)

    if not cleaned_timestamps:
        raise ValueError(f"All returned price data for '{ticker}' contained null/invalid values.")

    return {
        'meta': meta,
        'timestamp': cleaned_timestamps,
        'open': cleaned_opens,
        'high': cleaned_highs,
        'low': cleaned_lows,
        'close': cleaned_closes,
        'volume': cleaned_volumes
    }


def get_slice_index(data, current_data=True, timestamp=None):
    """
    Determines the slice end index (exclusive) for the dataset based on current_data/timestamp.
    
    If current_data is True (and timestamp is None), returns the full dataset length (includes midday data).
    If timestamp is provided, parses and restricts data to points <= target timestamp.
    """
    timestamps = data.get('timestamp', [])
    if not timestamps:
        return 0
    
    if current_data and timestamp is None:
        return len(timestamps)
        
    if timestamp is None:
        return len(timestamps)

    target = parse_target_date_or_timestamp(timestamp)
    gmtoffset = data.get('meta', {}).get('gmtoffset', 0)
    
    # Determine if input was a date-only (or treated as date-only) query
    is_date_only = False
    if isinstance(timestamp, datetime.date) and not isinstance(timestamp, datetime.datetime):
        is_date_only = True
    elif isinstance(timestamp, str):
        if ':' not in timestamp and 'T' not in timestamp:
            is_date_only = True

    target_date = None
    if is_date_only:
        if isinstance(timestamp, datetime.date):
            target_date = timestamp
        else:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                try:
                    target_date = datetime.datetime.strptime(timestamp.strip(), fmt).date()
                    break
                except ValueError:
                    pass
            if target_date is None:
                target_date = target.date()

    last_idx = -1
    for idx, ts in enumerate(timestamps):
        # Convert timestamp to exchange's local time
        bar_dt = datetime.datetime.fromtimestamp(ts + gmtoffset, tz=datetime.timezone.utc)
        if is_date_only:
            if bar_dt.date() <= target_date:
                last_idx = idx
        else:
            if bar_dt <= target:
                last_idx = idx
                
    if last_idx == -1:
        raise ValueError(f"No data available on or before the specified timestamp: {timestamp}")
    
    return last_idx + 1


def get_fast_stochastic(ticker_or_data, current_data=True, timestamp=None, k_period=14, d_period=3):
    """
    Calculates the Fast Stochastic Oscillator (%K and %D) for the specified ticker or pre-fetched data.
    
    - Fast %K = ((Close - Low_N) / (High_N - Low_N)) * 100
    - Fast %D = SMA of Fast %K over d_period
    
    Returns a dict with:
        'fast_k': float
        'fast_d': float
        'timestamp': int (unix timestamp of the matched bar)
        'date': str (YYYY-MM-DD of the matched bar in exchange timezone)
    """
    if isinstance(ticker_or_data, str):
        data = get_data(ticker_or_data, data_range='60d', interval='1d')
    else:
        data = ticker_or_data

    slice_idx = get_slice_index(data, current_data, timestamp)
    
    highs = data['high'][:slice_idx]
    lows = data['low'][:slice_idx]
    closes = data['close'][:slice_idx]
    timestamps = data['timestamp'][:slice_idx]
    gmtoffset = data.get('meta', {}).get('gmtoffset', 0)
    
    n_bars = len(closes)
    required_bars = k_period + d_period - 1
    if n_bars < required_bars:
        raise ValueError(
            f"Not enough data points to compute Fast Stochastic. "
            f"Need at least {required_bars} bars, but only have {n_bars} after filtering/slicing."
        )

    # Compute Fast %K for the last d_period bars to calculate Fast %D (SMA)
    fast_k_values = []
    for i in range(n_bars - d_period, n_bars):
        window_highs = highs[i - k_period + 1 : i + 1]
        window_lows = lows[i - k_period + 1 : i + 1]
        
        highest_high = max(window_highs)
        lowest_low = min(window_lows)
        current_close = closes[i]
        
        denom = highest_high - lowest_low
        if denom == 0:
            k_val = 50.0  # standard fallback for flat price channels
        else:
            k_val = ((current_close - lowest_low) / denom) * 100
        fast_k_values.append(k_val)
        
    fast_d = sum(fast_k_values) / d_period
    fast_k = fast_k_values[-1]
    
    matched_ts = timestamps[-1]
    matched_date = datetime.datetime.fromtimestamp(matched_ts + gmtoffset, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
    
    return {
        'fast_k': fast_k,
        'fast_d': fast_d,
        'timestamp': matched_ts,
        'date': matched_date,
    }


def get_slow_stochastic(ticker_or_data, current_data=True, timestamp=None, k_period=14, d_period=3, slow_d_period=3):
    """
    Calculates the Slow Stochastic Oscillator (%K and %D) for the specified ticker or pre-fetched data.
    
    - Slow %K = Fast %D (SMA of Fast %K over d_period)
    - Slow %D = SMA of Slow %K over slow_d_period
    
    Returns a dict with:
        'slow_k': float
        'slow_d': float
        'timestamp': int (unix timestamp of the matched bar)
        'date': str (YYYY-MM-DD of the matched bar in exchange timezone)
    """
    if isinstance(ticker_or_data, str):
        data = get_data(ticker_or_data, data_range='60d', interval='1d')
    else:
        data = ticker_or_data

    slice_idx = get_slice_index(data, current_data, timestamp)
    
    highs = data['high'][:slice_idx]
    lows = data['low'][:slice_idx]
    closes = data['close'][:slice_idx]
    timestamps = data['timestamp'][:slice_idx]
    gmtoffset = data.get('meta', {}).get('gmtoffset', 0)
    
    n_bars = len(closes)
    required_fast_k_count = d_period + slow_d_period - 1
    required_bars = k_period + required_fast_k_count - 1
    if n_bars < required_bars:
        raise ValueError(
            f"Not enough data points to compute Slow Stochastic. "
            f"Need at least {required_bars} bars, but only have {n_bars} after filtering/slicing."
        )

    # Compute Fast %K for the last required_fast_k_count bars
    fast_k_values = []
    start_k_idx = n_bars - required_fast_k_count
    for i in range(start_k_idx, n_bars):
        window_highs = highs[i - k_period + 1 : i + 1]
        window_lows = lows[i - k_period + 1 : i + 1]
        
        highest_high = max(window_highs)
        lowest_low = min(window_lows)
        current_close = closes[i]
        
        denom = highest_high - lowest_low
        if denom == 0:
            k_val = 50.0
        else:
            k_val = ((current_close - lowest_low) / denom) * 100
        fast_k_values.append(k_val)
        
    # Calculate Slow %K (which is SMA of Fast %K)
    slow_k_values = []
    for j in range(slow_d_period):
        window_k = fast_k_values[j : j + d_period]
        slow_k = sum(window_k) / d_period
        slow_k_values.append(slow_k)
        
    slow_d = sum(slow_k_values) / slow_d_period
    slow_k = slow_k_values[-1]
    
    matched_ts = timestamps[-1]
    matched_date = datetime.datetime.fromtimestamp(matched_ts + gmtoffset, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
    
    return {
        'slow_k': slow_k,
        'slow_d': slow_d,
        'timestamp': matched_ts,
        'date': matched_date,
    }


def get_momentum(ticker_or_data, current_data=True, timestamp=None, period=10):
    """
    Calculates the Price Momentum (absolute and percentage) for the specified ticker or pre-fetched data.
    
    - Momentum = Close_t - Close_{t-period}
    - Momentum % (ROC) = ((Close_t - Close_{t-period}) / Close_{t-period}) * 100
    
    Returns a dict with:
        'momentum': float (absolute momentum)
        'momentum_pct': float (percentage change)
        'current_close': float
        'historical_close': float
        'timestamp': int (unix timestamp of the matched bar)
        'date': str (YYYY-MM-DD of the matched bar in exchange timezone)
    """
    if isinstance(ticker_or_data, str):
        data = get_data(ticker_or_data, data_range='60d', interval='1d')
    else:
        data = ticker_or_data

    slice_idx = get_slice_index(data, current_data, timestamp)
    
    closes = data['close'][:slice_idx]
    timestamps = data['timestamp'][:slice_idx]
    gmtoffset = data.get('meta', {}).get('gmtoffset', 0)
    
    n_bars = len(closes)
    required_bars = period + 1
    if n_bars < required_bars:
        raise ValueError(
            f"Not enough data points to compute Momentum of period {period}. "
            f"Need at least {required_bars} bars, but only have {n_bars} after filtering/slicing."
        )
        
    current_close = closes[-1]
    historical_close = closes[-1 - period]
    momentum = current_close - historical_close
    
    if historical_close != 0:
        momentum_pct = (momentum / historical_close) * 100
    else:
        momentum_pct = 0.0
        
    matched_ts = timestamps[-1]
    matched_date = datetime.datetime.fromtimestamp(matched_ts + gmtoffset, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
    
    return {
        'momentum': momentum,
        'momentum_pct': momentum_pct,
        'current_close': current_close,
        'historical_close': historical_close,
        'period': period,
        'timestamp': matched_ts,
        'date': matched_date,
    }


if __name__ == "__main__":
    # If file is executed directly, show a demonstration of the indicators.
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    print(f"=== Retrieving data for {ticker} ===")
    try:
        data = get_data("SPY", data_range="1y", interval="1d")
        print(f"Data retrieved successfully. Number of bars: {len(data['timestamp'])}")
        
        # Compute Latest (current midday / close)
        fast = get_fast_stochastic(data, k_period=21, d_period=5)
        slow = get_slow_stochastic(data, k_period=21, d_period=5, slow_d_period=5)
        mom  = get_momentum(data, period=21)
        
        print(f"\nLatest Data Point Date: {fast['date']} (Timestamp: {fast['timestamp']})")
        print(f"Current Price (Close): ${mom['current_close']:.2f}")
        print(f"Fast Stochastic: %K = {fast['fast_k']:.2f}%, %D = {fast['fast_d']:.2f}%")
        print(f"Slow Stochastic: %K = {slow['slow_k']:.2f}%, %D = {slow['slow_d']:.2f}%")
        print(f"Momentum (21-day): absolute = ${mom['momentum']:.2f}, percent = {mom['momentum_pct']:.2f}%")
        
        # Demo historical query (e.g. as of 1 week ago)
        # Find a date about 5 bars ago to query historically
        if len(data['timestamp']) > 5:
            hist_ts = data['timestamp'][-6]
            gmtoffset = data['meta'].get('gmtoffset', 0)
            hist_date_str = datetime.datetime.fromtimestamp(hist_ts + gmtoffset, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
            
            print(f"\n--- Querying Historically as of {hist_date_str} ---")
            hist_fast = get_fast_stochastic(data, current_data=False, timestamp=hist_date_str)
            hist_slow = get_slow_stochastic(data, current_data=False, timestamp=hist_date_str)
            hist_mom = get_momentum(data, current_data=False, timestamp=hist_date_str)
            
            print(f"Historical Close Price: ${hist_mom['current_close']:.2f}")
            print(f"Fast Stochastic: %K = {hist_fast['fast_k']:.2f}%, %D = {hist_fast['fast_d']:.2f}%")
            print(f"Slow Stochastic: %K = {hist_slow['slow_k']:.2f}%, %D = {hist_slow['slow_d']:.2f}%")
            print(f"Momentum (10-day): absolute = ${hist_mom['momentum']:.2f}, percent = {hist_mom['momentum_pct']:.2f}%")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
