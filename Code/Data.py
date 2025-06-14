"""
Purpose: Handles data acquisition, preprocessing, and asset universe setup.

Contents:
    - Asset selection (manual or filtered by volume/liquidity)

    - PCA/clustering for asset grouping

    - Historical price downloading (e.g., via yfinance, ccxt, etc.)

    - Standardization, missing value handling
"""
import numpy as np
import pandas as pd
import ccxt
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Here we are defining a function that just collects the data of the closes for our selected symbols
def close_prices(symbol, candles=1000, timeframe='15m'):  # 'days=365' put this in if I want to change to days
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')['close']
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")
        return None


