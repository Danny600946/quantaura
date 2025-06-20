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
import math
import requests
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

class CryptoData: 
    """
    Handles cryptocurrency data acquisition, preprocessing, and feature extraction for asset universe analysis.
    """
    
    def __init__(self, symbol, exchange,  timeframe='1h', candles=1000, window=500):
        """
        Initialize the CryptoData object with symbol, timeframe, candle count, and window size.

        Args:
            symbol (str): The trading symbol to fetch data for (e.g., 'BTC/USDT').
            timeframe (str, optional): Timeframe for each candlestick (default is '1h').
            candles (int, optional): Number of historical candles to retrieve (default is 1000).
            window (int, optional): Window size for rolling feature calculations (default is 500).
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles = candles
        self.window = window
        self.df = None  
        self.feature = {}
        self.exchange = exchange

   
    def fetchdata(self): 
        """
        Fetch historical OHLCV data for the selected symbol and store it as a DataFrame.

        This function retrieves candlestick data (Open, High, Low, Close, Volume) using the configured
        exchange, symbol, timeframe, and number of candles. It converts timestamps to datetime format
        and stores the result in `self.df`.

        On failure, it prints an error message and sets `self.df` to None.

        Returns:
            None
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.candles)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.df = df

            # Here we are getting the funding rate from the binance api
            symbol_for_api = self.symbol.replace('/', '')
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol_for_api}&limit=1000"
            resp = requests.get(url)
            data = resp.json()
            df_funding = pd.DataFrame(data)
            df_funding['fundingTime'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
            df_funding.set_index('fundingTime', inplace=True)
            df_funding['fundingRate'] = df_funding['fundingRate'].astype(float)

            # Forward fill funding rates on your OHLCV timestamps
            self.df['fundingRate'] = df_funding['fundingRate'].reindex(self.df.index, method='ffill')

        except Exception as e:
            print(f"Failed to fetch {self.symbol}: {e}")
            self.df = None
