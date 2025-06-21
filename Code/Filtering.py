from Data import CryptoData
from Features import FeatureExtractor
import numpy as np
import pandas as pd
import ccxt
import time
import math
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

class Filtering(FeatureExtractor):
    def __init__(self, symbol, exchange, timeframe='1h', candles=1000, window=500):
        super().__init__(symbol, exchange, timeframe, candles, window)

    def needed_assets(self, cluster_symbols):
        # cluster_symbols is a list of symbols for one cluster
        if self.symbol in cluster_symbols:
            print(f"{self.symbol} found in cluster, fetching data...")
            self.fetchdata()
            return self.df
        else:
            print(f"{self.symbol} NOT found in cluster.")
            return None
    
    def pairs(self, symbols):
        pairs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pairs.append((symbols[i], symbols[j]))
        return pairs

    def spread(self, price1, price2):
        if price1 is None or price2 is None:
            return None
        df = pd.concat([np.log(price1), np.log(price2)], axis=1).dropna()
        df.columns = ['price1', 'price2']
        spread_series = df['price1'] - df['price2']
        spread_series = spread_series - spread_series.mean()  # center
        return spread_series


    #Here we are going to define the r value function to help
    def pearson(self, price1, price2):
        if price1 is None or price2 is None:
            return None
        log_price1 = np.log(price1)
        log_price2 = np.log(price2)
        df = pd.concat([log_price1, log_price2], axis=1).dropna()
        r = df.iloc[:, 0].corr(df.iloc[:, 1])
        return r

    # Define the function to calculate the spread from the log prices
    def spread(self, price1, price2):
        if price1 is None or price2 is None:
            return None
        df = pd.concat([np.log(price1), np.log(price2)], axis=1).dropna()
        df.columns = ['price1', 'price2']
        spread_series = df['price1'] - df['price2']
        spread_series = spread_series - spread_series.mean()  # center
        return spread_series

    # Here we define the variance function of the spread
    def spreadvariance(self, price1, price2):
        s = self.spread(price1, price2)
        if s is None:
            return None
        return s.var()
