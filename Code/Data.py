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
from scipy.linalg import eigh
#make this a class that we can call for a list of symbols 
class CryptoData: 
    
    def __init__(self, symbol, timeframe='1h', candles=1000, window=500):
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles = candles
        self.window = window
        self.df = None  
        self.feature = {} # here we are making a dictionary for all the features we want to use on the PCA matrix
    # here we are defining a function that just collects the data of the closes for our selected symbols
    def fetchdata(self): 
        try:
            ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.candles)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.df = df
        except Exception as e:
            print(f"Failed to fetch {self.symbol}: {e}")
            self.df = None
    def closeprices(self):
        return self.df['close']
    #returns between each hourly candle 
    def log_hourly_returns(self):
        self.df['Log Hourly Returns'] = np.log(self.df['close']) - np.log(self.df['close'].shift(1))
        return self.df['Log Hourly Returns'].dropna()
    def mean_hourly_function(self):
        self.df['Mean Hourly Returns'] = self.df['Log Hourly Returns'].rolling(window=self.window).mean() #this gets the mean hourly function so calculates an average return over last 500 candles for example
        return self.df['Mean Hourly Returns'].dropna()
    def volatility_returns(self):
        self.df['Volatility'] = self.df['Log Hourly Returns'].rolling(window=self.window).std()
        return self.df['Volatility'].dropna()
    def skewness(self): # this is the method defining skewness using the typical formula skewness = 1/n * Σ [ ((x_i - mean) / std) ** 3 ]
        prices = self.log_hourly_returns().values
        mean = np.mean(prices)
        std = np.std(prices)
        skew = np.sum(((prices - mean) / std) ** 3) / len(prices)
        return skew # should return 1 value for each coing
    def autocorrelation(self): # defining the autocorrelation function. Autocorrelation gives a value between -1 and 1: -1 means likely reversal, 1 means it’ll keep going, 0 means uncertainty. basically seeing how the returns at time = T differ to returns at time = T - k where k = lag
        lag = 1 #we have to define a lag period, we can test different values for this 
        prices = self.log_hourly_returns()
        prices = prices.dropna()
        laggedprices = prices.shift(lag).dropna() 
        prices, laggedprices = prices.align(laggedprices, join='inner') # making these two variables aligned in the table
        
        mean = np.mean(prices)
        autocorrelation =  np.sum((prices - mean) * (laggedprices- mean)) /np.sum((prices - mean) ** 2)
        return autocorrelation

    



def pca_reduce(fvectors: np.ndarray, n_components: int = 10):
    """
    Reduce the dimensionality of input feature vectors using Principal Component Analysis (PCA).

    This function standardizes the input data, computes the covariance matrix, performs eigenvalue
    decomposition, and projects the data onto the top principal components.

    Args:
        fvectors (np.ndarray): 2D array of feature vectors.
        n_components (int): Number of principal components to keep.

    Returns:
        reduced_fvectors (np.ndarray): The PCA-reduced feature vectors with shape (num_samples, n_components).
    """
    epsilon = 1e-8

    # Normalize the data
    mean = np.mean(fvectors, axis=0)
    std = np.std(fvectors, axis=0)
    normalized_fvectors = (fvectors - mean) / (std + epsilon)

    # Compute covariance matrix
    cov_matrix = np.cov(normalized_fvectors, rowvar=False)

    eig_val, eig_vec = eigh(cov_matrix)
    # Sort by descending eigenvalues
    eig_vec = eig_vec[:, np.argsort(eig_val)[::-1]] 

    # Project to reduced dimensions
    reduced_fvectors = np.dot(normalized_fvectors, eig_vec[:, :n_components])

    return reduced_fvectors

exchange = ccxt.binance()
markets = exchange.load_markets()
