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
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

class CryptoData: 
    """
    Handles cryptocurrency data acquisition, preprocessing, and feature extraction for asset universe analysis.

    Purpose:
        This class serves as the core handler for preparing cryptocurrency market data for further
        analysis such as clustering, PCA-based dimensionality reduction, and portfolio filtering.

    Key Responsibilities:
        - Asset-level OHLCV data downloading via ccxt (e.g., Binance).
        - Calculation of key statistical features: log returns, volatility, skewness, autocorrelation, etc.
        - Data standardization and rolling window smoothing.
        - Stationarity testing using the Augmented Dickey-Fuller test.
        - Structuring feature vectors for PCA and clustering pipelines.
        - Supporting asset selection workflows based on volume, volatility, and return behavior.

    Attributes:
        symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): The time interval for each OHLCV candle (default is '1h').
        candles (int): The number of past candlesticks to fetch (default is 1000).
        window (int): The rolling window size used for feature calculations (default is 500).
        df (pd.DataFrame or None): DataFrame storing the raw and derived historical market data.
        feature (dict): Dictionary for storing computed feature values to be used in PCA or clustering.
    """
    
    def __init__(self, symbol, timeframe='1h', candles=1000, window=500):
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
            ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.candles)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.df = df
        except Exception as e:
            print(f"Failed to fetch {self.symbol}: {e}")
            self.df = None

    def closeprices(self):
        """
        Retrieve the close prices from the dataset.

        Returns:
            pd.Series: A series containing the 'close' prices.
        """
        return self.df['close']
    
    
    def log_hourly_returns(self):
        """
        Calculate log returns between each hourly candle.

        This function computes the natural logarithmic returns from the 'close' prices to capture
        relative price changes between consecutive hourly intervals.

        Returns:
            pd.Series: A series of log hourly returns, with NaNs dropped.
        """
        self.df['Log Hourly Returns'] = np.log(self.df['close']) - np.log(self.df['close'].shift(1))
        return self.df['Log Hourly Returns'].dropna()
    
    def mean_hourly_function(self):
        """
        Compute the rolling mean of log hourly returns.

        This function calculates the rolling average of log hourly returns over a specified window,
        providing a smoothed view of recent average returns (e.g., over the last 500 time intervals).

        Returns:
            pd.Series: A series of rolling mean values for hourly returns, with NaNs dropped.
        """
        self.df['Mean Hourly Returns'] = self.df['Log Hourly Returns'].rolling(window=self.window).mean()
        return self.df['Mean Hourly Returns'].dropna()
    
    def volatility_returns(self):
        """
        Compute the rolling volatility (standard deviation) of log hourly returns.

        This function calculates the rolling standard deviation over a specified window to measure return variability,
        representing the volatility of the asset over time.

        Returns:
            pd.Series: A series of rolling volatility values, with NaNs dropped.
        """
        self.df['Volatility'] = self.df['Log Hourly Returns'].rolling(window=self.window).std()
        return self.df['Volatility'].dropna()
    
    def skewness(self):
        """
        Calculate the skewness of log hourly returns.

        Skewness measures the asymmetry of the return distribution. Positive skew indicates a longer right tail,
        while negative skew indicates a longer left tail. The formula used is:
            skewness = (1/n) * Σ [ ((x_i - mean) / std) ** 3 ]

        Returns:
            skew (float): The skewness value of the return distribution.
        """
        prices = self.log_hourly_returns().values
        mean = np.mean(prices)
        std = np.std(prices)
        skew = np.sum(((prices - mean) / std) ** 3) / len(prices)
        # Return 1 value for each coing.
        return skew 
    
    def autocorrelation(self): 
        """
        Compute the lag-1 autocorrelation of log hourly returns.

        This function measures the linear relationship between current and lagged returns, providing insight
        into momentum or mean-reverting behavior in the time series. Autocorrelation values range from:
            -1: strong negative correlation (likely reversal),
            0: no correlation (uncertainty),
            1: strong positive correlation (momentum).

        Returns:
            autocorrelation (float): The autocorrelation coefficient at lag 1.
        """
        # Lag period, we can test different values for this 
        lag = 1
        returns = self.log_hourly_returns()
        prices = returns.dropna()
        laggedprices = prices.shift(lag).dropna() 
        # Aligns the two variables in the table.
        prices, laggedprices = prices.align(laggedprices, join='inner') 
        
        mean = np.mean(prices)
        autocorrelation =  np.sum((prices - mean) * (laggedprices- mean)) /np.sum((prices - mean) ** 2)
        return autocorrelation
    
    def adftest(self):
        """
        Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in log hourly returns.

        This function calculates log hourly returns, applies the ADF test to determine if the time series
        is stationary, and returns a binary indicator based on the p-value.

        Returns:
            stationaryflag (int): 1 if the time series is stationary (p < 0.05), otherwise 0.
        """
        returns = self.log_hourly_returns()
        results = adfuller(returns)
        # Normalise this before using pca test, try a binary stationarity to help normalise.
        # Checks p values are below the hypothesis test level. 
        stationaryflag = int(results[1] < 0.05) 
        return stationaryflag
    
    def Arch_P_Test(self):
        returns = self.log_hourly_returns()
        garchtest = arch_model(returns, vol = 'Garch', p = 1 , q = 1, mean = 'Zero') # defining parameters for the garch test 
        garch = garchtest.fit(disp = 'off')
        results = garch.params # this is to quantitatively define the parameters for the garch test
        return results['alpha[1]'] + results['beta[1]'] # if these sum close to one then this means we have strong persistence, if not then bad volatility clustering
    
    def average_hourly_volume(self):
        volume = self.df['volume'].dropna()
        return volume.mean() # the mean is just the hourly average volume 1/n * sum of volume 
    
    def volume_volatility(self): # this literally just calculates the volume volatility as a function of the lookback period (the standard deviation)
        volume = self.df['volume'].rolling(window=self.window).std()
        return volume.dropna()

    def max_drawdown(self): #this needs to be a for loop because we always gotta adjust from peak to trough
        prices = self.df['close']
        max_price = prices.iloc[0] #this basically just makes max_price start at the very first digit in the prices dataframe
        max_drawdown = 0  
        for price in prices:
            if price > max_price:
                max_price = price
            drawdown = max_price - price
            if drawdown > max_drawdown:
                    max_drawdown = drawdown
        return max_drawdown

    

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

    def k_means_cluster(number_of_clusters: int, max_iterations: int, reduced_fvectors: np.ndarray, tolerance=1e-4):
        """
        Perform K-means clustering on a set of feature vectors.

        This function partitions the input feature vectors into a specified number of clusters by
        iteratively assigning points to the nearest centroid and updating centroids based on the 
        mean of assigned points. The algorithm stops when centroids converge (change less than a 
        given tolerance) or the maximum number of iterations is reached.

        Args:
            number_of_clusters (int): The number of clusters to form (K).
            max_iterations (int): Maximum number of iterations for the algorithm to run.
            reduced_fvectors (np.ndarray): 2D array of input feature vectors with shape (num_samples, num_features).
            tolerance (float, optional): Threshold for centroid movement to determine convergence. Defaults to 1e-4.

        Returns:
            assignments (np.ndarray): 1D array of shape (num_samples,) indicating the index of the assigned cluster for each point.
            centroids (np.ndarray): 2D array of final centroid positions with shape (number_of_clusters, num_features).
        """

        # Selects initial centorids from reduced fvectors
        random_reduce_fvectors = np.random.choice(reduced_fvectors.shape[0], size=number_of_clusters, replace=False)
        centroids = reduced_fvectors[random_reduce_fvectors]

        # Loops until convergence of clusters or max iterations
        for iteration in range(max_iterations):
            # Calcs distances pairwise (Each point to each centroid)
            distances = cdist(reduced_fvectors, centroids, metric='euclidean')
            # Finds the index of smallest distance and stores the cluster number.
            assignments = np.argmin(distances, axis=1)

            # Calcs mean for each cluster, assigns value as new centroid.
            # If a cluster has no points, old centroid value is used. 
            new_centroids = np.array([
                reduced_fvectors[assignments == cluster_number].mean(axis=0)
                if np.any(assignments == cluster_number) else centroids[cluster_number]
                for cluster_number in range(number_of_clusters)
            ])

            # Check for convergence based on threshold value.
            if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < tolerance):
                print(f"Converged at iteration {iteration}")
                break
            # Updates centroids.
            centroids = new_centroids

        return assignments, centroids

exchange = ccxt.binance()
markets = exchange.load_markets()
