"""
Purpose: Feature engineering from raw data and spreads for ML models.

    Contents:
        - Spread Z-score, velocity

        - RSI, Bollinger Bands, volatility indicators

        - ARCH/GARCH-based volatility features

        - Regime indicators (e.g., regime label as categorical feature)
"""
from Data import CryptoData
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

#makes the features a child class of CryptoData
class FeatureExtractor(CryptoData):
        
    """
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

    #This is just taking the original initialisation from the CryptoData class based on the defintions
    def __init__(self, symbol, exchange, timeframe='1h', candles=1000, window=500):
        super().__init__(symbol, exchange, timeframe, candles, window)

    def closeprices(self):
        """
        Retrieve the close prices from the dataset.

        Returns:
            pd.Series: A series containing the 'close' prices.
        """
        self.fetchdata()
        return self.df
    
    
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
    
    def half_life(self):
        """
        Calculate the half-life of mean reversion for the price series in self.df['close'].
        This performs a regression on the demeaned series:
            Δy_t = β * y_{t-1} + errorand then calculates half-life as:
            half_life = -ln(2) / β

        Assumes equal time intervals (e.g., 1 hour, 1 day).
        """
        prices = self.df['close'].dropna()
        # Subtract mean from the series
        mu = prices.mean()
        y = prices - mu
        # Lagged series
        y_lag = y.shift(1).dropna()
        y_current = y.loc[y_lag.index]  # align indexes
        # Differences Δy_t = y_t - y_{t-1}
        delta_y = y_current - y_lag
        
        # Regression β = cov(Δy_t, y_{t-1}) / var(y_{t-1})
        cov = (delta_y * y_lag).mean() - delta_y.mean() * y_lag.mean()
        var = ((y_lag - y_lag.mean()) ** 2).mean()
        beta = cov / var
        
        # Calculate half-life
        halflife = -math.log(2) / beta
        
        return halflife
    
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
    
    def Arch_P_Test(self): #VOlatility Persistence
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
    
    def feature_vector_builder(self):
        """
        Build a feature vector for PCA and clustering from price data.

        Returns:
            np.ndarray: 1D array of features of fixed length 6.
        """
        try:
            self.log_hourly_returns()
            vector = [
                self.half_life(),
                self.volatility_returns().mean(),
                self.adftest(),
                self.Arch_P_Test(),
                self.average_hourly_volume(),
                self.volume_volatility().mean()
            ]
            vector = np.array(vector, dtype=float)

            # Check length to be safe
            if len(vector) != 6:
                raise ValueError("Feature vector has incorrect length")

        except Exception as e:
            print(f"Feature extraction failed for {self.symbol}: {e}")
            # Return NaNs if something fails
            vector = np.full(6, np.nan)

        return vector
    

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
    sorted_indices = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]

    # Project to reduced dimensions
    reduced_fvectors = np.dot(normalized_fvectors, eig_vec[:, :n_components])

    # Return both reduced vectors and loadings (components)
    return reduced_fvectors, eig_vec[:, :n_components]

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

    return assignments, centroids, distances

def calc_WCSS(distances, assignments):
    # Get the distance from each point to its assigned centroid
    assigned_distances = distances[np.arange(len(assignments)), assignments]

    # Square the distances
    squared_distances = assigned_distances ** 2

    # Sum them all for total WCSS
    wcss = np.sum(squared_distances)

    return wcss

def clusters(assignments,symbols):
    """
    Build a dictionary mapping cluster numbers to lists of coin symbols.

    Args:
        assignments (np.ndarray or list): cluster assignment for each coin (e.g., [0,1,2,0,1,...])
        symbols (list of str): list of coin symbols in the same order as assignments

    Returns:
        dict: keys = cluster numbers, values = list of symbols assigned to that cluster
    """
    clusters = {} 
    for i, cluster_num in enumerate(assignments):
        if cluster_num not in clusters:
            clusters[cluster_num] = []
        clusters[cluster_num].append(symbols[i])
    return clusters

def BTC_Cluster(clusters_dict):
    for cluster_id, assets in clusters_dict.items():
        if 'BTC/USDT' in assets:
            print(f"'BTC/USDT' is in cluster {cluster_id}")
            return cluster_id
    print("BTC/USDT not found in any cluster")
    return [] 
        