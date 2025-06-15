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
