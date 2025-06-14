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
from scipy.linalg import eigh

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

