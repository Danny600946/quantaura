"""
Purpose: Core strategy logic for signals, position sizing, and execution logic.

    Contents:
        - Entry/exit conditions (ML + threshold-based logic)

        - Signal generation

        - Position sizing (confidence, volatility targeting)

        - Position management

    This is the main file we will run the singals with.
    
"""
import Code.Features_Cluster as Features_Cluster
import numpy as np
import pandas as pd
import ccxt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from arch import arch_model


if __name__ == '__main__':

    exchange = ccxt.binance()
    markets = exchange.load_markets()
    # Only getting the symbols with volume
    usdt_markets = {
    symbol: data for symbol, data in markets.items()
    if symbol.endswith('/USDT') and data.get('active') and data.get('quote') == 'USDT'
    }
    # Sort markets by 24h quote volume (descending) and pick top 100
    top_100_symbols = sorted(
        usdt_markets.keys(),
        key=lambda s: usdt_markets[s].get('info', {}).get('quoteVolume', 0),
        reverse=True
    )[:100]

    all_feature_vectors = []
    # Construct a feature vector for each symbol.
    for symbol in top_100_symbols:
        crypto = Features_Cluster.FeatureExtractor(symbol, exchange)
        crypto.fetchdata()
        features = crypto.feature_vector_builder()
        all_feature_vectors.append(features)

    # Filter out any feature vectors that contain NaNs or are invalid
    valid_vectors = [
        v for v in all_feature_vectors
        if isinstance(v, np.ndarray) and v.shape == (6,) and not np.isnan(v).any()
    ]

    fvectors = np.array(valid_vectors)

    print(f"Collected {len(fvectors)} valid feature vectors out of {len(all_feature_vectors)}")

    # Reduces and clusters.
    reduced_fvectors, loadings = Features_Cluster.pca_reduce(fvectors, n_components=2)

    wcss_values = np.empty(9)

    for i, k in enumerate(range (2,11)):
        assignments, centroids, distances = Features_Cluster.k_means_cluster(
            number_of_clusters=k,
            max_iterations=100,
            reduced_fvectors=reduced_fvectors
        )

        wcss_values[i] = Features_Cluster.calc_WCSS(distances, assignments)

    feature_names = [
        "half_life",
        "volatility_returns_mean",
        "adftest",
        "arch_p_test",
        "average_hourly_volume",
        "volume_volatility_mean"
    ]

    print("PCA Loadings (feature weights per principal component):")
    for i in range(loadings.shape[1]):
        print(f"\nPrincipal Component {i+1}:")
        for feature_name, loading in zip(feature_names, loadings[:, i]):
            print(f"  {feature_name}: {loading:.4f}")


    # Plot the clusters (2D)
    df = pd.DataFrame({
    'PC1': reduced_fvectors[:, 0],
    'PC2': reduced_fvectors[:, 1],
    'Cluster': assignments
    })
    #This method allows us for a legend
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab10', legend='full')
    plt.title("Asset Clustering via PCA + K-Means")
    plt.show()

    cluster = Features_Cluster.clusters(assignments,top_100_symbols)
    print(cluster)
    Features_Cluster.BTC_Cluster(cluster)