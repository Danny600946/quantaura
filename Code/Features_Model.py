import Features_Cluster as FeaturesCluster
import numpy as np
import pandas as pd
import ccxt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

#Here we are going to generate all the pairs together
def coin_handler(df):
    clusters = FeaturesCluster.clusters()
    cluster_id = FeaturesCluster.Biggest_Cluster
    symbols = clusters[cluster_id]
    for i in range(len(symbols)):
        for j in range(i+1, symbols):
            sym1 = symbols[i]
            sym2 = symbols[j]

            price1 = df[sym1]
            price2 = df[sym2]

