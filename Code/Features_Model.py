import Features_Cluster as FeaturesCluster
from Data import CryptoData
import numpy as np
import pandas as pd

class Features_Model(CryptoData):
    
    def __init__(self, symbol, exchange, timeframe='1h', candles=1000, window=500):
        super().__init__(symbol, exchange, timeframe, candles, window)

    def coin_handler(self):
        clusters = FeaturesCluster.Cluster_Symbols
        cluster_id = FeaturesCluster.Biggest_Cluster_ID
        return clusters[cluster_id]

    def closeprices(self):
        """
        Retrieve the close prices from the dataset for this symbol.
        """
        symbols = self.coin_handler()
        if self.symbol in symbols:
            self.fetchdata()
            return self.df['close']
        else:
            return None

def compute_log_spreads(exchange):
    if FeaturesCluster.Biggest_Cluster_ID is None or FeaturesCluster.Cluster_Symbols is None:
        raise ValueError("Clusters not initialized")
    
    spreads = {}
    symbols = FeaturesCluster.Cluster_Symbols[FeaturesCluster.Biggest_Cluster_ID]

    # Fetch all price data once and cache - this basically just makes it a lot faster for the nested for loop
    price_cache = {}
    for sym in symbols:
        model = Features_Model(sym, exchange)
        price_cache[sym] = model.closeprices()

    # Now compute pairs using cached price data
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym1 = symbols[i]
            sym2 = symbols[j]

            price1 = price_cache.get(sym1)
            price2 = price_cache.get(sym2)

            if price1 is None or price2 is None:
                continue

            df = pd.concat([price1, price2], axis=1).dropna()
            if df.shape[0] > 0:
                log_spread = np.log(df.iloc[:, 0] / df.iloc[:, 1])
                spreads[(sym1, sym2)] = log_spread

    print(spreads)
    return spreads
