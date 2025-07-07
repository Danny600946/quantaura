import Features_Cluster as FeaturesCluster
from Data import CryptoData
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

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

class Calculations_Model():

    def __init__(self, exchange):
        self.exchange = exchange
        self.spread = pd.DataFrame() 

    def compute_log_spreads(self):
        if FeaturesCluster.Biggest_Cluster_ID is None or FeaturesCluster.Cluster_Symbols is None:
            raise ValueError("Clusters not initialized")
        
        spreads = {}
        symbols = FeaturesCluster.Cluster_Symbols[FeaturesCluster.Biggest_Cluster_ID]

        # Fetch all price data once and cache - this basically just makes it a lot faster for the nested for loop stores it in a hash table
        price_cache = {}
        for sym in symbols:
            model = Features_Model(sym, self.exchange)
            price_cache[sym] = model.closeprices()

        # access and compute pairs using hash table price data
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
                    pair_name = f"{sym1}/{sym2}"
                    self.spread_df[pair_name] = log_spread
                    spreads[(sym1, sym2)] = log_spread
        
        return spreads

    def ADF_PTEST(self, alpha = 0.05):
        spreads = self.compute_log_spreads()
        ADF_Results = {}
        for pair, spread in spreads.items():
            result = adfuller(spread)
            p_value = result[1]
            if p_value < alpha:
                ADF_Results[pair] = p_value
                
        return ADF_Results
