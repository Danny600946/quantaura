import Features_Cluster as FeaturesCluster
from Data import CryptoData
import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller
from arch import arch_model 


class Features_Model(CryptoData):
    
    def __init__(self, symbol, exchange, timeframe='1h', candles=1000, window=500):
        super().__init__(symbol, exchange, timeframe, candles, window)
        self._data_fetched = False

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
            if self.df is None or self.df.empty:
                self.fetchdata()
            if self.df is not None and 'close' in self.df:
                return self.df['close']
        return None

class Calculations_Model():

    def __init__(self, exchange):
        self.exchange = exchange
        self.spread = pd.DataFrame() 
        self.cached_spreads = None

    def compute_log_spreads(self):
        if FeaturesCluster.Biggest_Cluster_ID is None or FeaturesCluster.Cluster_Symbols is None:
            raise ValueError("Clusters not initialized")
        if self.cached_spreads is not None:
            return self.cached_spreads
        spreads = {}
        symbols = FeaturesCluster.Cluster_Symbols[FeaturesCluster.Biggest_Cluster_ID]

        # Fetch all price data once and cache - this basically just makes it a lot faster for the nested for loop stores it in a hash table
        price_cache = {}
        for sym in symbols:
            model = Features_Model(sym, self.exchange)
            price_cache[sym] = model.closeprices()

        all_spreads = {}
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
                    all_spreads[pair_name] = log_spread
                    spreads[(sym1, sym2)] = log_spread
        
        self.spread = pd.DataFrame(all_spreads)
        self.cached_spreads = spreads
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
    
    def half_life(self):
        """
        Calculate the half-life of mean reversion for the price series in self.df['close'].
        This performs a regression on the demeaned series:
            Δy_t = β * y_{t-1} + errorand then calculates half-life as:
            half_life = -ln(2) / β

        Assumes equal time intervals (e.g., 1 hour, 1 day).
        """
        prices = self.compute_log_spreads()
        half_life_results = {}
        # Subtract mean from the series
        for pair, prices in prices.items():
            y = prices - prices.mean()
            # Lagged series
            y_lag = y.shift(1).dropna()
            y_current = y.loc[y_lag.index]  # align indexes
            # Differences Δy_t = y_t - y_{t-1}
            delta_y = y_current - y_lag
            
            # Regression β = cov(Δy_t, y_{t-1}) / var(y_{t-1})
            cov = (delta_y * y_lag).mean() - delta_y.mean() * y_lag.mean()
            var = ((y_lag - y_lag.mean()) ** 2).mean()
            beta = cov / var
            if var == 0:
                continue
            # Calculate half-life
            halflife = -math.log(2) / beta
            
            if halflife> 12:
                half_life_results[pair] = halflife

        return half_life_results
    
    def GARCH_Test(self): #VOlatility Persistence
        returns = self.compute_log_spreads()
        GARCH_Results = {}
        for pair, returns in returns.items():
            garchtest = arch_model(returns, vol = 'Garch', p = 1 , q = 1, mean = 'Zero') # defining parameters for the garch test 
            garch = garchtest.fit(disp = 'off')
            results = garch.params # this is to quantitatively define the parameters for the garch test
            persistence = results['alpha[1]'] + results['beta[1]'] # if these sum close to one then this means we have strong persistence, if not then bad volatility clustering
            if persistence > 0.9:
                GARCH_Results[pair] = persistence
        return GARCH_Results
    
    #Just calculating the hourly volume for the pairs and we are finding ones that fit in our pairs
    def Average_Hourly_Volume(self):
        volume_results = {}
        symbols = FeaturesCluster.Cluster_Symbols[FeaturesCluster.Biggest_Cluster_ID]
        
        for sym in symbols:
            model = Features_Model(sym, self.exchange)
            model.fetchdata()
            vol = model.df['volume'].dropna()
            vol_mean = vol.mean()
            if vol_mean > 1_000_000:
                volume_results[sym] = vol_mean
                
        return volume_results
    
    def filtering_pairs(self, spreads):
        adf_results = self.ADF_PTEST()
        garch_results = self.GARCH_Test()
        half_life_results = self.half_life()
        volume_results = self.Average_Hourly_Volume()
        best_pairs = []
        for pair in spreads.keys():
        # Check that pair exists in all result dicts
            if (pair in adf_results and 
                pair in garch_results and 
                pair in half_life_results):
                # For volume, it's by symbol, so check both symbols' volume
                sym1, sym2 = pair
                if sym1 in volume_results and sym2 in volume_results:
                    best_pairs.append(pair)

        return best_pairs        
