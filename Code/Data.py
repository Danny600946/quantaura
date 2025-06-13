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

#Here we are going to define the r value function to help
def pearson(price1, price2):
    if price1 is None or price2 is None:
        return None
    log_price1 = np.log(price1)
    log_price2 = np.log(price2)
    df = pd.concat([log_price1, log_price2], axis=1).dropna()
    r = df.iloc[:, 0].corr(df.iloc[:, 1])
    return r

# Define the function to calculate the spread from the log prices
def spread(price1, price2):
    if price1 is None or price2 is None:
        return None
    df = pd.concat([np.log(price1), np.log(price2)], axis=1).dropna()
    df.columns = ['price1', 'price2']
    spread_series = df['price1'] - df['price2']
    spread_series = spread_series - spread_series.mean()  # center
    return spread_series

# Here we define the variance function of the spread
def spreadvariance(price1, price2):
    s = spread(price1, price2)
    if s is None:
        return None
    return s.var()

# Johansen test to check for cointegration
def johansen_test(price1, price2, det_order=0, k_ar_diff=1):
    # Combine log prices into a single dataframe
    data = pd.concat([np.log(price1), np.log(price2)], axis=1).dropna()
    data.columns = ['price1', 'price2']

    # Run Johansen cointegration test
    result = coint_johansen(data, det_order, k_ar_diff)

    # result.lr1 contains trace statistics
    # result.cvt contains critical values (90%, 95%, 99%) for trace statistics

    print("Trace Statistic:", result.lr1)
    print("Critical Values (90%, 95%, 99%):")
    print(result.cvt)

    # Interpretation:
    # Compare trace statistic to critical values to decide how many cointegrating relationships exist
    # For 2 series, 1 cointegrating vector means cointegration exists

    return result, data

# Function to plot the cointegrated spread using the Johansen cointegrating vector
def plot_multiple_cointegrated_spreads(top_pairs, all_prices_dict):
    window = 30 * 24  # 720 for 30 days of hourly data

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # no sharing

    for idx, pair in enumerate(top_pairs[:4]):
        sym1, sym2 = pair.split(' vs ')
        price1 = all_prices_dict[sym1]
        price2 = all_prices_dict[sym2]

        if price1 is None or price2 is None:
            continue

        result, data = johansen_test(price1, price2)

        coint_vec = result.evec[:, 0]
        spread_series = data.values @ coint_vec
        spread_series = pd.Series(spread_series, index=data.index)

        rolling_mean = spread_series.rolling(window=window).mean()
        rolling_std = spread_series.rolling(window=window).std()

        # Find first valid rolling mean index
        first_valid_idx = rolling_mean.first_valid_index()

        # Slice all series from first valid rolling mean index onward
        spread_series_plot = spread_series.loc[first_valid_idx:]
        rolling_mean_plot = rolling_mean.loc[first_valid_idx:]
        rolling_std_plot = rolling_std.loc[first_valid_idx:]

        ax = axes[idx]
        ax.plot(spread_series_plot.index, spread_series_plot, label='Spread')
        ax.plot(rolling_mean_plot.index, rolling_mean_plot, color='black', linestyle='--', label='Rolling Mean (30d)')
        ax.plot(rolling_mean_plot.index, rolling_mean_plot + 1.5 * rolling_std_plot, color='red', linestyle='--', label='+2 Rolling Std (30d)')
        ax.plot(rolling_mean_plot.index, rolling_mean_plot - 1.5 * rolling_std_plot, color='green', linestyle='--', label='-2 Rolling Std (30d)')
        ax.set_title(pair)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

        if idx == 0:
            ax.set_ylabel('Spread Value')

    fig.tight_layout()
    fig.suptitle('Cointegrated Spreads for Top Pairs with Rolling Mean and Std (30 Days)', fontsize=16, y=1.02)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.show()


exchange = ccxt.binance()
markets = exchange.load_markets()

# Filter for USDT quote pairs (e.g. BTC/USDT, ETH/USDT)
symbols = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
symbols = symbols[:6]  # Limit to top 5 USDT pairs to save time

# Cop all the close prices and store them in a dictionary
all_prices = {} 
for sym in symbols:
    all_prices[sym] = close_prices(sym)
    time.sleep(exchange.rateLimit / 1000) #disconnects from exchange 

# Now calculate this for every pair and save the results
results = []

#here we do a giga-slow nested for loop to compare the two different assets relations (FIND A FASTER WAY TO DO THIS)
for i in range(len(symbols)):
    for j in range(i + 1, len(symbols)):
        sym1 = symbols[i]
        sym2 = symbols[j]

        price1 = all_prices[sym1]
        price2 = all_prices[sym2]

        r = pearson(price1, price2)
        var = spreadvariance(price1, price2)

        if r is not None and var is not None:
            results.append({
                'Pair': f'{sym1} vs {sym2}',
                'Pearson r': r,
                'Spread Variance': var
            })


# Output results
results_df = pd.DataFrame(results)
results_df = results_df.dropna().sort_values(by='Pearson r', ascending=False)
print(results_df.head(10))
results_df.to_csv("binance_pairwise_stats.csv", index=False)  # Optional: save results

# Scatter plot of Pearson r vs Spread Variance
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Pearson r'], results_df['Spread Variance'], alpha=0.6)
plt.xlabel('Pearson r')
plt.ylabel('Spread Variance')
plt.title('Pearson Correlation vs Spread Variance for Binance Top 5 USDT Pairs')
plt.grid(True)
plt.tight_layout()
plt.savefig("pearson_vs_variance.png")
plt.show()

top_pairs = results_df.head(4)['Pair']
plot_multiple_cointegrated_spreads(top_pairs, all_prices)

