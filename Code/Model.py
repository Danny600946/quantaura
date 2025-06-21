import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
initial_capital = 500
years = 2
num_days = 252 * years  # 2 years of trading days
num_simulations = 100

# === Realistic Sharpe ranges ===
high_sharpe_range = (2, 5)
low_sharpe_range = (0.5, 1.5)

# === Realistic annual volatility ===
annual_vol = 0.075  # 10% annual volatility
daily_vol = annual_vol / np.sqrt(252)  # Convert to daily vol

# === Helper Function ===
def simulate_capital_path(sharpe, initial, num_steps):
    daily_return = sharpe * daily_vol  # μ = Sharpe × σ
    returns = np.random.normal(loc=daily_return, scale=daily_vol, size=num_steps)
    capital_path = initial * np.cumprod(1 + returns)
    return capital_path

def meanmontecarlo(returns, years):
    yearly_returns = ((returns[:, -1] / returns[:, 0]) ** (1 / years)) - 1
    yearly_returns_mean = yearly_returns.mean()
    max_yearly_returns = yearly_returns.max()
    min_yearly_returns = yearly_returns.min()
    print('The Average Annual Return is {:.2%}, the minimum yearly return is {:.2%}, the maximum was {:.2%}'.format(
        yearly_returns_mean, min_yearly_returns, max_yearly_returns))

# === Simulate Monte Carlo Paths ===
full_paths = []
for _ in range(num_simulations):
    # High Sharpe phase
    sharpe_high = np.random.uniform(*high_sharpe_range)
    high_path = simulate_capital_path(sharpe_high, initial_capital, num_days // 2)

    # Low Sharpe phase (starting from last capital of high phase)
    sharpe_low = np.random.uniform(*low_sharpe_range)
    low_path = simulate_capital_path(sharpe_low, high_path[-1], num_days // 2)

    # Combine both phases
    full_path = np.concatenate([high_path, low_path])
    full_paths.append(full_path)

full_paths = np.array(full_paths)
meanmontecarlo(full_paths,years)


# === Plotting ===
plt.figure(figsize=(12, 6))
for path in full_paths:
    plt.plot(path, alpha=0.7)

plt.axvline(num_days // 2, color='black', linestyle='--', label='Transition Point')
plt.ylim(top=1_000_000)
plt.title("Monte Carlo Simulation: £500 Initial → High Sharpe (2–5) → Low Sharpe (0.5–1.5)")
plt.xlabel("Days")
plt.ylabel("Capital (£)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
