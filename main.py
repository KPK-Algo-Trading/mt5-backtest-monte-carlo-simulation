import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data with encoding
df = pd.read_csv('data.csv', sep='\t', encoding='utf-16')

# 2. DEBUG: Print columns to see exactly what they are named
print("Actual column names in file:")
print(df.columns.tolist())

# 3. Clean column names (remove hidden spaces or symbols)
df.columns = df.columns.str.strip()

# 4. Now attempt to convert
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# 2. Calculate periodic returns
# We use equity to account for total account performance
returns = df['EQUITY'].pct_change().dropna()

# 3. Monte Carlo Parameters
num_simulations = 1000
num_periods = len(df)
initial_equity = df['EQUITY'].iloc[0]

# 4. Run Simulation
results = np.zeros((num_periods, num_simulations))
results[0] = initial_equity

for i in range(num_simulations):
    # Randomly sample returns with replacement
    random_returns = np.random.choice(returns, size=num_periods-1, replace=True)
    results[1:, i] = initial_equity * (1 + random_returns).cumprod()

# 5. Analysis
mean_path = np.mean(results, axis=1)
p5_path = np.percentile(results, 5, axis=1)
p95_path = np.percentile(results, 95, axis=1)

# 6. Plotting
plt.figure(figsize=(12, 6))
plt.plot(results, color='gray', alpha=0.1)
plt.plot(mean_path, color='blue', label='Mean Equity')
plt.plot(p5_path, color='red', linestyle='--', label='5th Percentile')
plt.plot(p95_path, color='green', linestyle='--', label='95th Percentile')
plt.title('Monte Carlo Simulation of Trading Equity')
plt.xlabel('Trade Period')
plt.ylabel('Equity')
plt.legend()
plt.show()

# Output final stats
print(f"Final Expected Equity (Mean): {mean_path[-1]:.2f}")
print(f"5th Percentile Outcome: {p5_path[-1]:.2f}")
print(f"95th Percentile Outcome: {p95_path[-1]:.2f}")