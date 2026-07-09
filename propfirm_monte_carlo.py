import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATA LOADING & CLEANING (From your original script)
# =====================================================================
df = pd.read_csv('data.csv', sep='\t', encoding='utf-16')
df.columns = df.columns.str.strip()
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# Calculate historical returns (assumed to be daily returns)
returns = df['EQUITY'].pct_change().dropna()

# =====================================================================
# 2. PROP FIRM TEST PARAMETERS
# =====================================================================
initial_equity = df['EQUITY'].iloc[0]  # Uses your data's starting equity
num_simulations = 1000
num_periods = 22  # Standard trading days in a month (1 month to pass)

# Rules
profit_target_pct = 0.10      # 10% Profit Target
max_daily_dd_pct = 0.05       # 5% Max Daily Drawdown
max_total_dd_pct = 0.10       # 10% Max Absolute Drawdown

target_equity = initial_equity * (1 + profit_target_pct)
max_total_dd_limit = initial_equity * (1 - max_total_dd_pct)

# =====================================================================
# 3. RUN PROP FIRM SIMULATION
# =====================================================================
# Array to store full equity paths (rows = days, cols = simulations)
equity_paths = np.zeros((num_periods + 1, num_simulations))
equity_paths[0] = initial_equity

outcomes = []
days_to_outcome = []

for sim in range(num_simulations):
    current_equity = initial_equity
    path = [initial_equity]
    status = 'Fail (Time Limit)'  # Default if 22 days pass without hit/violation
    outcome_day = num_periods
    
    # Sample returns for this simulation path
    sim_returns = np.random.choice(returns, size=num_periods, replace=True)
    
    for day in range(num_periods):
        day_start_equity = current_equity
        r = sim_returns[day]
        
        # Update equity for the day
        current_equity = current_equity * (1 + r)
        path.append(current_equity)
        
        # Rule 1: Max Daily Drawdown check (relative to day's starting equity)
        daily_dd = (day_start_equity - current_equity) / day_start_equity
        if daily_dd >= max_daily_dd_pct:
            status = 'Fail (Daily Drawdown)'
            outcome_day = day + 1
            # Pad the rest of the path with current equity for flatline visualization
            path.extend([current_equity] * (num_periods - day - 1))
            break
            
        # Rule 2: Max Total Drawdown check (Absolute from initial balance)
        total_dd = (initial_equity - current_equity) / initial_equity
        if total_dd >= max_total_dd_pct:
            status = 'Fail (Max Total Drawdown)'
            outcome_day = day + 1
            path.extend([current_equity] * (num_periods - day - 1))
            break
            
        # Rule 3: Profit Target check
        if current_equity >= target_equity:
            status = 'Pass'
            outcome_day = day + 1
            path.extend([current_equity] * (num_periods - day - 1))
            break
            
    outcomes.append(status)
    days_to_outcome.append(outcome_day)
    equity_paths[:, sim] = path

# =====================================================================
# 4. METRICS & ANALYSIS
# =====================================================================
total_sims = len(outcomes)
passes = outcomes.count('Pass')
fail_daily = outcomes.count('Fail (Daily Drawdown)')
fail_total = outcomes.count('Fail (Max Total Drawdown)')
fail_time = outcomes.count('Fail (Time Limit)')

pass_rate = (passes / total_sims) * 100
fail_daily_rate = (fail_daily / total_sims) * 100
fail_total_rate = (fail_total / total_sims) * 100
fail_time_rate = (fail_time / total_sims) * 100

# Average days to complete
pass_days = [days_to_outcome[i] for i in range(total_sims) if outcomes[i] == 'Pass']
avg_days_to_pass = np.mean(pass_days) if pass_days else 0

print("\n" + "="*40)
print("       PROP FIRM TEST METRICS         ")
print("="*40)
print(f"Starting Equity:          ${initial_equity:,.2f}")
print(f"Target Equity (10%):      ${target_equity:,.2f}")
print(f"Drawdown Limit (10%):     ${max_total_dd_limit:,.2f}")
print("-"*40)
print(f"PASS RATE:                {pass_rate:.2f}% ({passes}/{total_sims})")
print(f"FAIL (Daily Drawdown):    {fail_daily_rate:.2f}% ({fail_daily}/{total_sims})")
print(f"FAIL (Max Drawdown):      {fail_total_rate:.2f}% ({fail_total}/{total_sims})")
print(f"FAIL (Time Limit Out):    {fail_time_rate:.2f}% ({fail_time}/{total_sims})")
print("-"*40)
if pass_days:
    print(f"Avg Days to Pass:         {avg_days_to_pass:.1f} days")
print("="*40)

# =====================================================================
# 5. PLOTTING THE RESULTS
# =====================================================================
plt.figure(figsize=(12, 7))

# Identify indices for clear color grouping
pass_idx = [i for i, x in enumerate(outcomes) if x == 'Pass']
fail_dd_idx = [i for i, x in enumerate(outcomes) if 'Drawdown' in x]
fail_time_idx = [i for i, x in enumerate(outcomes) if 'Time' in x]

# Vectorized plotting for efficiency (handles 1000 lines instantly)
if pass_idx:
    plt.plot(equity_paths[:, pass_idx], color='green', alpha=0.03)
if fail_dd_idx:
    plt.plot(equity_paths[:, fail_dd_idx], color='red', alpha=0.03)
if fail_time_idx:
    plt.plot(equity_paths[:, fail_time_idx], color='orange', alpha=0.04)

# Threshold lines
plt.axhline(y=target_equity, color='darkgreen', linestyle='--', linewidth=2, label='Profit Target (+10%)')
plt.axhline(y=max_total_dd_limit, color='darkred', linestyle='--', linewidth=2, label='Max Drawdown Limit (-10%)')
plt.axhline(y=initial_equity, color='black', linestyle=':', alpha=0.5, label='Starting Balance')

# Custom Legend Labels to avoid 1000 duplicate entries
plt.plot([], [], color='green', alpha=0.6, label='Passed Paths')
plt.plot([], [], color='red', alpha=0.6, label='Breached Drawdown Paths')
plt.plot([], [], color='orange', alpha=0.6, label='Timed Out Paths')

plt.title('Prop Firm Evaluation: Monte Carlo Simulation', fontsize=14, fontweight='bold')
plt.xlabel('Trading Days', fontsize=12)
plt.ylabel('Account Equity ($)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()