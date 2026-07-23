"""
Multi-Strategy Prop Firm Evaluation Monte Carlo Simulation
------------------------------------------------------------
Simulates a prop-firm challenge (profit target / daily DD / total DD rules)
as if several strategies were being traded concurrently in one account, with
their combined dollar P&L determining pass/fail.

Usage:
    python propfirm_monte_carlo_multi.py --data strategy1.csv strategy2.csv strategy3.csv
    python propfirm_monte_carlo_multi.py --data s1.csv s2.csv --sims 5000 --periods 22 \
        --target 0.10 --daily-dd 0.05 --total-dd 0.10

Each CSV is expected to have DATE and EQUITY columns (same format as your
original single-strategy script).

Methodology note:
    Like the companion multi_strategy_monte_carlo.py script, this uses a
    CORRELATED bootstrap: each simulated day draws ONE random historical
    day-index and applies that day's return to every strategy at once, so
    the strategies' historical co-movement is preserved rather than assumed
    away. Daily/total drawdown rules are then evaluated on the COMBINED
    (summed) account equity, since that's what the prop firm actually
    measures when several strategies share one funded account.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_strategy(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='utf-16')
    except (UnicodeError, UnicodeDecodeError):
        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception:
            df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()

    if 'DATE' not in df.columns or 'EQUITY' not in df.columns:
        raise ValueError(
            f"'{filepath}' must contain DATE and EQUITY columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    return df[['DATE', 'EQUITY']]


def build_return_matrix(filepaths: list[str]):
    """
    Load all strategies (each with irregular per-trade timestamps), resample each
    onto a shared DAILY calendar (last trade's equity per day, forward-filled on
    non-trading days), then restrict to the window where ALL strategies are active.
    """
    strategy_names = [os.path.splitext(os.path.basename(fp))[0] for fp in filepaths]
    daily_series = {}

    for name, fp in zip(strategy_names, filepaths):
        df = load_strategy(fp)
        s = df.set_index('DATE')['EQUITY']
        daily_series[name] = s.resample('D').last()

    combined = pd.DataFrame(daily_series)
    combined = combined.ffill()

    starts = [s.first_valid_index() for s in daily_series.values()]
    ends = [s.last_valid_index() for s in daily_series.values()]
    overlap_start, overlap_end = max(starts), min(ends)

    if overlap_start >= overlap_end:
        raise ValueError(
            "No overlapping active period across the provided strategies "
            "(one strategy may have finished trading before another started)."
        )

    combined = combined.loc[overlap_start:overlap_end].dropna()

    if len(combined) < 2:
        raise ValueError("Not enough overlapping daily data across the provided strategies.")

    initial_equities = {name: combined[name].iloc[0] for name in strategy_names}
    returns = combined[strategy_names].pct_change().dropna().reset_index(drop=True)

    trading_days = int((returns[strategy_names] != 0).any(axis=1).sum())

    print(f"Loaded {len(strategy_names)} strategies: {strategy_names}")
    print(f"Overlap window (all strategies active): {overlap_start.date()} -> {overlap_end.date()}")
    print(f"Calendar days in bootstrap pool: {len(returns)} (of which ~{trading_days} had at least one trade)")

    return returns, initial_equities, strategy_names


def main():
    parser = argparse.ArgumentParser(description="Multi-strategy prop firm evaluation Monte Carlo simulation")
    parser.add_argument('--data', nargs='+', required=True, help="Paths to one or more strategy CSV files")
    parser.add_argument('--sims', type=int, default=1000, help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument('--periods', type=int, default=22, help="Trading days per evaluation window (default: 22)")
    parser.add_argument('--target', type=float, default=0.10, help="Profit target, as fraction (default: 0.10)")
    parser.add_argument('--daily-dd', type=float, default=0.05, help="Max daily drawdown, as fraction (default: 0.05)")
    parser.add_argument('--total-dd', type=float, default=0.10, help="Max total drawdown, as fraction (default: 0.10)")
    args = parser.parse_args()

    returns_df, initial_equities, strategy_names = build_return_matrix(args.data)
    returns_matrix = returns_df.to_numpy()
    num_hist_periods = len(returns_df)

    num_simulations = args.sims
    num_periods = args.periods
    profit_target_pct = args.target
    max_daily_dd_pct = args.daily_dd
    max_total_dd_pct = args.total_dd

    total_initial_equity = sum(initial_equities.values())
    target_equity = total_initial_equity * (1 + profit_target_pct)
    max_total_dd_limit = total_initial_equity * (1 - max_total_dd_pct)

    equity_paths = np.zeros((num_periods + 1, num_simulations))
    equity_paths[0] = total_initial_equity

    outcomes = []
    days_to_outcome = []

    for sim in range(num_simulations):
        strat_equity = np.array([initial_equities[name] for name in strategy_names], dtype=float)
        current_equity = strat_equity.sum()
        path = [current_equity]
        status = 'Fail (Time Limit)'
        outcome_day = num_periods

        # Correlated bootstrap: one shared historical day-index per simulated day
        day_indices = np.random.choice(num_hist_periods, size=num_periods, replace=True)

        for day in range(num_periods):
            day_start_equity = current_equity
            day_returns = returns_matrix[day_indices[day]]  # same historical day applied to all strategies

            strat_equity = strat_equity * (1 + day_returns)
            current_equity = strat_equity.sum()
            path.append(current_equity)

            # Rule 1: Max Daily Drawdown (relative to combined equity at start of day)
            daily_dd = (day_start_equity - current_equity) / day_start_equity
            if daily_dd >= max_daily_dd_pct:
                status = 'Fail (Daily Drawdown)'
                outcome_day = day + 1
                path.extend([current_equity] * (num_periods - day - 1))
                break

            # Rule 2: Max Total Drawdown (absolute, from combined starting balance)
            total_dd = (total_initial_equity - current_equity) / total_initial_equity
            if total_dd >= max_total_dd_pct:
                status = 'Fail (Max Total Drawdown)'
                outcome_day = day + 1
                path.extend([current_equity] * (num_periods - day - 1))
                break

            # Rule 3: Profit Target
            if current_equity >= target_equity:
                status = 'Pass'
                outcome_day = day + 1
                path.extend([current_equity] * (num_periods - day - 1))
                break

        outcomes.append(status)
        days_to_outcome.append(outcome_day)
        equity_paths[:, sim] = path

    # Metrics
    total_sims = len(outcomes)
    passes = outcomes.count('Pass')
    fail_daily = outcomes.count('Fail (Daily Drawdown)')
    fail_total = outcomes.count('Fail (Max Total Drawdown)')
    fail_time = outcomes.count('Fail (Time Limit)')

    pass_rate = (passes / total_sims) * 100
    fail_daily_rate = (fail_daily / total_sims) * 100
    fail_total_rate = (fail_total / total_sims) * 100
    fail_time_rate = (fail_time / total_sims) * 100

    pass_days = [days_to_outcome[i] for i in range(total_sims) if outcomes[i] == 'Pass']
    avg_days_to_pass = np.mean(pass_days) if pass_days else 0

    print("\n" + "=" * 40)
    print("   MULTI-STRATEGY PROP FIRM METRICS   ")
    print("=" * 40)
    print(f"Strategies:               {strategy_names}")
    print(f"Combined Starting Equity: ${total_initial_equity:,.2f}")
    print(f"Target Equity (+{profit_target_pct*100:.0f}%):    ${target_equity:,.2f}")
    print(f"Drawdown Limit (-{max_total_dd_pct*100:.0f}%):   ${max_total_dd_limit:,.2f}")
    print("-" * 40)
    print(f"PASS RATE:                {pass_rate:.2f}% ({passes}/{total_sims})")
    print(f"FAIL (Daily Drawdown):    {fail_daily_rate:.2f}% ({fail_daily}/{total_sims})")
    print(f"FAIL (Max Drawdown):      {fail_total_rate:.2f}% ({fail_total}/{total_sims})")
    print(f"FAIL (Time Limit Out):    {fail_time_rate:.2f}% ({fail_time}/{total_sims})")
    print("-" * 40)
    if pass_days:
        print(f"Avg Days to Pass:         {avg_days_to_pass:.1f} days")
    print("=" * 40)

    # Plotting
    plt.figure(figsize=(12, 7))

    pass_idx = [i for i, x in enumerate(outcomes) if x == 'Pass']
    fail_dd_idx = [i for i, x in enumerate(outcomes) if 'Drawdown' in x]
    fail_time_idx = [i for i, x in enumerate(outcomes) if 'Time' in x]

    if pass_idx:
        plt.plot(equity_paths[:, pass_idx], color='green', alpha=0.03)
    if fail_dd_idx:
        plt.plot(equity_paths[:, fail_dd_idx], color='red', alpha=0.03)
    if fail_time_idx:
        plt.plot(equity_paths[:, fail_time_idx], color='orange', alpha=0.04)

    plt.axhline(y=target_equity, color='darkgreen', linestyle='--', linewidth=2, label=f'Profit Target (+{profit_target_pct*100:.0f}%)')
    plt.axhline(y=max_total_dd_limit, color='darkred', linestyle='--', linewidth=2, label=f'Max Drawdown Limit (-{max_total_dd_pct*100:.0f}%)')
    plt.axhline(y=total_initial_equity, color='black', linestyle=':', alpha=0.5, label='Starting Balance')

    plt.plot([], [], color='green', alpha=0.6, label='Passed Paths')
    plt.plot([], [], color='red', alpha=0.6, label='Breached Drawdown Paths')
    plt.plot([], [], color='orange', alpha=0.6, label='Timed Out Paths')

    plt.title(f'Multi-Strategy Prop Firm Evaluation ({len(strategy_names)} strategies combined)', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Combined Account Equity ($)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    out_path = 'propfirm_monte_carlo_multi.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()