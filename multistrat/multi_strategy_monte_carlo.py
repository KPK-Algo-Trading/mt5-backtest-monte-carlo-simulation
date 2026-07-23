"""
Multi-Strategy Monte Carlo Simulation
--------------------------------------
Simulates several strategies' equity curves as if they were all being traded
concurrently in the same account (i.e. their dollar P&L is summed day by day).

Usage:
    python multi_strategy_monte_carlo.py --data strategy1.csv strategy2.csv strategy3.csv
    python multi_strategy_monte_carlo.py --data strategy1.csv strategy2.csv --sims 5000

Each CSV is expected to have DATE and EQUITY columns (same format as your
original single-strategy script: tab-separated, UTF-16 by default, with a
plain-CSV fallback).

Methodology note (important):
    Instead of bootstrapping each strategy's returns independently, this
    script performs a CORRELATED bootstrap: for each simulated day it draws
    ONE random historical day-index and applies that day's return to every
    strategy simultaneously. This preserves whatever correlation existed
    between your strategies historically (e.g. if two strategies both tend
    to lose on high-volatility days, that joint behavior is retained).
    Sampling each strategy independently would understate portfolio risk
    by assuming the strategies are uncorrelated, which is rarely true for
    strategies trading similar markets/timeframes.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_strategy(filepath: str) -> pd.DataFrame:
    """Load a single strategy's equity curve, with a couple of encoding fallbacks."""
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


def build_return_matrix(filepaths: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Load all strategies (each with irregular per-trade timestamps), resample each
    onto a shared DAILY calendar (last trade's equity per day, forward-filled on
    non-trading days), then restrict to the window where ALL strategies are active.

    Returns:
      - a DataFrame of daily pct returns, one column per strategy
      - a dict of each strategy's own starting equity (at the start of the overlap window)
      - the list of strategy names
    """
    strategy_names = [os.path.splitext(os.path.basename(fp))[0] for fp in filepaths]
    daily_series = {}

    for name, fp in zip(strategy_names, filepaths):
        df = load_strategy(fp)
        s = df.set_index('DATE')['EQUITY']
        # Collapse multiple same-day trades to the last equity mark of that day
        daily_series[name] = s.resample('D').last()

    combined = pd.DataFrame(daily_series)

    # Forward-fill non-trading days (equity doesn't change if a strategy didn't trade)
    combined = combined.ffill()

    # Restrict to the window where every strategy has already started and none has ended
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
    parser = argparse.ArgumentParser(description="Multi-strategy portfolio Monte Carlo simulation")
    parser.add_argument('--data', nargs='+', required=True, help="Paths to one or more strategy CSV files")
    parser.add_argument('--sims', type=int, default=1000, help="Number of Monte Carlo simulations (default: 1000)")
    args = parser.parse_args()

    returns_df, initial_equities, strategy_names = build_return_matrix(args.data)

    num_periods = len(returns_df)
    num_simulations = args.sims

    # Correlation between strategies' historical returns, useful sanity check
    if len(strategy_names) > 1:
        print("\nHistorical return correlation matrix:")
        print(returns_df.corr().round(3).to_string())

    total_initial_equity = sum(initial_equities.values())
    returns_matrix = returns_df.to_numpy()  # shape: (num_periods, num_strategies)

    # results[t, sim] = combined portfolio equity at period t for simulation sim
    portfolio_paths = np.zeros((num_periods + 1, num_simulations))
    portfolio_paths[0] = total_initial_equity

    # per-strategy running equity for the current simulation (reset each sim)
    for sim in range(num_simulations):
        strat_equity = np.array([initial_equities[name] for name in strategy_names], dtype=float)
        # Draw one shared day-index per period -> correlated bootstrap
        day_indices = np.random.choice(num_periods, size=num_periods, replace=True)
        for t, day_idx in enumerate(day_indices, start=1):
            day_returns = returns_matrix[day_idx]  # one return per strategy, same historical day
            strat_equity = strat_equity * (1 + day_returns)
            portfolio_paths[t, sim] = strat_equity.sum()

    # 5. Analysis
    mean_path = np.mean(portfolio_paths, axis=1)
    p5_path = np.percentile(portfolio_paths, 5, axis=1)
    p95_path = np.percentile(portfolio_paths, 95, axis=1)

    # 6. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_paths, color='gray', alpha=0.1)
    plt.plot(mean_path, color='blue', label='Mean Combined Equity')
    plt.plot(p5_path, color='red', linestyle='--', label='5th Percentile')
    plt.plot(p95_path, color='green', linestyle='--', label='95th Percentile')
    plt.title(f'Monte Carlo Simulation of Combined Equity ({len(strategy_names)} strategies)')
    plt.xlabel('Trade Period')
    plt.ylabel('Combined Equity')
    plt.legend()
    plt.tight_layout()

    out_path = 'multi_strategy_monte_carlo.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    print("\n" + "=" * 40)
    print(f"Combined Starting Equity: {total_initial_equity:,.2f}")
    print(f"Final Expected Equity (Mean): {mean_path[-1]:,.2f}")
    print(f"5th Percentile Outcome: {p5_path[-1]:,.2f}")
    print(f"95th Percentile Outcome: {p95_path[-1]:,.2f}")
    print("=" * 40)


if __name__ == '__main__':
    main()