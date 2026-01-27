import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def calculate_consecutive_losses(trades_df: pd.DataFrame) -> int:
    """Calculate the maximum number of consecutive losses."""
    if trades_df.empty:
        return 0

    # Identify losing trades (net_pnl < 0)
    is_loss = trades_df['net_pnl'] < 0

    # Group consecutive identical values
    # cumsum() increments every time the value changes (True->False or False->True)
    # This gives us groups of consecutive True or False
    groups = is_loss.ne(is_loss.shift()).cumsum()

    # Filter for only loss groups and get their size
    consecutive_losses = is_loss[is_loss].groupby(groups).sum()

    if consecutive_losses.empty:
        return 0

    return int(consecutive_losses.max())

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino Ratio based on per-trade returns.

    Note: Ideally this would be time-based (e.g. daily returns), but with per-trade data
    we use the distribution of trade returns.
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    mean_return = returns.mean()
    downside = returns[returns < 0]

    if downside.empty:
        return np.inf

    # Downside deviation: sqrt(mean(downside_returns^2))
    downside_deviation = np.sqrt((downside ** 2).mean())

    if downside_deviation == 0:
        return np.inf

    return (mean_return - risk_free_rate) / downside_deviation

def calculate_metrics(simulation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate detailed metrics from simulation data."""
    trades = simulation_data.get('trades', [])

    base_metrics = {
        'bot_id': simulation_data.get('bot_id'),
        'symbol': simulation_data.get('symbol'),
        'strategy': simulation_data.get('bot_id').split('-strategy-')[-1] if '-strategy-' in simulation_data.get('bot_id', '') else 'unknown',
        'initial_balance': simulation_data.get('initial_balance'),
        'final_balance': simulation_data.get('final_balance'),
        'total_pnl': simulation_data.get('total_pnl'),
        'total_trades': simulation_data.get('total_trades'),
        'win_rate': simulation_data.get('win_rate'),
        'max_drawdown': simulation_data.get('max_drawdown')
    }

    if not trades:
        return {
            **base_metrics,
            'profit_factor': 0.0,
            'sortino_ratio': 0.0,
            'max_consecutive_losses': 0,
            'avg_trade_duration_min': 0.0,
            'avg_win_pnl': 0.0,
            'avg_loss_pnl': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }

    df = pd.DataFrame(trades)

    # Ensure numeric columns
    numeric_cols = ['net_pnl', 'gross_pnl', 'pnl_percentage', 'duration_minutes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Profit Factor
    gross_profits = df[df['gross_pnl'] > 0]['gross_pnl'].sum()
    gross_losses = abs(df[df['gross_pnl'] < 0]['gross_pnl'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf

    # Sortino Ratio (using pnl_percentage)
    sortino = calculate_sortino_ratio(df['pnl_percentage'])

    # Max Consecutive Losses
    max_consecutive_losses = calculate_consecutive_losses(df)

    # Average Trade Duration
    avg_duration = df['duration_minutes'].mean()

    # Win/Loss Averages
    winning_trades = df[df['net_pnl'] > 0]
    losing_trades = df[df['net_pnl'] <= 0]

    avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0.0

    return {
        **base_metrics,
        'profit_factor': round(profit_factor, 2),
        'sortino_ratio': round(sortino, 2),
        'max_consecutive_losses': max_consecutive_losses,
        'avg_trade_duration_min': round(avg_duration, 1),
        'avg_win_pnl': round(avg_win, 2),
        'avg_loss_pnl': round(avg_loss, 2),
        'largest_win': round(df['net_pnl'].max(), 2),
        'largest_loss': round(df['net_pnl'].min(), 2)
    }

def process_results(input_dir: str, output_file: str):
    """Read all JSON results and save aggregated analysis to CSV."""
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    results = []
    print(f"Found {len(json_files)} simulation result files.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics = calculate_metrics(data)
                results.append(metrics)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if results:
        df = pd.DataFrame(results)

        # Reorder columns for better readability
        cols = [
            'symbol', 'strategy', 'total_pnl', 'win_rate', 'profit_factor',
            'max_drawdown', 'sortino_ratio', 'max_consecutive_losses',
            'total_trades', 'avg_trade_duration_min', 'initial_balance',
            'final_balance', 'bot_id'
        ]
        # Only include columns that verify exist
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[cols]

        # Sort by Total PnL descending
        df = df.sort_values('total_pnl', ascending=False)

        df.to_csv(output_file, index=False)
        print(f"\nAnalysis complete. Results saved to: {output_file}")
        print("\nTop 5 Performing Strategies:")
        print(df[['symbol', 'strategy', 'total_pnl', 'win_rate', 'profit_factor']].head())
    else:
        print("No valid results to analyze.")

if __name__ == "__main__":
    # Default paths based on project structure
    # Assuming script is in src/trading/tools/
    # And results are in results/simulation/

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

    simulation_dir = os.path.join(project_root, "results", "simulation")
    output_csv = os.path.join(project_root, "results", "simulation_analysis.csv")

    print(f"Reading from: {simulation_dir}")
    print(f"Writing to: {output_csv}")

    process_results(simulation_dir, output_csv)
