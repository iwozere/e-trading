import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Use Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')

def plot_p09_results(signal_df: pd.DataFrame, trades_df: pd.DataFrame, equity_curve: pd.Series, pair_name: str, timeframe: str, output_path: Path):
    """
    Generates a diagnostic plot for P09 Arbitrage results.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Price Action Overlay
    ax1.plot(signal_df.index, signal_df['price_a'], label=f'Price A ({pair_name.split("_")[0]})', alpha=0.7, color='blue')
    ax1.plot(signal_df.index, signal_df['price_b'], label=f'Price B ({pair_name.split("_")[1]})', alpha=0.7, color='orange')
    
    # Markers for trades
    if not trades_df.empty:
        # Long Spread Entry (Buy A)
        long_entries = trades_df[trades_df['side'] == 'Long Spread']
        if not long_entries.empty:
            ax1.scatter(long_entries['entry_time'], long_entries['entry_price_a'], 
                        marker='^', color='green', s=100, label='Long Spread Entry', zorder=5)
        
        # Short Spread Entry (Sell A)
        short_entries = trades_df[trades_df['side'] == 'Short Spread']
        if not short_entries.empty:
            ax1.scatter(short_entries['entry_time'], short_entries['entry_price_a'], 
                        marker='v', color='red', s=100, label='Short Spread Entry', zorder=5)
            
        # Exits
        exits = trades_df.dropna(subset=['exit_time'])
        if not exits.empty:
            ax1.scatter(exits['exit_time'], exits['exit_price_a'], 
                        marker='x', color='black', s=50, label='Exit', zorder=5)

    ax1.set_title(f"P09 Arbitrage: {pair_name} ({timeframe}) - Price & Trades", fontsize=16)
    ax1.legend(loc='upper left')
    ax1.set_ylabel("Price")
    
    # 2. Z-Score Subplot
    ax2.plot(signal_df.index, signal_df['zscore'], label='Z-Score', color='purple', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Thresholds (assumed from common config, but signal_df has the actual signals)
    # We can highlight where signal != 0
    ax2.fill_between(signal_df.index, 0, signal_df['zscore'], where=(signal_df['signal'] != 0), 
                     color='gray', alpha=0.2, label='In Trade')
    
    ax2.set_title("Spread Z-Score", fontsize=14)
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc='upper left')
    
    # 3. Equity Curve Subplot
    if not equity_curve.empty:
        ax3.plot(equity_curve.index, equity_curve.values, label='Strategy Equity', color='darkgreen', linewidth=2)
        ax3.fill_between(equity_curve.index, equity_curve.iloc[0], equity_curve.values, color='green', alpha=0.1)
        
    ax3.set_title("Cumulative Equity (Realized)", fontsize=14)
    ax3.set_ylabel("Portfolio Value")
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_path}")
