"""
Market Regime Visualization Script.

This script serves as a command-line tool to visualize pre-calculated HMM
market regimes from a CSV file. It reads the data, maps the numerical regime
labels to descriptive categories (bullish, bearish, sideways), and generates
a multi-panel plot.

The output plot contains two subplots:
1.  The top plot shows the 'close' price, with data points colored according
    to their assigned market regime.
2.  The bottom plot shows the regime sequence as distinct, colored dots on
    three separate levels, making it easy to see regime changes.

Usage:
    python your_script_name.py --csv path/to/your_regime_data.csv
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # Go up 3 levels from 'src/ml/hmm'
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import argparse

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def map_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps numerical HMM regimes to descriptive string labels.

    The mapping is determined by calculating the mean `log_return` for each
    regime. The regime with the lowest mean is labeled 'bearish', the highest
    is 'bullish', and the one in the middle is 'sideways'. This assumes
    exactly three regimes.
    """
    means = df.groupby('regime')['log_return'].mean()
    sorted_means = means.sort_values()

    if len(sorted_means) != 3:
        raise ValueError("This function requires exactly 3 regimes to map to bullish, bearish, and sideways.")

    regime_mapping = {
        sorted_means.index[0]: 'bearish',
        sorted_means.index[1]: 'sideways',
        sorted_means.index[2]: 'bullish'
    }
    label_to_numeric = {'bearish': 0, 'sideways': 1, 'bullish': 2}

    df['regime_label'] = df['regime'].map(regime_mapping)
    df['regime_numeric'] = df['regime_label'].map(label_to_numeric)

    return df


def plot_regimes(df: pd.DataFrame, output_path: str):
    """
    Generates and saves a two-panel plot of market price and regimes.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp', 'close', 'regime_label',
                           and 'regime_numeric' columns.
        output_path (str): The file path where the plot image will be saved.
    """
    color_map = {'bullish': 'green', 'bearish': 'red', 'sideways': 'black'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(60, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})

    # --- Top Subplot: Price Visualization ---
    ax1.plot(df['timestamp'], df['close'], color='gray', alpha=0.3, linewidth=1)

    for label, color in color_map.items():
        subset = df[df['regime_label'] == label]
        ax1.scatter(subset['timestamp'], subset['close'],
                    s=1,  # Small dot size for price
                    label=label.capitalize(),
                    color=color,
                    alpha=0.7)

    ax1.set_title('Market Regimes Visualization')
    ax1.set_ylabel('Price')
    ax1.legend(markerscale=10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Bottom Subplot: Regime Sequence (with colored dots) ---
    # Loop through the regimes again to plot colored dots on ax2
    for label, color in color_map.items():
        subset = df[df['regime_label'] == label]
        # Use scatter to plot dots on their respective numeric levels
        ax2.scatter(subset['timestamp'], subset['regime_numeric'],
                    color=color,
                    s=5, # Make these dots slightly larger to be visible
                    alpha=0.7)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Bearish', 'Sideways', 'Bullish'])
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Regime')
    ax2.grid(True, axis='x', linestyle='--', linewidth=0.5)
    # Set y-limits to give a little padding around the dots
    ax2.set_ylim(-0.5, 2.5)

    # --- Final Touches ---
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """
    Main execution function for the regime visualization pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Visualize HMM market regimes from a CSV file."
    )
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to the CSV file with a pre-calculated "regime" column.'
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['timestamp'])
    df_labeled = map_regimes(df)

    base_name = Path(args.csv).stem
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}.png"

    _logger.info("Generating plot for %s...")
    plot_regimes(df_labeled, output_path)
    _logger.info("Plot saved successfully to %s")


if __name__ == '__main__':
    main()
