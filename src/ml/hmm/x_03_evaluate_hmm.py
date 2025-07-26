"""
Market Regime Visualization Script.

This script serves as a command-line tool to visualize pre-calculated HMM
(Hidden Markov Model) market regimes from a CSV file. It reads the data,
maps the numerical regime labels to descriptive categories (bullish, bearish,
sideways), and generates a price chart where data points are colored according
to their assigned regime.

Input:
    A CSV file specified via the command line. This file must contain at least
    the following columns:
    - 'timestamp': The datetime for each data point.
    - 'close': The closing price.
    - 'log_return': The logarithmic return, used for interpretation.
    - 'regime': An integer (e.g., 0, 1, 2) representing the HMM state.

Output:
    A PNG image file named after the input CSV, saved to a fixed location
    ('src/ml/hmm/model/'). The plot visualizes the price data colored by the
    interpreted market regime.

Usage:
    python your_script_name.py --csv path/to/your_regime_data.csv
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # Go up 3 levels from 'src/ml/hmm'
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def map_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps numerical HMM regimes to descriptive string labels.

    The mapping is determined by calculating the mean `log_return` for each
    regime. The regime with the lowest mean is labeled 'bearish', the highest
    is 'bullish', and the one in the middle is 'sideways'. This assumes
    exactly three regimes.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'regime' (int) and
                           'log_return' (float) columns.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'regime_label' column added.
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
    df['regime_label'] = df['regime'].map(regime_mapping)
    return df


def plot_regimes(df: pd.DataFrame, output_path: str):
    """
    Generates and saves a plot of market regimes.

    This function plots the 'close' price over time, coloring each data point
    based on its 'regime_label'. A predefined color map is used for consistency.
    The resulting plot is saved to the specified output path.

    Args:
        df (pd.DataFrame): DataFrame containing 'timestamp', 'close', and
                           'regime_label' columns.
        output_path (str): The file path where the plot image will be saved.
    """
    color_map = {'bullish': 'green', 'bearish': 'red', 'sideways': 'black'}

    plt.figure(figsize=(16, 8))
    for label, color in color_map.items():
        subset = df[df['regime_label'] == label]
        # Use scatter plot for individual points instead of line plot
        plt.scatter(subset['timestamp'], subset['close'], s=1, label=label, color=color)

    plt.title('Market Regimes Visualization')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend(markerscale=10) # Increase legend marker size for visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """
    Main execution function to run the regime visualization pipeline.

    Parses command-line arguments, reads the data, orchestrates the regime
    mapping and plotting, and saves the final visualization.
    """
    parser = argparse.ArgumentParser(
        description="Visualize HMM market regimes from a CSV file."
    )
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to the CSV file containing data with a pre-calculated "regime" column.'
    )
    args = parser.parse_args()

    # Read and process the data
    df = pd.read_csv(args.csv, parse_dates=['timestamp'])
    df_labeled = map_regimes(df)

    # Define and create output path
    base = os.path.splitext(os.path.basename(args.csv))[0]
    # Note: The output directory is hardcoded.
    output_dir = 'src/ml/hmm/model/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{base}.png')

    print(f"Generating plot for {args.csv}...")
    plot_regimes(df_labeled, output_path)
    print(f"Plot saved successfully to {output_path}")


if __name__ == '__main__':
    main()