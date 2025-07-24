# File: scripts/evaluate_hmm.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def map_regimes(df):
    means = df.groupby('regime')['log_return'].mean()
    sorted_means = means.sort_values()
    regime_mapping = {}
    regime_mapping[sorted_means.index[0]] = 'bearish'
    regime_mapping[sorted_means.index[1]] = 'sideways'
    regime_mapping[sorted_means.index[2]] = 'bullish'
    df['regime_label'] = df['regime'].map(regime_mapping)
    return df


def plot_regimes(df, output_path):
    color_map = {'bullish': 'green', 'bearish': 'red', 'sideways': 'black'}

    plt.figure(figsize=(16, 8))
    for label, color in color_map.items():
        subset = df[df['regime_label'] == label]
        plt.plot(subset['timestamp'], subset['close'], '.', markersize=1, label=label, color=color)

    plt.title('Market Regimes')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to regime-labeled CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['timestamp'])
    df = map_regimes(df)

    base = os.path.splitext(os.path.basename(args.csv))[0]
    output_path = f'src/ml/hmm/model/{base}.png'
    plot_regimes(df, output_path)


if __name__ == '__main__':
    main()
