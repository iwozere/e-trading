import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

# Use Agg backend for headless environments (like Pi)
import matplotlib
matplotlib.use('Agg')

class P07Visualizer:
    """
    Diagnostic Visualization Suite for p07_combined.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="darkgrid")

    def plot_tbm_hits(self, labels: pd.Series):
        """Frequency of barrier hits."""
        plt.figure(figsize=(10, 6))
        counts = labels.value_counts().sort_index()
        counts.index = counts.index.map({1: "Profit Take", -1: "Stop Loss", 0: "Time Out"})
        counts.plot(kind='bar', color=['red', 'gray', 'green'])
        plt.title("Triple Barrier Hit Frequency")
        plt.tight_layout()
        plt.savefig(self.output_dir / "tbm_barrier_hits.png")
        plt.close()

    def plot_prediction_diagnostics(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Scatter of Prob vs Actual returns and Error Distribution."""
        # Note: y_true here is usually the actual log return, not the label
        # but for simplicity in this client implementation, we'll assume y_true is the label for now
        # or we just plot a confusion matrix / probability distribution

        plt.figure(figsize=(12, 5))

        # 1. Error Distribution (Probabilities for the correct class)
        plt.subplot(1, 2, 1)
        sns.histplot(y_prob.max(axis=1), kde=True, color="blue")
        plt.title("Model Confidence (Max Prob) Distribution")

        # 2. Probability Calibration (Simplified)
        plt.subplot(1, 2, 2)
        # Assuming y_true mapped to [0, 1, 2]
        correct_probs = [y_prob[i, val] for i, val in enumerate(y_true)]
        sns.histplot(correct_probs, kde=True, color="green")
        plt.title("Correct Class Probability Distribution")

        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_diagnostics.png")
        plt.close()

    def plot_master_overlay(self, ohlcv: pd.DataFrame, signals: pd.Series, pf: Any):
        """
        Comprehensive multi-pane visualization.
        1. Price + Signals
        2. Equity Curve
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # 1. Price + Signals
        ax1.plot(ohlcv.index, ohlcv['close'], label='Close Price', alpha=0.4, color='black')

        buys = signals[signals == 1]
        sells = signals[signals == -1]

        if not buys.empty:
            ax1.scatter(buys.index, ohlcv.loc[buys.index, 'close'], marker='^', color='green', label='Buy / Close Short', s=60, alpha=0.8)
        if not sells.empty:
            ax1.scatter(sells.index, ohlcv.loc[sells.index, 'close'], marker='v', color='red', label='Sell / Close Long', s=60, alpha=0.8)

        ax1.set_title("Price Action & Realized Signals (Action-Based)")
        ax1.legend(loc='best')

        # 2. Equity Curve
        equity = pf.value()
        ax2.plot(equity.index, equity, label='Strategy Equity', color='blue')

        # Benchmark (Buy & Hold)
        benchmark = (ohlcv['close'] / ohlcv['close'].iloc[0]) * equity.iloc[0]
        ax2.plot(benchmark.index, benchmark, label='Benchmark (B&H)', color='gray', linestyle='--')

        ax2.set_title("Equity Curve vs Benchmark")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "strategy_overlay.png")
        plt.close()
