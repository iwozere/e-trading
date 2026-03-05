import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, List, Dict

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
        plt.figure(figsize=(15, 10))
        counts = labels.value_counts().sort_index()
        counts.index = counts.index.map({1: "Profit Take", -1: "Stop Loss", 0: "Time Out"})
        counts.plot(kind='bar', color=['red', 'gray', 'green'])
        plt.title("Triple Barrier Hit Frequency", fontsize=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "tbm_barrier_hits.png", dpi=300)
        plt.close()

    def plot_prediction_diagnostics(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Scatter of Prob vs Actual returns and Error Distribution."""
        plt.figure(figsize=(24, 12))

        # 1. Error Distribution (Probabilities for the correct class)
        plt.subplot(1, 2, 1)
        sns.histplot(y_prob.max(axis=1), kde=True, color="blue")
        plt.title("Model Confidence (Max Prob) Distribution", fontsize=18)

        # 2. Probability Calibration (Simplified)
        plt.subplot(1, 2, 2)
        # Assuming y_true mapped to [0, 1, 2]
        correct_probs = [y_prob[i, val] for i, val in enumerate(y_true)]
        sns.histplot(correct_probs, kde=True, color="green")
        plt.title("Correct Class Probability Distribution", fontsize=18)

        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_diagnostics.png", dpi=300)
        plt.close()

    def plot_master_overlay(self, ohlcv: pd.DataFrame, signals: pd.Series, pf: Any):
        """
        Comprehensive multi-pane visualization.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 24), sharex=True)

        # 1. Price + Signals
        ax1.plot(ohlcv.index, ohlcv['close'], label='Close Price', alpha=0.4, color='black')

        buys = signals[signals == 1]
        sells = signals[signals == -1]

        if not buys.empty:
            ax1.scatter(buys.index, ohlcv.loc[buys.index, 'close'], marker='^', color='green', label='Buy / Close Short', s=200, alpha=0.9)
        if not sells.empty:
            ax1.scatter(sells.index, ohlcv.loc[sells.index, 'close'], marker='v', color='red', label='Sell / Close Long', s=200, alpha=0.9)

        ax1.set_title("Price Action & Realized Signals (Action-Based)", fontsize=24)
        ax1.legend(loc='best', fontsize=16)

        # 2. Equity Curve (Realized only to avoid mirroring price)
        # VectorBT trades.pnl.to_pd() gives PnL at exit points, indexed by original ohlcv
        # FillNa(0) and cumsum() gives the 'steppy' realized equity Curve
        pnl_series = pf.trades.pnl.to_pd().fillna(0.0)

        # Mask terminal exit PnL (forced close by VectorBT at end of period)
        if not pnl_series.empty:
            pnl_series.iloc[-1] = 0.0

        realized_pnl = pnl_series.cumsum()
        realized_equity = realized_pnl + pf.init_cash

        ax2.plot(realized_equity.index, realized_equity, label='Realized Equity (Steppy)', color='blue', linewidth=3)

        # Benchmark (Buy & Hold) - Mark-to-Market
        benchmark = (ohlcv['close'] / ohlcv['close'].iloc[0]) * pf.init_cash
        ax2.plot(benchmark.index, benchmark, label='Benchmark (B&H)', color='gray', linestyle='--', alpha=0.6)

        ax2.set_title("Equity Curve vs Benchmark", fontsize=24)
        ax2.legend(loc='best', fontsize=16)

        plt.tight_layout()
        plt.savefig(self.output_dir / "strategy_overlay.png", dpi=300)
        plt.close()

    def plot_walk_forward_equity(self, combined_equity: pd.Series):
        """Plots the combined OOS equity curve from Walk-Forward Analysis."""
        plt.figure(figsize=(15, 8))
        combined_equity.plot(color='purple', linewidth=2)
        plt.title("Walk-Forward Analysis: Combined Out-of-Sample Equity", fontsize=18)
        plt.ylabel("Portfolio Value")
        plt.tight_layout()
        plt.savefig(self.output_dir / "wfa_equity.png", dpi=300)
        plt.close()

    def plot_monte_carlo_distribution(self, shuffled_results: List[float], skipped_results: List[float]):
        """Plots the distribution of final equity outcomes from Monte Carlo runs."""
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)
        sns.histplot(shuffled_results, kde=True, color="blue")
        plt.axvline(1.0, color='red', linestyle='--')
        plt.title("MC: Shuffled Returns Outcomes", fontsize=14)

        plt.subplot(1, 2, 2)
        sns.histplot(skipped_results, kde=True, color="green")
        plt.axvline(1.0, color='red', linestyle='--')
        plt.title("MC: Random Skips Outcomes", fontsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / "monte_carlo_dist.png", dpi=300)
        plt.close()

    def plot_sensitivity_report(self, sensitivity_results: List[Dict[str, Any]]):
        """Plots a bar chart of Sharpe ratios for different parameter perturbations."""
        df = pd.DataFrame(sensitivity_results)
        if df.empty: return

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='perturbation', y='sharpe', palette='viridis')
        plt.title("Parameter Sensitivity Analysis (Sharpe Ratio)", fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "sensitivity_report.png", dpi=300)
        plt.close()
