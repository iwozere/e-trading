import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

class P13Plotter:
    @staticmethod
    def plot_results(results: pd.DataFrame, markers: Dict[str, List], ticker: str, entry_tiers: Dict[str, Dict[str, float]], output_path: str):
        """
        Generates enhanced performance chart with trade markers and VIX subplots.
        Price Charting with Triangles (^, v) and X markers.
        Subplots for Price and VIX Z-Score.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Price and Markers
        ax1.plot(results.index, results["Price"], label=f"{ticker} Price", color="blue", alpha=0.5)
        
        # Draw Buy Markers (Green Triangles)
        buy_dates = markers.get("buy", [])
        if buy_dates:
            # Filtering existance in results to avoid KeyError
            valid_buy_dates = [d for d in buy_dates if d in results.index]
            if valid_buy_dates:
                buy_prices = results.loc[valid_buy_dates, "Price"]
                ax1.scatter(valid_buy_dates, buy_prices, marker="^", color="green", s=100, label="Buy Tier", zorder=5)
            
        # Draw Sell Markers (Red Triangles)
        sell_dates = markers.get("sell", [])
        if sell_dates:
            valid_sell_dates = [d for d in sell_dates if d in results.index]
            if valid_sell_dates:
                sell_prices = results.loc[valid_sell_dates, "Price"]
                ax1.scatter(valid_sell_dates, sell_prices, marker="v", color="red", s=100, label="VIX Exit", zorder=5)
            
        # Draw Stop-Loss Markers (Black X)
        sl_dates = markers.get("stop_loss", [])
        if sl_dates:
            valid_sl_dates = [d for d in sl_dates if d in results.index]
            if valid_sl_dates:
                sl_prices = results.loc[valid_sl_dates, "Price"]
                ax1.scatter(valid_sl_dates, sl_prices, marker="x", color="black", s=120, label="Stop-Loss", zorder=6, linewidths=2)
            
        ax1.set_title(f"Performance: {ticker} with VIX Scaling Signals")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: VIX Z-Score
        ax2.plot(results.index, results["Z_Score"], label="VIX Z-Score", color="purple")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        
        # Draw threshold lines from config
        colors = ["orange", "red", "darkred"]
        for i, (name, tier) in enumerate(entry_tiers.items()):
            color = colors[i % len(colors)]
            ax2.axhline(tier['z_threshold'], color=color, linestyle="--", alpha=0.8, label=f"{name} ({tier['z_threshold']})")
        
        ax2.set_ylabel("VIX Z-Score")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left", fontsize="small")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved enhanced performance chart to {output_path}")
