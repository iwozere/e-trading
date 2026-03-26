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
        Generates enhanced performance chart with trade markers, VIX, and Equity subplots.
        Price Charting with Triangles (^, v) and X markers.
        Subplots for Price, VIX Z-Score, and Equity Growth.
        """
        # Significantly enlarged figure size (5x wider and higher than default 14x10 would be 70x50, 
        # but let's use a very large but readable 30x20)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 20), sharex=True, 
                                            gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot 1: Price and Markers
        ax1.plot(results.index, results["Price"], label=f"{ticker} Price", color="blue", alpha=0.5, linewidth=2)
        
        # Draw Buy Markers (Green Triangles)
        buy_dates = markers.get("buy", [])
        if buy_dates:
            valid_buy_dates = [d for d in buy_dates if d in results.index]
            if valid_buy_dates:
                buy_prices = results.loc[valid_buy_dates, "Price"]
                ax1.scatter(valid_buy_dates, buy_prices, marker="^", color="green", s=200, label="Buy Tier", zorder=5)
            
        # Draw Sell Markers (Red Triangles)
        sell_dates = markers.get("sell", [])
        if sell_dates:
            valid_sell_dates = [d for d in sell_dates if d in results.index]
            if valid_sell_dates:
                sell_prices = results.loc[valid_sell_dates, "Price"]
                ax1.scatter(valid_sell_dates, sell_prices, marker="v", color="red", s=200, label="VIX Exit", zorder=5)
            
        # Draw Stop-Loss Markers (Black X)
        sl_dates = markers.get("stop_loss", [])
        if sl_dates:
            valid_sl_dates = [d for d in sl_dates if d in results.index]
            if valid_sl_dates:
                sl_prices = results.loc[valid_sl_dates, "Price"]
                ax1.scatter(valid_sl_dates, sl_prices, marker="x", color="black", s=250, label="Stop-Loss", zorder=6, linewidths=3)
            
        ax1.set_title(f"Performance: {ticker} with VIX Scaling Signals", fontsize=20)
        ax1.set_ylabel("Price ($)", fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: VIX Z-Score
        ax2.plot(results.index, results["Z_Score"], label="VIX Z-Score", color="purple", linewidth=1.5)
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        
        # Draw threshold lines from config
        colors = ["orange", "red", "darkred"]
        for i, (name, tier) in enumerate(entry_tiers.items()):
            color = colors[i % len(colors)]
            ax2.axhline(tier['z_threshold'], color=color, linestyle="--", alpha=0.8, 
                        label=f"{name} ({tier['z_threshold']})", linewidth=1.5)
        
        ax2.set_ylabel("VIX Z-Score", fontsize=16)
        ax2.legend(loc="upper left", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Equity Growth (Normalized to $1000)
        # Assuming results["Portfolio_Value"] starts at config.initial_capital
        initial_capital = results["Portfolio_Value"].iloc[0]
        equity_growth = (results["Portfolio_Value"] / initial_capital) * 1000
        
        # Calculate Buy and Hold equity (normalized to $1000)
        initial_price = results["Price"].iloc[0]
        buy_and_hold = (results["Price"] / initial_price) * 1000
        
        ax3.plot(results.index, equity_growth, label="Strategy Equity ($1000 Invested)", color="forestgreen", linewidth=2.5)
        ax3.plot(results.index, buy_and_hold, label=f"Buy & Hold {ticker}", color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
        
        ax3.fill_between(results.index, 1000, equity_growth, where=(equity_growth >= 1000), 
                         facecolor='green', alpha=0.1)
        ax3.fill_between(results.index, 1000, equity_growth, where=(equity_growth < 1000), 
                         facecolor='red', alpha=0.1)
        
        ax3.set_ylabel("Equity ($)", fontsize=16)
        ax3.set_xlabel("Date", fontsize=16)
        ax3.set_title("Strategy Equity Growth vs Buy & Hold (Initial $1000)", fontsize=18)
        ax3.legend(loc="upper left", fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved enhanced performance chart to {output_path}")
