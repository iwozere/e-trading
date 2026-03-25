import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_manager import DataManager

def get_live_signals(window=30):
    """
    Example of how to get live VIX signals using the shared DataManager.
    """
    # 1. Fetch recent data using DataManager
    dm = DataManager()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=60)
    
    # We only need VIX for signals in this example
    vix_df = dm.get_ohlcv("^VIX", "1d", start_dt, end_dt)
    
    if vix_df is None or vix_df.empty:
        return "ERROR: Could not fetch VIX data."

    # 2. Calculate Current VIX Z-Score
    vix = vix_df['close']
    vix_mean = vix.rolling(window=window).mean()
    vix_std = vix.rolling(window=window).std()
    current_z = (vix.iloc[-1] - vix_mean.iloc[-1]) / vix_std.iloc[-1]

    print(f"--- Market State ---")
    print(f"Current VIX: {vix.iloc[-1]:.2f}")
    print(f"Current VIX Z-Score: {current_z:.2f}\n")

    # 3. Define Signal Logic
    if current_z > 3.5:
        note = "EXTREME FEAR: Signal is BUY TIER 3 (Max Exposure)"
    elif current_z > 2.5:
        note = "HIGH FEAR: Signal is BUY TIER 2"
    elif current_z > 1.5:
        note = "ELEVATED FEAR: Signal is BUY TIER 1"
    elif current_z < 0.0:
        note = "COMPLACENCY: Signal is SELL / EXIT ALL"
    else:
        note = "NEUTRAL: Signal is HOLD existing positions"

    return note

# Usage
if __name__ == "__main__":
    print(get_live_signals())