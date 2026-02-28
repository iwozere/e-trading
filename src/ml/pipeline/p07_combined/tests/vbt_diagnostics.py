import vectorbt as vbt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def run_vbt_diagnostics():
    """
    Research and Diagnostic script for VectorBT Portfolio operations.
    Used for verifying signal extraction and realized equity calculations.
    """
    _logger.info("Running VectorBT Diagnostics...")

    # 1. Setup Mock Data
    close = pd.Series([10, 11, 15, 8, 10, 15, 12], name='close')
    # Trade 1: Buy (0), Sell (2). PnL = 5
    # Trade 2: Buy (4), Sell (6). PnL = 2
    entries = pd.Series([True, False, False, False, True, False, False])
    exits = pd.Series([False, False, True, False, False, False, True])

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=100)

    # 2. Verify Signal Extraction (Assets Diff)
    _logger.info("Verifying Signal Extraction (pf.assets().diff())...")
    assets = pf.assets()
    diff = assets.diff()
    if not diff.empty:
        diff.iloc[0] = assets.iloc[0] # Capture initial entry

    plot_sigs = pd.Series(0, index=close.index)
    plot_sigs[diff > 0] = 1
    plot_sigs[diff < 0] = -1

    _logger.info("Realized Signals:\n%s", plot_sigs)

    # 3. Verify Realized Equity (Steppy Curve)
    _logger.info("Verifying Realized Equity Calculation...")
    # trade_pnl.to_pd() gives PnL at exit points
    trade_pnl_series = pf.trades.pnl.to_pd().fillna(0.0)
    realized_equity = trade_pnl_series.cumsum() + pf.init_cash

    _logger.info("Realized Equity Curve:\n%s", realized_equity)

    # 4. Correctness Check
    expected_final = 100 + 5 + 2 # (15-10) + (12-10) assuming 1 unit
    # Note: VectorBT might use fractional units/compounding if not specified
    # but for this simple case:
    _logger.info("Final Portfolio Value: %.2f", pf.value().iloc[-1])
    _logger.info("Final Realized Equity: %.2f", realized_equity.iloc[-1])

if __name__ == "__main__":
    run_vbt_diagnostics()
