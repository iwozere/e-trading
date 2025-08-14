#!/usr/bin/env python3
"""
Test a single LSTM trial to identify optimization issues.
"""

import sys
from pathlib import Path
import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.ml.pipeline.hmm_lstm_01.x_05_optuna_lstm import LSTMOptimizer
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

def test_single_trial():
    """Test a single LSTM trial for PSNY."""
    try:
        print("🧪 Testing single LSTM trial for PSNY...")

        # Initialize optimizer
        optimizer = LSTMOptimizer()

        # Override configuration for testing
        optimizer.n_trials = 1  # Just one trial
        optimizer.timeout = 300  # 5 minutes timeout

        symbol = "PSNY"
        timeframe = "1d"
        provider = "yfinance"

        print(f"Testing with: n_trials={optimizer.n_trials}, timeout={optimizer.timeout}s")

        # Run optimization
        result = optimizer.optimize_lstm(symbol, timeframe, provider)

        if result['success']:
            print(f"✅ Single trial successful!")
            print(f"   Best objective value: {result['best_objective_value']}")
            print(f"   Best parameters: {result['best_params']}")
        else:
            print(f"❌ Single trial failed: {result['error']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_trial()

