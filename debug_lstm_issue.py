#!/usr/bin/env python3
"""
Debug script to identify LSTM optimization issues.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.ml.pipeline.hmm_lstm_01.x_05_optuna_lstm import LSTMOptimizer
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

def debug_psny_issue():
    """Debug the PSNY LSTM optimization issue."""
    try:
        print("🔍 Debugging PSNY LSTM optimization issue...")

        # Initialize optimizer
        optimizer = LSTMOptimizer()

        # Test data loading
        print("\n1. Testing data loading...")
        symbol = "PSNY"
        timeframe = "1d"
        provider = "yfinance"

        # Find labeled data file
        pattern = f"{provider}_{symbol}_{timeframe}_*_labeled.csv"
        csv_files = list(optimizer.labeled_data_dir.glob(pattern))

        if not csv_files:
            print(f"❌ No labeled data found for {symbol} {timeframe}")
            return

        csv_file = sorted(csv_files)[-1]
        print(f"✅ Found data file: {csv_file}")

        # Load data
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return', 'regime']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return
        else:
            print(f"✅ All required columns present")

        # Check data quality
        print(f"\n2. Checking data quality...")
        print(f"   - Null values: {df.isnull().sum().sum()}")
        print(f"   - Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        print(f"   - Regime values: {df['regime'].unique()}")
        print(f"   - Log return range: {df['log_return'].min():.6f} to {df['log_return'].max():.6f}")

        # Test optimized indicators loading
        print(f"\n3. Testing optimized indicators loading...")
        optimized_indicators = optimizer.load_optimized_indicators(symbol, timeframe)
        if optimized_indicators:
            print(f"✅ Loaded optimized indicators: {list(optimized_indicators.keys())}")
        else:
            print(f"❌ No optimized indicators found")
            return

        # Test feature preparation
        print(f"\n4. Testing feature preparation...")
        try:
            # Apply optimized indicators
            df_with_indicators = optimizer.apply_optimized_indicators(df, optimized_indicators)
            print(f"✅ Applied optimized indicators, now have {len(df_with_indicators.columns)} columns")

            # Prepare features
            features = optimizer.prepare_lstm_features(df_with_indicators)
            print(f"✅ Selected {len(features)} features: {features[:5]}...")

        except Exception as e:
            print(f"❌ Error in feature preparation: {e}")
            return

        # Test data preparation
        print(f"\n5. Testing data preparation...")
        try:
            sequence_length = 60  # Default value
            data_dict = optimizer.prepare_data(df_with_indicators, features, sequence_length)
            print(f"✅ Data preparation successful:")
            print(f"   - X_train shape: {data_dict['X_train'].shape}")
            print(f"   - X_val shape: {data_dict['X_val'].shape}")
            print(f"   - X_test shape: {data_dict['X_test'].shape}")
            print(f"   - n_features: {data_dict['n_features']}")

        except Exception as e:
            print(f"❌ Error in data preparation: {e}")
            return

        # Test scaling
        print(f"\n6. Testing data scaling...")
        try:
            scaled_data, scalers = optimizer.scale_data(data_dict)
            print(f"✅ Data scaling successful")

        except Exception as e:
            print(f"❌ Error in data scaling: {e}")
            return

        # Test model creation
        print(f"\n7. Testing model creation...")
        try:
            import torch
            from src.ml.pipeline.hmm_lstm_01.x_05_optuna_lstm import LSTMModel

            model = LSTMModel(
                input_size=scaled_data['n_features'],
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            )
            print(f"✅ Model creation successful")
            print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        except Exception as e:
            print(f"❌ Error in model creation: {e}")
            return

        print(f"\n✅ All tests passed! The issue might be in the Optuna optimization loop.")
        print(f"   Try running with fewer trials or different hyperparameters.")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_psny_issue()

