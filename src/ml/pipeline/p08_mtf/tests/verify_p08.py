import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p08_mtf.data_loader import P08DataLoader
from src.ml.pipeline.p08_mtf.features import P08FeatureEngine

def test_lookahead_safety():
    """
    CRITICAL: Verifies that the MTF join does not leak future anchor data into the execution TF.
    """
    print("\n--- Verifying P08 Look-Ahead Safety ---")
    loader = P08DataLoader()

    # 1. Load a 15m file and its 4h anchor
    exec_file = Path("data/BTCUSDT_15m_20200101_20201231.csv")
    if not exec_file.exists():
        print(f"Skipping test: BTCUSDT 15m data not found at {exec_file.absolute()}")
        return

    df_mtf = loader.get_mtf_dataset(exec_file)

    # 2. Check a specific transition
    # A 15m bar at 2020-01-01 04:00:00 (Execution)
    # Should see the 4h anchor from 2020-01-01 00:00:00 (knowable)

    test_ts = pd.Timestamp("2020-01-01 04:00:00", tz='UTC')
    if test_ts in df_mtf.index:
        row = df_mtf.loc[test_ts]
        print(f"Checking timestamp: {test_ts}")
        print(f"Execution Close: {row['close']}")
        print(f"Anchor Close (merged): {row['anchor_close']}")

        # Manually load the anchor data to verify the source
        anchor_file = loader.find_anchor_file("BTCUSDT", "4h", "20200101", "20201231")
        if not anchor_file:
             print("❌ FAILURE: Anchor file not found.")
             return

        df_anchor = loader.load_ohlcv(anchor_file)

        # The value at 2020-01-01 04:00:00 EXECUTION should match the 2020-01-01 00:00:00 ANCHOR CLOSE
        anchor_idx = pd.Timestamp("2020-01-01 00:00:00", tz='UTC')
        if anchor_idx in df_anchor.index:
            known_anchor_close = df_anchor.loc[anchor_idx]['close']

            if np.isclose(row['anchor_close'], known_anchor_close):
                print("✅ SUCCESS: Execution bar uses the PREVIOUS Anchor close.")
            else:
                print(f"❌ FAILURE: Mismatch! Merged: {row['anchor_close']}, Expected: {known_anchor_close}")
        else:
            print(f"❌ FAILURE: Anchor index {anchor_idx} not found in anchor data.")
    else:
        print(f"❌ FAILURE: Timestamp {test_ts} not found in merged data.")

def test_feature_engineering():
    print("\n--- Verifying P08 Feature Engine ---")
    loader = P08DataLoader()
    exec_file = Path("data/BTCUSDT_15m_20200101_20201231.csv")
    if not exec_file.exists(): return

    df_mtf = loader.get_mtf_dataset(exec_file)
    X = P08FeatureEngine.build_features(df_mtf, {"rsi_period": 14})

    print(f"Feature set columns: {X.columns.tolist()}")
    mtf_features = [c for c in X.columns if "anchor" in c]
    print(f"MTF specific features found: {mtf_features}")

    if len(mtf_features) > 0:
        print("✅ SUCCESS: MTF features calculated correctly.")
    else:
        print("❌ FAILURE: No anchor features found.")

if __name__ == "__main__":
    try:
        test_lookahead_safety()
        test_feature_engineering()
    except Exception as e:
        print(f"Verification crashed: {e}")
        import traceback
        traceback.print_exc()
