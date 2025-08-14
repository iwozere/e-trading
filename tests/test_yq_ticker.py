#!/usr/bin/env python3
"""
Test script to debug YQ ticker issue
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import yfinance as yf
import numpy as np
import pandas as pd
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

def test_yq_ticker():
    """Test the YQ ticker specifically"""
    print("Testing YQ ticker...")

    try:
        # Download data
        df = yf.download("YQ", period="2y", interval="1d")
        print(f"Downloaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        print(f"Last few rows:")
        print(df.tail())

        # Check for NaN values
        print(f"NaN counts per column:")
        print(df.isna().sum())

        # Check data types
        print(f"Data types:")
        print(df.dtypes)

        # Clean data
        df_clean = df.dropna()
        print(f"After dropna: {df_clean.shape}")

        if df_clean.empty:
            print("No data after cleaning!")
            return

        # Convert to arrays
        high = df_clean['High'].values.astype(float)
        low = df_clean['Low'].values.astype(float)
        close = df_clean['Close'].values.astype(float)
        volume = df_clean['Volume'].values.astype(float)

        print(f"Array lengths: high={len(high)}, low={len(low)}, close={len(close)}, volume={len(volume)}")

        # Check for NaN in arrays
        print(f"NaN in arrays: high={np.isnan(high).sum()}, low={np.isnan(low).sum()}, close={np.isnan(close).sum()}, volume={np.isnan(volume).sum()}")

        # Check for infinite values
        print(f"Infinite in arrays: high={np.isinf(high).sum()}, low={np.isinf(low).sum()}, close={np.isinf(close).sum()}, volume={np.isinf(volume).sum()}")

        # Check array shapes
        print(f"Array shapes: high={high.shape}, low={low.shape}, close={close.shape}, volume={volume.shape}")

        # Try to find the minimum length
        min_length = min(len(high), len(low), len(close), len(volume))
        print(f"Minimum length: {min_length}")

        if min_length < 50:
            print("Insufficient data!")
            return

        # Truncate arrays
        high = high[-min_length:]
        low = low[-min_length:]
        close = close[-min_length:]
        volume = volume[-min_length:]

        print(f"After truncation: all arrays have length {len(close)}")

        # Test TA-Lib
        import talib
        try:
            rsi = talib.RSI(close, timeperiod=14)
            print(f"RSI calculation successful, length: {len(rsi)}")
            print(f"RSI values (last 5): {rsi[-5:]}")
        except Exception as e:
            print(f"RSI calculation failed: {e}")
            return

        print("YQ ticker test completed successfully!")

    except Exception as e:
        logger.exception("Error testing YQ ticker: %s", str(e))

if __name__ == "__main__":
    test_yq_ticker()