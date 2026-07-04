import os
import sys
from datetime import datetime, timedelta

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.downloader.binance_data_downloader import BinanceDataDownloader


def test_derivatives_methods():
    print("Initializing BinanceDataDownloader...")
    downloader = BinanceDataDownloader()

    symbol = "BTCUSDT"

    # We will test grabbing the last 2 days of data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=2)

    print("\n=======================================================")
    print(f"Testing derivatives methods for {symbol}")
    print(f"Time range: {start_date} to {end_date}")
    print("=======================================================\n")

    # Test Funding Rate
    print("1. Testing get_funding_rate_history...")
    try:
        df_funding = downloader.get_funding_rate_history(
            symbol=symbol, start_date=start_date, end_date=end_date, limit=100
        )
        if not df_funding.empty:
            print(f"SUCCESS: Retrieved {len(df_funding)} records.")
            print("Preview:")
            print(df_funding.head(2))
        else:
            print("WARNING: Returned empty DataFrame.")
    except Exception as e:
        print(f"ERROR testing funding rate: {e}")

    # Test Open Interest
    print("\n2. Testing get_open_interest_history (4h period)...")
    try:
        df_oi = downloader.get_open_interest_history(
            symbol=symbol, period="4h", start_date=start_date, end_date=end_date, limit=100
        )
        if not df_oi.empty:
            print(f"SUCCESS: Retrieved {len(df_oi)} records.")
            print("Preview:")
            print(df_oi.head(2))
        else:
            print("WARNING: Returned empty DataFrame.")
    except Exception as e:
        print(f"ERROR testing open interest: {e}")

    # Test Long/Short Ratio
    print("\n3. Testing get_long_short_ratio (4h period)...")
    try:
        df_ls = downloader.get_long_short_ratio(
            symbol=symbol, period="4h", start_date=start_date, end_date=end_date, limit=100
        )
        if not df_ls.empty:
            print(f"SUCCESS: Retrieved {len(df_ls)} records.")
            print("Preview:")
            print(df_ls.head(2))
        else:
            print("WARNING: Returned empty DataFrame.")
    except Exception as e:
        print(f"ERROR testing long/short ratio: {e}")


if __name__ == "__main__":
    test_derivatives_methods()
