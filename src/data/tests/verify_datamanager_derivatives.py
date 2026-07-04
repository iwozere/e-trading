import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.getcwd())

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main():
    dm = DataManager()

    symbol = "BTCUSDT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)

    print(f"\n--- Testing DataManager Derivatives Data for {symbol} ---")

    # 1. Test Funding Rate
    print("\n1. Testing get_funding_rate...")
    fr_df = dm.get_funding_rate(symbol, start_date, end_date)
    if not fr_df.empty:
        print(f"✅ Received {len(fr_df)} rows of funding rate data.")
        print(fr_df.head(2))
    else:
        print("❌ Funding rate data is empty.")

    # 2. Test Open Interest
    print("\n2. Testing get_open_interest...")
    oi_df = dm.get_open_interest(symbol, "1h", start_date, end_date)
    if not oi_df.empty:
        print(f"✅ Received {len(oi_df)} rows of OI data.")
        print(oi_df.head(2))
    else:
        print("❌ OI data is empty.")

    # 3. Test Long/Short Ratio
    print("\n3. Testing get_long_short_ratio...")
    ls_df = dm.get_long_short_ratio(symbol, "1h", start_date, end_date)
    if not ls_df.empty:
        print(f"✅ Received {len(ls_df)} rows of L/S ratio data.")
        print(ls_df.head(2))
    else:
        print("❌ L/S ratio data is empty.")

    # 4. Test Cache (run again)
    print("\n4. Testing Cache (should be cache hits now)...")
    fr_df2 = dm.get_funding_rate(symbol, start_date, end_date)
    if not fr_df2.empty:
        print(f"✅ Received {len(fr_df2)} rows from cache.")


if __name__ == "__main__":
    main()
