import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root
PROJECT_ROOT = Path("c:/dev/cursor/e-trading")
sys.path.insert(0, str(PROJECT_ROOT))

# Mock environment variables if needed
os.environ["FINNHUB_API_KEY"] = "mock"

from src.data.data_manager import DataManager
from src.data.cache.fundamentals_cache import get_fundamentals_cache

def test_ttl_overrides():
    dm = DataManager()
    symbol = "AAPL"
    
    print(f"Testing TTL overrides for {symbol}...")
    
    # 1. Test with a very short TTL (should miss if cache exists)
    data_short = dm.get_fundamentals(symbol, max_age_days=0)
    print(f"Short TTL (0 days) result: {'Hit' if data_short else 'Miss (Expected)'}")
    
    # 2. Test with a long TTL (should hit if cache exists)
    data_long = dm.get_fundamentals(symbol, max_age_days=30)
    print(f"Long TTL (30 days) result: {'Hit' if data_long else 'Miss'}")
    
    if data_long:
        print("Successfully retrieved data with 30-day TTL.")
    else:
        print("Note: Cache was empty, so both were misses. Try running after a real fetch.")

if __name__ == "__main__":
    try:
        test_ttl_overrides()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
