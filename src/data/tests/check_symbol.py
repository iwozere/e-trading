import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Mock some dependencies if needed or just import
from src.data.data_manager import get_provider_selector

def check():
    selector = get_provider_selector()
    symbol = "BTCUSDT"
    asset_type = selector.classify_symbol(symbol)
    print(f"SYMBOL_REPLY:{symbol}:{asset_type}")

if __name__ == "__main__":
    check()
