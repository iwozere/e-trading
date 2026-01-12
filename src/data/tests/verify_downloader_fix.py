import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Setup logging to see DEBUG logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_downloader_initialization():
    from src.data.data_manager import get_provider_selector

    print("\n--- Testing ProviderSelector Singleton and Lazy Loading ---")

    # 1. Get selector twice, should be the same instance
    print("\n[Step 1] Getting ProviderSelector twice...")
    selector1 = get_provider_selector()
    selector2 = get_provider_selector()

    print(f"Selector 1: {selector1}")
    print(f"Selector 2: {selector2}")
    print(f"Same instance: {selector1 is selector2}")

    # 2. Check downloaders dictionary, should be empty initially
    print(f"\n[Step 2] Initial downloaders in selector: {list(selector1.downloaders.keys())}")

    # 3. Classify a symbol (should NOT initialize any downloaders)
    print("\n[Step 3] Classifying symbol 'BTCUSDT' (Crypto)...")
    asset_type = selector1.classify_symbol("BTCUSDT")
    print(f"Asset Type: {asset_type}")
    print(f"Downloaders after classification: {list(selector1.downloaders.keys())}")

    # 4. Get a downloader (should initialize ONLY that downloader at DEBUG level)
    print("\n[Step 4] Getting Binance downloader...")
    downloader = selector1.get_best_downloader("BTCUSDT", "1h")
    print(f"Downloader: {downloader}")
    print(f"Downloaders now: {list(selector1.downloaders.keys())}")

    # 5. Get another downloader
    print("\n[Step 5] Getting Yahoo downloader for a stock...")
    # 'AAPL' has primary 'yahoo' for 1d
    selector1.get_best_downloader("AAPL", "1d")
    print(f"Downloaders now: {list(selector1.downloaders.keys())}")

if __name__ == "__main__":
    test_downloader_initialization()
