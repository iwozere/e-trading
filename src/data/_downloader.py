import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.binance_data_downloader import BinanceDataDownloader

# Define test scenarios
TEST_SCENARIOS = {
    'symbols': ['LTCUSDT', 'BTCUSDT', 'ETHUSDT'],
    'periods': [
        {'start_date': '20220101', 'end_date': '20250707'}
    ],
    'intervals': ['5m', '15m', '1h', '4h']
}


def download_all_scenarios():
    """Download data for all combinations of symbols, periods, and intervals."""
    downloader = BinanceDataDownloader()
    total_combinations = len(TEST_SCENARIOS['symbols']) * len(TEST_SCENARIOS['periods']) * len(TEST_SCENARIOS['intervals'])
    completed = 0

    print(f"\nStarting download of {total_combinations} data files...")

    # Create dataset directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    for symbol in TEST_SCENARIOS['symbols']:
        for period in TEST_SCENARIOS['periods']:
            for interval in TEST_SCENARIOS['intervals']:
                completed += 1
                filename = f"{symbol.upper()}_{interval}_{period['start_date'].replace("-", "")}_{period['end_date'].replace("-", "")}.csv"
                filepath = os.path.join('dataset', filename)

                if not os.path.exists(filepath):
                    print(f"\nDownloading {completed}/{total_combinations}: {filename}")
                    try:
                        downloader.download_historical_data(
                            symbol=symbol,
                            interval=interval,
                            start_date=period['start_date'],
                            end_date=period['end_date'],
                            save_to_csv=filepath
                        )
                        print(f"Successfully downloaded {filename}")
                    except Exception as e:
                        print(f"Error downloading {filename}: {str(e)}")
                else:
                    print(f"\nSkipping {completed}/{total_combinations}: {filename} (already exists)")

    print("\nDownload process completed!")

if __name__ == "__main__":
    # Download all data files first
    download_all_scenarios()
