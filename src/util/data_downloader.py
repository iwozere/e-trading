import os
import sys
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.data.downloader.binance_data_downloader import BinanceDataDownloader

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

                # Convert date strings to datetime.datetime objects
                start_dt = datetime.datetime.strptime(period['start_date'], "%Y%m%d")
                end_dt = datetime.datetime.strptime(period['end_date'], "%Y%m%d")

                if not os.path.exists(filepath):
                    print(f"\nDownloading {completed}/{total_combinations}: {filename}")
                    try:
                        df = downloader.get_ohlcv(
                            symbol=symbol,
                            interval=interval,
                            start_date=start_dt,
                            end_date=end_dt)
                        downloader.save_data(df, symbol, interval, start_dt, end_dt, "data")
                        print(f"Successfully downloaded {filename}")
                    except Exception as e:
                        print(f"Error downloading {filename}: {str(e)}")
                else:
                    print(f"\nSkipping {completed}/{total_combinations}: {filename} (already exists)")

    print("\nDownload process completed!")

if __name__ == "__main__":
    # Download all data files first
    download_all_scenarios()
