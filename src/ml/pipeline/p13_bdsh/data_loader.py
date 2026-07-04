import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

# Ensure project root is in sys.path for DataManager
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_data(tickers: List[str], vix_symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Downloads historical data for tickers and VIX using DataManager (with caching).
    """
    dm = DataManager()

    # Convert string dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_symbols = tickers + [vix_symbol]
    logger.info(f"Fetching data for {all_symbols} from {start_date} to {end_date} via DataManager")

    ticker_dfs = {}
    for symbol in all_symbols:
        try:
            # DataManager.get_ohlcv(symbol, timeframe, start_date, end_date)
            df = dm.get_ohlcv(symbol, "1d", start_dt, end_dt)
            if df is not None and not df.empty:
                # We need High, Low, Close for ATR
                ticker_dfs[symbol] = df[["open", "high", "low", "close"]]
            else:
                logger.warning(f"No data returned for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

    if not ticker_dfs:
        return None

    # Create a MultiIndex DataFrame manually (Symbol, PriceType)
    # This is better for handling multiple columns per ticker
    combined_df = pd.concat(ticker_dfs, axis=1)
    return combined_df


def preprocess_data(data: pd.DataFrame, tickers: List[str], vix_symbol: str) -> pd.DataFrame:
    """
    Handles missing data and cleans the dataset.
    R1: Forward-fill stock prices, drop dates where VIX is missing.
    """
    clean_data = data.copy()

    # Forward-fill stock prices (all OHLC columns)
    for ticker in tickers:
        if ticker in clean_data.columns.get_level_values(0):
            clean_data[ticker] = clean_data[ticker].ffill()

    # Drop rows where VIX (Close) is missing
    if vix_symbol in clean_data.columns.get_level_values(0):
        # We check the 'close' column of the vix_symbol
        if "close" in clean_data[vix_symbol].columns:
            clean_data = clean_data.dropna(subset=[(vix_symbol, "close")])
        else:
            # If for some reason VIX only has one column or different name
            clean_data = clean_data.dropna(subset=[clean_data[vix_symbol].columns[0]], axis=0)
    else:
        logger.warning(f"{vix_symbol} not found in downloaded data.")

    return clean_data


if __name__ == "__main__":
    # Test data loader
    tickers = ["SPY", "QQQ"]
    vix = "^VIX"
    raw_data = download_data(tickers, vix, "2020-01-01", "2021-01-01")
    if raw_data is not None:
        clean_data = preprocess_data(raw_data, tickers, vix)
        print(clean_data.head())
        print(f"Shapes - Raw: {raw_data.shape}, Clean: {clean_data.shape}")
