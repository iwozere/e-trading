"""
EODHD Options Data Downloader

This module provides functionality to fetch and process options data from the EODHD API.
It includes features for fetching option chains, computing UOA metrics, and handling errors.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from config.donotshare.donotshare import EODHD_API_KEY
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


# Constants
BASE_URL = "https://eodhd.com/api/options"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds

class EODHDApiError(Exception):
    """Custom exception for EODHD API errors."""
    pass

def _make_api_request(url: str, params: Optional[Dict] = None) -> Dict:
    """
    Make an API request with retry logic and error handling.

    Args:
        url: The API endpoint URL
        params: Optional query parameters

    Returns:
        Dict containing the JSON response

    Raises:
        EODHDApiError: If the request fails after retries
    """
    headers = {
        'User-Agent': 'e-trading/1.0',
        'Accept': 'application/json'
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', RATE_LIMIT_DELAY))
                _logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            else:
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            _logger.exception(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}):")
            if attempt == MAX_RETRIES - 1:
                raise EODHDApiError(f"API request failed after {MAX_RETRIES} attempts: {str(e)}")
            time.sleep(RATE_LIMIT_DELAY * (attempt + 1))

    raise EODHDApiError("Max retries exceeded for API request")

def fetch_chain(ticker: str, date: Union[datetime, str]) -> pd.DataFrame:
    """
    Fetch option chain data for a specific ticker and date.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        date: Date for the options data (datetime or 'YYYY-MM-DD' string)

    Returns:
        DataFrame with columns:
        ['ticker', 'date', 'expiration', 'type', 'strike', 'last', 'volume', 'open_interest', 'iv']

    Example:
        >>> df = fetch_chain('AAPL', '2023-12-05')
        >>> df.head()
    """
    try:
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date

        _logger.info(f"Fetching option chain for {ticker} on {date_str}")

        url = f"{BASE_URL}/{ticker}.US"
        params = {
            'api_token': EODHD_API_KEY,
            'fmt': 'json',
            'date': date_str
        }

        data = _make_api_request(url, params)

        if not data or 'data' not in data:
            _logger.warning(f"No data returned for {ticker} on {date_str}")
            return pd.DataFrame()

        rows = []
        for exp in data.get("data", []):
            expiration = exp.get("expiration")
            options = exp.get("options", {})
            for opt_type in ["call", "put"]:
                for opt in options.get(opt_type, []):
                    if isinstance(opt, dict):
                        rows.append({
                            "ticker": ticker,
                            "date": date_str,
                            "expiration": expiration,
                            "type": opt_type,  # Use the type from the loop
                            "strike": opt.get("strike"),
                            "last": opt.get("last_trade_price"),
                            "volume": opt.get("volume", 0),
                            "open_interest": opt.get("open_interest", 0),
                            "iv": opt.get("implied_volatility"),
                        })

        if not rows:
            _logger.warning(f"No options data found for {ticker} on {date_str}")
            return pd.DataFrame()

        return pd.DataFrame(rows)

    except EODHDApiError:
        raise  # Re-raise EODHDApiError
    except Exception as e:
        _logger.error(f"Error fetching option chain for {ticker}: {str(e)}", exc_info=True)
        return pd.DataFrame()

def download_for_date(tickers: List[str], date: Union[datetime, str]) -> pd.DataFrame:
    """
    Fetch options data for multiple tickers on a specific date.

    Args:
        tickers: List of stock ticker symbols
        date: Date for the options data (datetime or 'YYYY-MM-DD' string)

    Returns:
        Combined DataFrame with options data for all tickers

    Example:
        >>> df = download_for_date(['AAPL', 'MSFT'], '2023-12-05')
    """
    if not tickers:
        _logger.warning("No tickers provided")
        return pd.DataFrame()

    if isinstance(date, datetime):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = date

    _logger.info(f"Downloading options data for {len(tickers)} tickers on {date_str}")

    frames = []
    for i, ticker in enumerate(tickers):
        try:
            _logger.debug(f"Processing {ticker} ({i+1}/{len(tickers)})")
            df = fetch_chain(ticker, date)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            _logger.error(f"Error processing {ticker}: {str(e)}")
            continue

    if frames:
        result = pd.concat(frames, ignore_index=True)
        _logger.info(f"Successfully downloaded data for {len(result['ticker'].unique())} tickers")
        return result

    _logger.warning("No data was downloaded for any ticker")
    return pd.DataFrame()

def compute_30d_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 30-day rolling statistics for options data.

    Args:
        df: DataFrame with columns ['ticker', 'date', 'type', 'volume']

    Returns:
        DataFrame with 30-day rolling statistics

    Example:
        >>> stats = compute_30d_stats(options_df)
    """
    try:
        if df.empty:
            _logger.warning("Empty DataFrame provided")
            return pd.DataFrame()

        required_columns = ['ticker', 'date', 'type', 'volume']
        if not all(col in df.columns for col in required_columns):
            _logger.error(f"Missing required columns. Expected: {required_columns}")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"])

        # Aggregate daily call/put volume per ticker
        daily = df.groupby(["ticker", "date", "type"], as_index=False)["volume"].sum()

        # Pivot call/put into columns
        pivot = daily.pivot_table(
            index=["ticker", "date"],
            columns="type",
            values="volume",
            fill_value=0
        ).reset_index()

        # Rename columns for clarity
        pivot = pivot.rename(columns={
            "call": "call_volume",
            "put": "put_volume",
            "index": "date"
        })

        # Calculate 30-day rolling statistics
        for col in ["call_volume", "put_volume"]:
            if col in pivot.columns:
                pivot[f"{col}_30d_avg"] = (
                    pivot.groupby("ticker")[col]
                    .transform(lambda x: x.rolling(30, min_periods=1).mean())
                )
                pivot[f"{col}_30d_std"] = (
                    pivot.groupby("ticker")[col]
                    .transform(lambda x: x.rolling(30, min_periods=1).std().fillna(0))
                )

        return pivot

    except Exception as e:
        _logger.error(f"Error computing 30-day statistics: {str(e)}", exc_info=True)
        return pd.DataFrame()

def compute_uoa_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Unusual Options Activity (UOA) score.

    The score is based on:
    - Call volume vs 30-day average
    - Put/Call ratio
    - Volume concentration

    Args:
        df: DataFrame with options data including volume statistics

    Returns:
        DataFrame with UOA metrics and scores

    Example:
        >>> uoa_scores = compute_uoa_score(stats_df)
    """
    try:
        if df.empty:
            return pd.DataFrame()

        required_columns = ['ticker', 'date', 'call_volume', 'put_volume']
        if not all(col in df.columns for col in required_columns):
            _logger.error(f"Missing required columns: {required_columns}")
            return pd.DataFrame()

        df = df.copy()

        # Calculate basic metrics with safe division
        df['total_volume'] = df['call_volume'] + df['put_volume']

        # Safe calculation of put/call ratio
        df['put_call_ratio'] = 0  # Default to 0 when call_volume is 0
        mask = df['call_volume'] > 0
        df.loc[mask, 'put_call_ratio'] = df['put_volume'] / df['call_volume']

        # Initialize UOA score with default value of 0
        df['uoa_score'] = 0.0

        # Calculate volume ratios if 30-day averages exist
        if 'call_volume_30d_avg' in df.columns:
            # Replace zeros to avoid division by zero
            call_avg = df['call_volume_30d_avg'].replace(0, np.nan)
            df['call_volume_ratio'] = (df['call_volume'] / call_avg).fillna(0)

            # Calculate UOA score based on call volume ratio
            df['uoa_score'] = (df['call_volume_ratio'].clip(0, 5) * 20).clip(0, 100)

            # Adjust score based on put/call ratio
            df['uoa_score'] = df.apply(
                lambda x: x['uoa_score'] * (1 - min(x.get('put_call_ratio', 1), 1) * 0.5),
                axis=1
            )

        return df.round(2)

    except Exception as e:
        _logger.error(f"Error computing UOA score: {str(e)}", exc_info=True)
        return pd.DataFrame()

def save_to_file(df: pd.DataFrame, output_dir: str, filename: str) -> bool:
    """
    Save DataFrame to a file in the specified directory.

    Args:
        df: DataFrame to save
        output_dir: Directory to save the file in
        filename: Name of the output file (with extension)

    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        ext = filepath.suffix.lower()

        if ext == '.csv':
            df.to_csv(filepath, index=False)
        elif ext in ['.pkl', '.pickle']:
            df.to_pickle(filepath)
        elif ext == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        _logger.info(f"Data saved to {filepath}")
        return True

    except ValueError as e:
        _logger.error(f"Error saving to file: {str(e)}")
        raise  # Re-raise ValueError for unsupported formats
    except Exception as e:
        _logger.exception("Error saving to file:")
        return False