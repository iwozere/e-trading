import io
from typing import List

import pandas as pd
import requests
import yfinance as yf
from pathlib import Path


def get_circulating_supply(ticker):
    """
    Get the circulating supply for a cryptocurrency from CoinGecko.
    Args:
        ticker: The ticker symbol (e.g., 'BTCUSDT')
    Returns:
        float: The circulating supply or a default value if not found
    """
    # Map common tickers to CoinGecko IDs
    mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "LTC": "litecoin",
        "XRP": "xrp",
        "BCH": "bitcoin-cash",
        "XLM": "stellar",
        "ADA": "cardano",
        "DOT": "polkadot",
    }

    # Extract base symbol (remove USDT)
    base_symbol = ticker.replace("USDT", "").replace("USDC", "").replace("USDC.e", "")

    # Try to get the CoinGecko ID from the mapping
    coin_id = mapping.get(base_symbol)
    if coin_id is None:
        # If not found in mapping, try to use the base symbol
        coin_id = base_symbol.lower()

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["market_data"]["circulating_supply"]
        else:
            print(f"Error fetching supply for {ticker}: HTTP {response.status_code}")
            return 1e8  # Default value
    except Exception as e:
        print(f"Error fetching supply for {ticker}: {str(e)}")
        return 1e8  # Default value


def _load_tickers_from_csv(filename: str) -> List[str]:
    """
    Load tickers from a CSV file in the data/tickers directory.

    Args:
        filename: Name of the CSV file (e.g., 'all_us_tickers.csv')

    Returns:
        List of ticker symbols
    """
    try:
        # Get the path to the data/tickers directory relative to this file
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[1]
        csv_path = project_root / 'src' / 'data' / 'tickers' / filename

        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}")
            return []

        df = pd.read_csv(csv_path)
        if 'ticker' in df.columns:
            return df['ticker'].tolist()
        else:
            print(f"Warning: No 'ticker' column found in {filename}")
            return []

    except Exception as e:
        print(f"Error loading tickers from {filename}: {str(e)}")
        return []


# Download SIX ticker list
def get_six_tickers():
    url = "https://www.six-group.com/sheldon/equity_issuers/v1/equity_issuers.csv"
    response = requests.get(url, verify=False)
    df = pd.read_csv(io.StringIO(response.text), on_bad_lines="skip", header=0, sep=";")
    tickers = df["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]  # Convert for yfinance
    return tickers


# Download S&P 500 ticker list
def get_sp500_tickers_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    tickers = tables[0]["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]  # Convert for yfinance
    return tickers


# S&P400 midcap with pandas_datareader and Wikipedia
def get_sp_midcap_wikipedia():
    """
    Get the list of tickers S&P MidCap 400 from Wikipedia
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

    tables = pd.read_html(url)
    tickers = tables[0]["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]  # Convert for yfinance
    return tickers


# S&P400 midcap with yfinance
def get_sp_midcap_yfinance():
    """
    Get theS&P MidCap 400 tickers using yfinance
    """
    ticker = "^SP400"
    sp_midcap = yf.Ticker(ticker)

    try:
        # Попытка получить компоненты индекса
        components = sp_midcap.components
        return list(components.index)
    except:
        print("Could not get components from yfinance.")
        return []


def get_all_us_tickers():
    """Load all US tickers from CSV file"""
    return _load_tickers_from_csv('all_us_tickers.csv')


def get_us_delisted_tickers():
    """Load delisted US tickers from CSV file"""
    return _load_tickers_from_csv('us_delisted_tickers.csv')


def get_us_small_cap_tickers():
    """Load US small cap tickers from CSV file"""
    return _load_tickers_from_csv('us_small_cap_tickers.csv')


def get_us_medium_cap_tickers():
    """Load US medium cap tickers from CSV file"""
    return _load_tickers_from_csv('us_medium_cap_tickers.csv')


def get_us_large_cap_tickers():
    """Load US large cap tickers from CSV file"""
    return _load_tickers_from_csv('us_large_cap_tickers.csv')


# Example usage
if __name__ == "__main__":
    print("S&P400 mid cap wiki:")
    print(get_sp_midcap_wikipedia())
    print("S&P400 mid cap yfinance:")
    print(get_sp_midcap_yfinance())
    print("US Small Cap Tickers:")
    print(get_us_small_cap_tickers())
    print("\nUS Medium Cap Tickers:")
    print(get_us_medium_cap_tickers())
    print("\nUS Large Cap Tickers:")
    print(get_us_large_cap_tickers())
    print("\nCirculating Supply:")
    print(get_circulating_supply("bitcoin"))
    print(get_circulating_supply("ethereum"))
    print(get_circulating_supply("litecoin"))
    print(get_circulating_supply("xrp"))
