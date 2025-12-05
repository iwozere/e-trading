import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# --------------------------------------------------------------
# Tradier Options Data Downloader
# Mirrors the style of your yahoo_data_downloader.py
# --------------------------------------------------------------

class TradierDownloader:
    BASE_URL = "https://api.tradier.com/v1"

    def __init__(self, api_key: str, rate_limit_sleep: float = 0.3):
        self.api_key = api_key
        self.rate_limit_sleep = rate_limit_sleep

        self.session = requests.Session()
        self.session.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # ----------------------------------------------------------
    # Perform safe GET requests
    # ----------------------------------------------------------
    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 429:
                logging.warning("Rate limit hit. Sleeping...")
                time.sleep(2)
                return self._get(endpoint, params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    # ----------------------------------------------------------
    # Get available option expirations for a ticker
    # ----------------------------------------------------------
    def get_expirations(self, ticker: str) -> List[str]:
        data = self._get(f"/markets/options/expirations", {"symbol": ticker})
        if not data or "expirations" not in data or "date" not in data["expirations"]:
            return []
        return data["expirations"]["date"]

    # ----------------------------------------------------------
    # Download entire option chain for a specific expiration
    # ----------------------------------------------------------
    def get_chain(self, ticker: str, expiration: str) -> List[dict]:
        data = self._get(f"/markets/options/chains", {"symbol": ticker, "expiration": expiration})
        if not data or "options" not in data or "option" not in data["options"]:
            return []
        return data["options"]["option"]

    # ----------------------------------------------------------
    # Save JSON to disk
    # ----------------------------------------------------------
    def _save(self, path: str, data: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ----------------------------------------------------------
    # Main download routine for a ticker
    # ----------------------------------------------------------
    def download_ticker(self, ticker: str, out_dir: str):
        logging.info(f"Processing {ticker}...")

        expirations = self.get_expirations(ticker)
        if not expirations:
            logging.warning(f"No expirations for {ticker}")
            return

        for exp in expirations:
            chain = self.get_chain(ticker, exp)
            if not chain:
                continue

            path = os.path.join(out_dir, ticker.upper(), f"options_{exp}.json")
            self._save(path, chain)
            logging.info(f"Saved {ticker} {exp} ({len(chain)} contracts)")

            time.sleep(self.rate_limit_sleep)

    # ----------------------------------------------------------
    # Batch run
    # ----------------------------------------------------------
    def download_universe(self, tickers: List[str], out_dir: str):
        for t in tickers:
            try:
                self.download_ticker(t, out_dir)
            except Exception as e:
                logging.error(f"Failed {t}: {e}")
            time.sleep(self.rate_limit_sleep)


# --------------------------------------------------------------
# Example usage
# --------------------------------------------------------------
if __name__ == "__main__":
    API_KEY = os.environ.get("TRADIER_API", "YOUR_KEY_HERE")

    downloader = TradierDownloader(API_KEY)

    # Example small universe (you will pass your filtered list)
    universe = ["AAPL", "AMD", "TSLA"]

    downloader.download_universe(universe, out_dir="./data/tradier/")
