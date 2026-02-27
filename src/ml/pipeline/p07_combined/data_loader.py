import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional
import sys

# Ensure project root is in sys.path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.data_downloader_factory import DataDownloaderFactory

_logger = setup_logger(__name__)

class P07DataLoader:
    """
    Unified Data Loader for p07_combined.
    Features:
    - Dynamic metadata extraction from filename.
    - Integration with VIX and BTC Market Cap global features.
    - Pathlib-based I/O for cross-platform compatibility.
    """

    def __init__(self, data_root: Path = Path("data")):
        self.data_root = data_root
        self.vix_path = data_root / "vix" / "vix.csv"
        self.btc_mc_path = data_root / "btc_mc" / "btc_mc.csv"

    @staticmethod
    def parse_filename(filepath: Path) -> Tuple[str, str, str, str]:
        """
        Extract ticker, timeframe, start_date, end_date from filename.
        Schema: {ticker}_{timeframe}_{start_date}_{end_date}.csv
        """
        filename = filepath.name
        # Flexible pattern: allows alphanumeric, underscores, and hyphens in segments
        # Schema: {ticker}_{timeframe}_{start_date}_{end_date}.csv
        pattern = r"^(?P<ticker>.+?)_(?P<timeframe>[^_]+)_(?P<start>\d{8})_(?P<end>\d{8})\.csv$"
        match = re.match(pattern, filename)
        if not match:
            _logger.warning("Filename %s does not match expected schema.", filename)
            return "", "", "", ""

        return (
            match.group("ticker"),
            match.group("timeframe"),
            match.group("start"),
            match.group("end")
        )

    def load_ohlcv(self, filepath: Path) -> pd.DataFrame:
        """Load local ticker OHLCV data."""
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        # Ensure UTC if not present
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()

    def load_vix(self) -> pd.DataFrame:
        """Load VIX data from data/vix/vix.csv."""
        if not self.vix_path.exists():
            _logger.warning("VIX data not found at %s. Returning empty DF.", self.vix_path)
            return pd.DataFrame()

        df = pd.read_csv(self.vix_path, parse_dates=["date"])
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df[["vix"]].sort_index()

    def load_btc_marketcap(self) -> pd.DataFrame:
        """Load BTC Market Cap data, downloading it if necessary."""
        if not self.btc_mc_path.exists():
            _logger.info("BTC Market Cap data not found. Downloading...")
            loader = DataDownloaderFactory.create_downloader("btc_mc")
            if loader:
                # Use a broad range for macro analysis
                # CoinGecko Public API is limited to 365 days of historical data.
                # Requesting a broader range causes 10012/401 errors.
                start = datetime.now() - timedelta(days=364)
                end = datetime.now()
                # Check for specialized market cap method (merged coingecko)
                if hasattr(loader, 'get_market_cap'):
                    df = loader.get_market_cap("bitcoin", start, end)
                else:
                    # Fallback for other providers (historical)
                    df = loader.get_ohlcv("bitcoin", "1d", start, end)
                if not df.empty:
                    self.btc_mc_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(self.btc_mc_path)
                    _logger.info("BTC Market Cap saved to %s", self.btc_mc_path)
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()

        df = pd.read_csv(self.btc_mc_path, parse_dates=["timestamp"])
        df.rename(columns={"close": "btc_mc"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df[["btc_mc"]].sort_index()

    def get_merged_dataset(self, filepath: Path) -> pd.DataFrame:
        """
        Main entry point: loads OHLCV and merges with global macro features.
        The join uses 'ffill' for daily macro data to align with higher frequency OHLCV.
        """
        ohlcv = self.load_ohlcv(filepath)
        vix = self.load_vix()
        btc_mc = self.load_btc_marketcap()

        # Merge VIX
        if not vix.empty:
            ohlcv = ohlcv.join(vix, how="left")
            ohlcv["vix"] = ohlcv["vix"].ffill()

        # Merge BTC Market Cap
        if not btc_mc.empty:
            ohlcv = ohlcv.join(btc_mc, how="left")
            ohlcv["btc_mc"] = ohlcv["btc_mc"].ffill()

        # Drop rows where we don't have macro data if essential,
        # but usually we just want the macro features for the model.
        return ohlcv
