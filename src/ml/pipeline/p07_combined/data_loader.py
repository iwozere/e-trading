import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
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

        if not vix.empty:
            ohlcv = ohlcv.join(vix, how="left")
            ohlcv["vix"] = ohlcv["vix"].ffill()

        if not btc_mc.empty:
            ohlcv = ohlcv.join(btc_mc, how="left")
            ohlcv["btc_mc"] = ohlcv["btc_mc"].ffill()

        return ohlcv

    # ------------------------------------------------------------------
    # Multi-Timeframe (MTF) support — merged from P08DataLoader
    # ------------------------------------------------------------------

    # Maps execution timeframe → anchor timeframe for MTF context
    TF_MAPPING: Dict[str, str] = {
        "5m":  "1h",
        "15m": "4h",
        "30m": "4h",
        "1h":  "1d",
        "4h":  "1d",
    }

    def get_anchor_tf(self, execution_tf: str) -> Optional[str]:
        """Returns the mapped anchor timeframe for a given execution timeframe."""
        return self.TF_MAPPING.get(execution_tf)

    def find_anchor_file(self, ticker: str, anchor_tf: str, start_date: str, end_date: str) -> Optional[Path]:
        """Finds a matching anchor data file in the data root."""
        path = self.data_root / f"{ticker}_{anchor_tf}_{start_date}_{end_date}.csv"
        if path.exists():
            return path
        for f in self.data_root.glob(f"{ticker}_{anchor_tf}_*.csv"):
            return f
        return None

    def merge_mtf(self, df_exec: pd.DataFrame, df_anchor: pd.DataFrame) -> pd.DataFrame:
        """
        Look-ahead safe join between execution and anchor timeframes.

        Shifts anchor data by 1 bar then uses merge_asof(direction='backward')
        so each execution bar only sees anchor bars that were fully closed before it.
        """
        df_exec = df_exec.sort_index()
        df_anchor = df_anchor.sort_index()

        anchor_cols = {col: f"anchor_{col}" for col in df_anchor.columns}
        df_anchor_renamed = df_anchor.rename(columns=anchor_cols)

        # 1-bar shift ensures point-in-time validity (no look-ahead on anchor close)
        df_anchor_safe = df_anchor_renamed.shift(1)

        merged = pd.merge_asof(
            df_exec,
            df_anchor_safe,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        return merged

    def get_mtf_dataset(self, exec_path: Path) -> pd.DataFrame:
        """
        Loads execution file, finds matching anchor timeframe file, merges them,
        and adds macro features (VIX, BTC market cap).
        """
        ticker, timeframe, start, end = self.parse_filename(exec_path)
        if not ticker:
            raise ValueError(f"Could not parse filename: {exec_path.name}")

        _logger.info("Loading MTF dataset for %s %s [%s_%s]", ticker, timeframe, start, end)

        df_exec = self.get_merged_dataset(exec_path)

        anchor_tf = self.get_anchor_tf(timeframe)
        if anchor_tf:
            anchor_file = self.find_anchor_file(ticker, anchor_tf, start, end)
            if anchor_file:
                _logger.info("Merging with anchor TF: %s (%s)", anchor_tf, anchor_file.name)
                df_anchor = self.load_ohlcv(anchor_file)
                df_exec = self.merge_mtf(df_exec, df_anchor)
            else:
                _logger.warning(
                    "No anchor file found for %s %s. Proceeding without MTF features.", ticker, anchor_tf
                )
        else:
            _logger.info("No anchor TF mapped for %s. Skipping MTF join.", timeframe)

        return df_exec
