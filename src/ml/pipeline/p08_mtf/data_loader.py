import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p07_combined.data_loader import P07DataLoader

_logger = setup_logger(__name__)

class P08DataLoader(P07DataLoader):
    """
    P08 Data Loader: Extends P07 with Multi-Timeframe (MTF) capabilities.
    Safely joins Execution TF with Anchor TF using look-ahead protection.
    """

    # Primary mapping for MTF Anchor/Execution pairs
    TF_MAPPING = {
        "5m": "1h",
        "15m": "4h",
        "30m": "4h",
        "1h": "1d",
        "4h": "1d"
    }

    def __init__(self, data_root: Path = Path("data")):
        super().__init__(data_root)

    def get_anchor_tf(self, execution_tf: str) -> Optional[str]:
        """Returns the mapped anchor timeframe for a given execution timeframe."""
        return self.TF_MAPPING.get(execution_tf)

    def find_anchor_file(self, ticker: str, anchor_tf: str, start_date: str, end_date: str) -> Optional[Path]:
        """Attempts to find a matching anchor data file in the data root."""
        # Schema: {ticker}_{timeframe}_{start}_{end}.csv
        pattern = f"{ticker}_{anchor_tf}_{start_date}_{end_date}.csv"
        path = self.data_root / pattern
        if path.exists():
            return path

        # Fallback: find any file for ticker/tf that covers the range (simplified)
        for f in self.data_root.glob(f"{ticker}_{anchor_tf}_*.csv"):
            return f # For now, return the first match for simplicity

        return None

    def merge_mtf(self, df_exec: pd.DataFrame, df_anchor: pd.DataFrame) -> pd.DataFrame:
        """
        Look-ahead safe join between Execution and Anchor TFs.
        Uses the Anchor's PREVIOUS close to ensure point-in-time validity.
        """
        # Ensure indices are sorted
        df_exec = df_exec.sort_index()
        df_anchor = df_anchor.sort_index()

        # Prefix anchor columns to avoid collisions
        anchor_cols = {col: f"anchor_{col}" for col in df_anchor.columns}
        df_anchor_renamed = df_anchor.rename(columns=anchor_cols)

        # Shift anchor data by 1 bar to prevent look-ahead bias
        # This ensures that at any 'timestamp' in df_exec, we only see anchor data
        # that was 'knowable' (fully completed) before that timestamp.
        df_anchor_safe = df_anchor_renamed.shift(1)

        # Point-in-time join using merge_asof
        # direction='backward' matches the execution bar with the nearest preceding anchor bar
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
        Main entry: Loads Execution file, finds Anchor, merges them, and adds Macro features.
        """
        ticker, timeframe, start, end = self.parse_filename(exec_path)
        if not ticker:
            raise ValueError(f"Could not parse filename: {exec_path.name}")

        _logger.info("Loading P08 MTF dataset for %s %s [%s_%s]", ticker, timeframe, start, end)

        # 1. Load Execution Data
        df_exec = self.load_ohlcv(exec_path)

        # 2. Add Macro Features (Inherited from P07)
        vix = self.load_vix()
        btc_mc = self.load_btc_marketcap()

        if not vix.empty:
            df_exec = df_exec.join(vix, how="left")
            df_exec["vix"] = df_exec["vix"].ffill()

        if not btc_mc.empty:
            df_exec = df_exec.join(btc_mc, how="left")
            df_exec["btc_mc"] = df_exec["btc_mc"].ffill()

        # 3. Handle MTF Join
        anchor_tf = self.get_anchor_tf(timeframe)
        if anchor_tf:
            anchor_file = self.find_anchor_file(ticker, anchor_tf, start, end)
            if anchor_file:
                _logger.info("Merging with Anchor TF: %s (%s)", anchor_tf, anchor_file.name)
                df_anchor = self.load_ohlcv(anchor_file)
                df_exec = self.merge_mtf(df_exec, df_anchor)
            else:
                _logger.warning("No matching Anchor file found for %s %s. Proceeding without MTF features.", ticker, anchor_tf)
        else:
            _logger.info("No Anchor TF mapped for %s. Skipping MTF join.", timeframe)

        return df_exec

if __name__ == "__main__":
    # Quick test
    loader = P08DataLoader()
    # Test on a known file if possible
    data_dir = Path("data")
    test_file = list(data_dir.glob("BTCUSDT_15m_*.csv"))
    if test_file:
        df = loader.get_mtf_dataset(test_file[0])
        print(f"Loaded MTF columns: {[c for c in df.columns if 'anchor' in c]}")
