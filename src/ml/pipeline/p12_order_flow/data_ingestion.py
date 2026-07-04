import sys

# Ensure project root is in sys.path
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.ml.pipeline.p12_order_flow.config import OrderFlowConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class OrderFlowDataIngestor:
    """
    Handles fetching and merging OHLCV with crypto derivative data.
    """

    def __init__(self, config: OrderFlowConfig, data_manager: DataManager | None = None):
        self.config = config
        self.dm = data_manager or DataManager()

    def fetch_unified_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetches OHLCV, Funding Rate, OI, and L/S ratio for a symbol and merges them.
        """
        start_date = self.config.get_start_date()
        end_date = self.config.get_end_date()

        _logger.info("Fetching unified data for %s from %s to %s", symbol, start_date, end_date)

        # 1. Fetch OHLCV (the primary index)
        ohlcv = self.dm.get_ohlcv(symbol, self.config.timeframe, start_date, end_date)
        if ohlcv is None or ohlcv.empty:
            _logger.error("No OHLCV data for %s", symbol)
            return pd.DataFrame()

        # 2. Fetch Derivatives
        funding = self.dm.get_funding_rate(symbol, start_date, end_date)
        oi = self.dm.get_open_interest(symbol, self.config.oi_interval, start_date, end_date)
        ls_ratio = self.dm.get_long_short_ratio(symbol, self.config.ls_ratio_interval, start_date, end_date)

        # 3. Join logic
        # We start with OHLCV as base
        unified = ohlcv.copy()

        # Join Funding Rate (typically 8h) - we forward fill
        if not funding.empty:
            funding = funding[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
            unified = unified.join(funding, how="left")
            unified["funding_rate"] = unified["funding_rate"].ffill()
        else:
            _logger.warning("No funding rate data for %s", symbol)
            unified["funding_rate"] = 0.0

        # Join Open Interest
        if not oi.empty:
            oi = oi[["sumOpenInterest", "sumOpenInterestValue"]].rename(
                columns={"sumOpenInterest": "oi_base", "sumOpenInterestValue": "oi_value"}
            )
            unified = unified.join(oi, how="left")
            # OI might have gaps if interval is different, fill gaps
            unified["oi_base"] = unified["oi_base"].ffill()
            unified["oi_value"] = unified["oi_value"].ffill()
        else:
            _logger.warning("No OI data for %s", symbol)
            unified["oi_base"] = 0.0
            unified["oi_value"] = 0.0

        # Join Long/Short Ratio
        if not ls_ratio.empty:
            ls_ratio = ls_ratio[["longShortRatio", "longAccount", "shortAccount"]].rename(
                columns={"longShortRatio": "ls_ratio", "longAccount": "ls_long_acc", "shortAccount": "ls_short_acc"}
            )
            unified = unified.join(ls_ratio, how="left")
            unified["ls_ratio"] = unified["ls_ratio"].ffill()
        else:
            _logger.warning("No L/S ratio data for %s", symbol)
            unified["ls_ratio"] = 1.0  # Neutral

        # Final cleanup: fill any remaining NaNs at the beginning if any
        unified = unified.ffill().bfill()

        _logger.info("Successfully unified data for %s: %d rows", symbol, len(unified))
        return unified
