"""Stage 0 — Universe loader: Russell 3000 equities + top-20 crypto."""

from pathlib import Path
from typing import List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.russell3000_downloader import Russell3000Downloader
from src.ml.pipeline.p05_ai_selector.config import CRYPTO_TICKERS
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class UniverseLoader:
    """Builds the full screening universe from Russell 3000 + crypto."""

    def load(self) -> List[str]:
        """
        Return deduplicated list of ticker symbols.

        Returns:
            Sorted list of ticker strings (equity + crypto).
        """
        dl = Russell3000Downloader()
        equity_df = dl.load()
        equity_tickers: List[str] = equity_df["ticker"].dropna().tolist()

        all_tickers = list(dict.fromkeys(equity_tickers + CRYPTO_TICKERS))
        all_tickers.sort()

        _logger.info(
            "Universe loaded: %d equities + %d crypto = %d total",
            len(equity_tickers),
            len(CRYPTO_TICKERS),
            len(all_tickers),
        )
        return all_tickers
