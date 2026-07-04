"""
Form 4 Monitor

Surfaces significant insider sell transactions (Form 4) and Schedule 13D/G
ownership stake drops filed on a given date.  Acts as the near-real-time
signal layer of the pipeline — these filings arrive within 2 business days
of the transaction, independent of the quarterly 13F cadence.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SALE_CODES = {"S", "S-"}


class Form4Monitor:
    """
    Downloads and filters Form 4 and Schedule 13D/G filings for a given date,
    returning only significant sell-side events.
    """

    def __init__(
        self,
        edgar_downloader: EdgarDownloader | None = None,
        min_sale_value_usd: int = 500_000,
    ):
        """
        Args:
            edgar_downloader: Shared EdgarDownloader instance. Created fresh if None.
            min_sale_value_usd: Minimum total USD value of a Form 4 sale transaction
                to include in output. Default $500K filters out token-size disposals.
        """
        self._edgar = edgar_downloader or EdgarDownloader()
        self._min_value = min_sale_value_usd

    def get_significant_sells(
        self,
        as_of_date: date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return significant insider sell transactions filed on a given date.

        Args:
            as_of_date: Date to check. Defaults to yesterday.
            force_refresh: Re-download even if cached.

        Returns:
            DataFrame with columns: ticker, issuer_cik, insider_name,
            transaction_code, shares, price_per_share, total_value_usd, filed_date.
            Filtered to sales ≥ min_sale_value_usd. Empty if none found.
        """
        target = as_of_date or (datetime.now().date() - timedelta(days=1))
        df = self._edgar.download_form4_filings(as_of_date=target, force=force_refresh)

        if df.empty or "total_value_usd" not in df.columns:
            return pd.DataFrame()

        mask = df["transaction_code"].isin(list(_SALE_CODES)) & (df["total_value_usd"] >= self._min_value)
        filtered = df[mask]
        significant = pd.DataFrame(filtered)
        significant = significant.sort_values("total_value_usd", ascending=False).reset_index(drop=True)

        _logger.info(
            "Form 4 monitor: %d significant sell transactions on %s (≥$%s)",
            len(significant),
            target,
            f"{self._min_value:,}",
        )
        return significant

    def get_13dg_drops(
        self,
        watchlist_tickers: List[str],
        as_of_date: date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return Schedule 13D/G amendments filed on a given date for watchlist stocks.

        Ownership-drop amendments (13D/A, 13G/A) are particularly relevant when
        the filer is reducing a previously large stake.

        ``watchlist_tickers`` is reserved for future fuzzy-match filtering against
        entity names in the filing. Currently all amendments for the date are returned.

        Args:
            watchlist_tickers: Tickers already flagged by the exit screener.
                Only 13D/G amendments whose entity_name overlaps with these tickers
                (fuzzy match via company name) are returned. Pass an empty list to
                return all amendments.
            as_of_date: Date to check. Defaults to yesterday.
            force_refresh: Re-download even if cached.

        Returns:
            DataFrame with columns: cik, entity_name, accession_number, filed_date,
            form_type. Empty if none found.
        """
        del watchlist_tickers  # fuzzy-match filtering deferred to Phase 5
        target = as_of_date or (datetime.now().date() - timedelta(days=1))
        df = self._edgar.download_13dg_filings(as_of_date=target, force=force_refresh)

        if df.empty:
            return pd.DataFrame()

        # Filter to amendment types (stake reductions)
        filtered2 = df[df["form_type"].isin(["SC 13D/A", "SC 13G/A"])]
        amendments = pd.DataFrame(filtered2)  # explicit construction avoids typing ambiguity

        _logger.info("13D/G monitor: %d amendments on %s", len(amendments), target)
        return amendments
