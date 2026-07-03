"""
P20 Kestrel — Alias builder.

Rebuilds k20_company_aliases from fundamentals data and EDGAR names.
Called weekly by alias_refresh job.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.db.repos import (
    get_active_tickers,
    upsert_aliases,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_STRIP_SUFFIXES = re.compile(
    r"\b(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|LP|PLC|SE|NV|SA|AG|Co\.?|Holdings?|Group)\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


def normalize_alias(text: str) -> str:
    """
    Normalize an alias for GDELT matching (lowercase, strip legal suffixes, collapse spaces).

    Args:
        text: Raw company name or alias.

    Returns:
        Normalized alias string.
    """
    s = _STRIP_SUFFIXES.sub(" ", text)
    s = _WHITESPACE.sub(" ", s).strip().lower()
    return s


def _build_aliases_for_ticker(
    ticker: str,
    company_name: Optional[str],
    edgar_name: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Generate alias rows for one ticker from available name sources.

    Args:
        ticker: Ticker symbol.
        company_name: Primary company name from fundamentals.
        edgar_name: Name from EDGAR filings (may differ).

    Returns:
        List of alias dicts suitable for k20_company_aliases upsert.
    """
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []

    def _add(alias_text: str, alias_type: str) -> None:
        alias_text = alias_text.strip()
        if not alias_text or alias_text in seen:
            return
        seen.add(alias_text)
        normalized = normalize_alias(alias_text)
        rows.append({
            "ticker": ticker,
            "alias": alias_text,
            "alias_type": alias_type,
            "normalized_alias": normalized,
        })

    if company_name:
        _add(company_name, "legal_name")
        # Short name: strip legal suffix
        short = _STRIP_SUFFIXES.sub("", company_name).strip()
        if short and short != company_name:
            _add(short, "short_name")

    if edgar_name and edgar_name != company_name:
        _add(edgar_name, "legal_name")

    # Ticker itself as a fallback alias (low precision, but useful for small caps)
    _add(ticker, "short_name")

    return rows


def run() -> Dict[str, Any]:
    """
    Rebuild k20_company_aliases for all active universe tickers.

    Returns:
        Summary dict with tickers_processed, aliases_upserted.
    """
    import asyncio
    from src.common.fundamentals import get_fundamentals_unified

    tickers = get_active_tickers()
    _logger.info("Alias builder processing %d tickers", len(tickers))

    all_alias_rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        company_name: Optional[str] = None
        edgar_name: Optional[str] = None

        try:
            fund = asyncio.run(get_fundamentals_unified(ticker))
            company_name = getattr(fund, "company_name", None) or getattr(fund, "name", None)
        except Exception:
            _logger.debug("Could not get fundamentals for %s", ticker)

        alias_rows = _build_aliases_for_ticker(ticker, company_name, edgar_name)
        all_alias_rows.extend(alias_rows)

    upserted = upsert_aliases(all_alias_rows)
    _logger.info("Alias builder: %d tickers → %d aliases upserted", len(tickers), upserted)
    return {"tickers_processed": len(tickers), "aliases_upserted": upserted}
