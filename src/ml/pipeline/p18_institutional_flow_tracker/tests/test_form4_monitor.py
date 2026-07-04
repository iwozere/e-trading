"""Unit tests for Form4Monitor."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p18_institutional_flow_tracker.processors.form4_monitor import Form4Monitor


def _make_edgar_mock(form4_rows: list, dg_rows: list) -> MagicMock:
    mock = MagicMock()
    mock.download_form4_filings.return_value = pd.DataFrame(form4_rows) if form4_rows else pd.DataFrame()
    mock.download_13dg_filings.return_value = pd.DataFrame(dg_rows) if dg_rows else pd.DataFrame()
    return mock


def test_significant_sell_returned() -> None:
    edgar = _make_edgar_mock(
        form4_rows=[
            {
                "ticker": "AAPL",
                "issuer_cik": "100",
                "insider_name": "Tim Cook",
                "transaction_code": "S",
                "shares": 10000,
                "price_per_share": 180.0,
                "total_value_usd": 1_800_000,
                "filed_date": "2024-02-14",
            }
        ],
        dg_rows=[],
    )
    monitor = Form4Monitor(edgar_downloader=edgar, min_sale_value_usd=500_000)
    result = monitor.get_significant_sells(as_of_date=date(2024, 2, 14))
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "AAPL"


def test_small_sale_filtered_out() -> None:
    edgar = _make_edgar_mock(
        form4_rows=[
            {
                "ticker": "XYZ",
                "issuer_cik": "200",
                "insider_name": "CEO",
                "transaction_code": "S",
                "shares": 100,
                "price_per_share": 10.0,
                "total_value_usd": 1_000,
                "filed_date": "2024-02-14",
            }
        ],
        dg_rows=[],
    )
    monitor = Form4Monitor(edgar_downloader=edgar, min_sale_value_usd=500_000)
    result = monitor.get_significant_sells(as_of_date=date(2024, 2, 14))
    assert result.empty


def test_buy_transaction_excluded() -> None:
    edgar = _make_edgar_mock(
        form4_rows=[
            {
                "ticker": "MSFT",
                "issuer_cik": "300",
                "insider_name": "CFO",
                "transaction_code": "P",
                "shares": 5000,
                "price_per_share": 400.0,
                "total_value_usd": 2_000_000,
                "filed_date": "2024-02-14",
            }
        ],
        dg_rows=[],
    )
    monitor = Form4Monitor(edgar_downloader=edgar, min_sale_value_usd=500_000)
    result = monitor.get_significant_sells(as_of_date=date(2024, 2, 14))
    assert result.empty


def test_13dg_amendments_returned() -> None:
    edgar = _make_edgar_mock(
        form4_rows=[],
        dg_rows=[
            {
                "cik": "400",
                "entity_name": "Acme Capital",
                "accession_number": "0001234",
                "filed_date": "2024-02-14",
                "form_type": "SC 13D/A",
            }
        ],
    )
    monitor = Form4Monitor(edgar_downloader=edgar)
    result = monitor.get_13dg_drops(watchlist_tickers=["AAPL"], as_of_date=date(2024, 2, 14))
    assert len(result) == 1
    assert result.iloc[0]["form_type"] == "SC 13D/A"


def test_empty_form4_returns_empty() -> None:
    edgar = _make_edgar_mock(form4_rows=[], dg_rows=[])
    monitor = Form4Monitor(edgar_downloader=edgar)
    result = monitor.get_significant_sells(as_of_date=date(2024, 2, 14))
    assert result.empty
