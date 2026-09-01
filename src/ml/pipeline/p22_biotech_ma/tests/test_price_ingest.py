"""Tests for ingest/price_ingest.py. No live DB — repo is a MagicMock."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.price_ingest import write_daily_bars

_KNOWN_FROM = datetime(2026, 8, 31, tzinfo=timezone.utc)


def test_write_daily_bars_writes_price_row():
    repo = MagicMock()
    bars = [{"ticker": "MRNA", "date": "2026-08-28", "open": 100.0, "high": 105.0, "low": 99.0,
             "close": 103.0, "volume": 1_000_000, "dividends": 0.0, "stock_splits": 0.0}]

    result = write_daily_bars(7, bars, repo, known_from=_KNOWN_FROM)

    assert result == {"prices_written": 1, "actions_written": 0}
    repo.upsert_price_daily.assert_called_once_with(
        company_id=7, trade_date=date(2026, 8, 28), vendor="yfinance",
        open_raw=100.0, high_raw=105.0, low_raw=99.0, close_raw=103.0, volume_raw=1_000_000,
        known_from=_KNOWN_FROM,
    )
    repo.upsert_corporate_action.assert_not_called()


def test_write_daily_bars_skips_bar_with_no_close():
    repo = MagicMock()
    bars = [{"ticker": "MRNA", "date": "2026-08-28", "close": None}]

    result = write_daily_bars(7, bars, repo, known_from=_KNOWN_FROM)

    assert result == {"prices_written": 0, "actions_written": 0}
    repo.upsert_price_daily.assert_not_called()


def test_write_daily_bars_forward_split_writes_split_action():
    repo = MagicMock()
    bars = [{"ticker": "NVDA", "date": "2026-06-10", "open": 121.0, "high": 122.0, "low": 120.0,
             "close": 121.79, "volume": 500_000, "dividends": 0.0, "stock_splits": 10.0}]

    result = write_daily_bars(7, bars, repo, known_from=_KNOWN_FROM)

    assert result == {"prices_written": 1, "actions_written": 1}
    repo.upsert_corporate_action.assert_called_once_with(
        company_id=7, ex_date=date(2026, 6, 10), action_type="split", ratio=10.0,
        source="yfinance", is_verified=False, known_from=_KNOWN_FROM,
    )


def test_write_daily_bars_reverse_split_ratio_below_one():
    repo = MagicMock()
    bars = [{"ticker": "PENNY", "date": "2026-06-10", "close": 5.0, "stock_splits": 0.05, "dividends": 0.0}]

    write_daily_bars(7, bars, repo, known_from=_KNOWN_FROM)

    kwargs = repo.upsert_corporate_action.call_args.kwargs
    assert kwargs["action_type"] == "reverse_split"
    assert kwargs["ratio"] == 0.05


def test_write_daily_bars_dividend_writes_dividend_action():
    repo = MagicMock()
    bars = [{"ticker": "PFE", "date": "2026-08-01", "close": 28.0, "dividends": 0.42, "stock_splits": 0.0}]

    result = write_daily_bars(7, bars, repo, known_from=_KNOWN_FROM)

    assert result == {"prices_written": 1, "actions_written": 1}
    repo.upsert_corporate_action.assert_called_once_with(
        company_id=7, ex_date=date(2026, 8, 1), action_type="dividend", cash_amount=0.42,
        source="yfinance", is_verified=False, known_from=_KNOWN_FROM,
    )


def test_write_daily_bars_defaults_known_from_to_now():
    repo = MagicMock()
    bars = [{"ticker": "MRNA", "date": "2026-08-28", "close": 103.0}]

    write_daily_bars(7, bars, repo)

    kwargs = repo.upsert_price_daily.call_args.kwargs
    assert kwargs["known_from"] is not None
    assert kwargs["known_from"].tzinfo is not None


def test_write_daily_bars_multiple_bars_accumulate_counts():
    repo = MagicMock()
    bars = [
        {"ticker": "MRNA", "date": "2026-08-27", "close": 100.0, "stock_splits": 0.0, "dividends": 0.0},
        {"ticker": "MRNA", "date": "2026-08-28", "close": 103.0, "stock_splits": 0.0, "dividends": 0.0},
    ]

    result = write_daily_bars(7, bars, repo, known_from=_KNOWN_FROM)

    assert result == {"prices_written": 2, "actions_written": 0}
