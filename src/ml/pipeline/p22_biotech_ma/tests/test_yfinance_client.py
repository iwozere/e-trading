"""Tests for ingest/yfinance_client.py — mocked yfinance.Ticker, no network calls."""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client import fetch_recent_daily_bars


def _history_df(rows):
    """rows: list of (date, open, high, low, close, volume, dividends, splits)."""
    index = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [r[5] for r in rows],
            "Dividends": [r[6] for r in rows],
            "Stock Splits": [r[7] for r in rows],
        },
        index=index,
    )


def test_fetch_recent_daily_bars_parses_rows():
    df = _history_df([
        ("2026-08-28", 100.0, 105.0, 99.0, 103.0, 1_000_000, 0.0, 0.0),
        ("2026-08-31", 103.0, 106.0, 102.0, 104.5, 900_000, 0.0, 0.0),
    ])
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client.yf.Ticker", return_value=mock_ticker):
        bars = fetch_recent_daily_bars("MRNA")

    assert len(bars) == 2
    assert bars[0]["ticker"] == "MRNA"
    assert bars[0]["date"] == "2026-08-28"
    assert bars[0]["close"] == 103.0
    assert bars[0]["volume"] == 1_000_000
    assert bars[1]["close"] == 104.5


def test_fetch_recent_daily_bars_captures_split_and_dividend():
    df = _history_df([("2026-06-10", 121.79, 122.5, 120.5, 121.79, 500_000, 0.5, 10.0)])
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client.yf.Ticker", return_value=mock_ticker):
        bars = fetch_recent_daily_bars("NVDA")

    assert bars[0]["stock_splits"] == 10.0
    assert bars[0]["dividends"] == 0.5


def test_fetch_recent_daily_bars_empty_history_returns_empty_list():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client.yf.Ticker", return_value=mock_ticker):
        assert fetch_recent_daily_bars("NOSUCHTICKER") == []


def test_fetch_recent_daily_bars_exception_returns_empty_list_not_raises():
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = RuntimeError("network broke")

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client.yf.Ticker", return_value=mock_ticker):
        assert fetch_recent_daily_bars("MRNA") == []


def test_fetch_recent_daily_bars_uses_narrow_window_not_full_history():
    """The whole point of this client is a narrow trailing window — see module docstring's
    retroactive-split-adjustment trap. Verify the requested start date respects lookback_days."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client.yf.Ticker", return_value=mock_ticker):
        fetch_recent_daily_bars("MRNA", lookback_days=5)

    call_kwargs = mock_ticker.history.call_args.kwargs
    assert call_kwargs["start"] == date.today() - timedelta(days=5)
    assert call_kwargs["auto_adjust"] is False  # must request non-auto-adjusted (even though Close is still split-adjusted historically)
