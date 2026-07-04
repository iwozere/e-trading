"""Tests for EarningsCalendar."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.signals.earnings_calendar import EarningsCalendar


def _make_calendar_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["symbol", "date", "eps", "epsEstimated"])


class TestEarningsCalendar:
    def test_fmp_failure_returns_empty(self, tmp_path):
        """When FMP returns an HTTP error, returns empty dict without raising."""
        cal = EarningsCalendar(api_key="fake", cache_dir=tmp_path)

        with patch("src.ml.pipeline.p05_ai_selector.signals.earnings_calendar.requests.get") as mock_get:
            mock_get.side_effect = Exception("network error")
            result = cal.get_earnings_within_days(["AAPL"], date(2026, 6, 14))

        assert result == {}

    def test_filters_by_window(self, tmp_path):
        """Only returns tickers with earnings within the requested window."""
        cal = EarningsCalendar(api_key="fake", cache_dir=tmp_path)
        ref_date = date(2026, 6, 14)
        earnings_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date": "2026-06-18", "eps": None, "epsEstimated": None},
                {"symbol": "MSFT", "date": "2026-06-25", "eps": None, "epsEstimated": None},
            ]
        )

        with patch.object(cal, "_fetch_from_fmp", return_value=earnings_df):
            result = cal.get_earnings_within_days(["AAPL", "MSFT"], ref_date, window_days=7)

        assert "AAPL" in result
        assert "MSFT" not in result

    def test_cache_hit_no_http_call(self, tmp_path):
        """When cache is fresh, no HTTP call is made."""
        cal = EarningsCalendar(api_key="fake", cache_dir=tmp_path)

        # Pre-populate fresh cache
        cache_file = tmp_path / "2026-06.csv.gz"
        df = pd.DataFrame([{"symbol": "NVDA", "date": "2026-06-16", "eps": None, "epsEstimated": None}])
        df.to_csv(cache_file, index=False, compression="gzip")

        with patch("src.ml.pipeline.p05_ai_selector.signals.earnings_calendar.requests.get") as mock_get:
            result = cal.get_earnings_within_days(["NVDA"], date(2026, 6, 14))

        mock_get.assert_not_called()
        assert "NVDA" in result
