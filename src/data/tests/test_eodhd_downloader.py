import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import os

# Import the module to test
from src.data.downloader.eodhd_downloader import (
    EODHDDataDownloader,
    EODHDApiError
)

# Test data
SAMPLE_OPTION_CHAIN = {
    "data": [{
        "expiration": "2025-12-19",
        "options": {
            "call": [
                {
                    "strike": 150.0,
                    "last_trade_price": 5.25,
                    "volume": 1000,
                    "open_interest": 2500,
                    "implied_volatility": 0.35
                }
            ],
            "put": [
                {
                    "strike": 150.0,
                    "last_trade_price": 4.75,
                    "volume": 800,
                    "open_interest": 2000,
                    "implied_volatility": 0.32
                }
            ]
        }
    }]
}

# Fixtures
@pytest.fixture
def sample_option_data():
    """Sample options data for testing."""
    return {
        "ticker": "AAPL",
        "date": "2025-12-05",
        "expiration": "2025-12-19",
        "type": "call",
        "strike": 150.0,
        "last": 5.25,
        "volume": 1000,
        "open_interest": 2500,
        "iv": 0.35
    }

@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing statistics."""
    dates = pd.date_range(end=datetime.today(), periods=30).strftime('%Y-%m-%d')
    data = []
    for i, date in enumerate(dates):
        data.append({
            "ticker": "AAPL",
            "date": date,
            "type": "call",
            "volume": 1000 + (i * 10),
            "open_interest": 2500 + (i * 20)
        })
        data.append({
            "ticker": "AAPL",
            "date": date,
            "type": "put",
            "volume": 800 + (i * 5),
            "open_interest": 2000 + (i * 10)
        })
    return pd.DataFrame(data)

# Test cases
class TestFetchChain:
    @patch('src.data.downloader.eodhd_downloader.EODHDDataDownloader._make_api_request')
    def test_fetch_chain_success(self, mock_api_request):
        """Test successful option chain fetch."""
        # Setup
        mock_api_request.return_value = SAMPLE_OPTION_CHAIN
        downloader = EODHDDataDownloader()

        # Execute
        result = downloader.fetch_chain("AAPL", "2025-12-05")

        # Verify
        assert not result.empty
        assert len(result) == 2  # One call and one put
        assert "AAPL" in result["ticker"].values
        assert 150.0 in result["strike"].values

    @patch('src.data.downloader.eodhd_downloader.EODHDDataDownloader._make_api_request')
    def test_fetch_chain_no_data(self, mock_api_request):
        """Test handling of no data from API."""
        # Setup
        mock_api_request.return_value = {"data": []}
        downloader = EODHDDataDownloader()

        # Execute
        result = downloader.fetch_chain("INVALID", "2025-12-05")

        # Verify
        assert result.empty

    @patch('src.data.downloader.eodhd_downloader.EODHDDataDownloader._make_api_request')
    def test_fetch_chain_api_error(self, mock_api_request):
        """Test handling of API errors."""
        # Setup
        mock_api_request.side_effect = EODHDApiError("API Error")
        downloader = EODHDDataDownloader()

        # Execute & Verify
        with pytest.raises(EODHDApiError):
            downloader.fetch_chain("AAPL", "2025-12-05")

class TestDownloadForDate:
    @patch('src.data.downloader.eodhd_downloader.EODHDDataDownloader.fetch_chain')
    def test_download_for_date_single_ticker(self, mock_fetch):
        """Test downloading data for a single ticker."""
        # Setup
        mock_df = pd.DataFrame([{"ticker": "AAPL", "volume": 1000}])
        mock_fetch.return_value = mock_df
        downloader = EODHDDataDownloader()

        # Execute
        result = downloader.download_for_date(["AAPL"], "2025-12-05")

        # Verify
        assert not result.empty
        assert len(result) == 1
        mock_fetch.assert_called_once()

    @patch('src.data.downloader.eodhd_downloader.EODHDDataDownloader.fetch_chain')
    def test_download_for_date_multiple_tickers(self, mock_fetch):
        """Test downloading data for multiple tickers."""
        # Setup
        mock_fetch.side_effect = [
            pd.DataFrame([{"ticker": "AAPL", "volume": 1000}]),
            pd.DataFrame([{"ticker": "MSFT", "volume": 800}])
        ]
        downloader = EODHDDataDownloader()

        # Execute
        result = downloader.download_for_date(["AAPL", "MSFT"], "2025-12-05")

        # Verify
        assert len(result) == 2
        assert set(result["ticker"]) == {"AAPL", "MSFT"}

class TestCompute30dStats:
    def test_compute_30d_stats_basic(self, sample_historical_data):
        """Test computing 30-day statistics."""
        downloader = EODHDDataDownloader()
        # Execute
        result = downloader.compute_30d_stats(sample_historical_data)

        # Verify
        assert not result.empty
        assert "call_volume_30d_avg" in result.columns
        assert "put_volume_30d_avg" in result.columns
        assert result["call_volume"].iloc[-1] > result["call_volume_30d_avg"].iloc[-1]

    def test_compute_30d_stats_empty_input(self):
        """Test with empty input DataFrame."""
        downloader = EODHDDataDownloader()
        # Execute
        result = downloader.compute_30d_stats(pd.DataFrame())

        # Verify
        assert result.empty

class TestComputeUOAScore:
    def test_compute_uoa_score_basic(self, sample_historical_data):
        """Test computing UOA score."""
        downloader = EODHDDataDownloader()
        # Setup
        stats = downloader.compute_30d_stats(sample_historical_data)

        # Execute
        result = downloader.compute_uoa_score(stats)

        # Verify
        assert not result.empty
        assert "uoa_score" in result.columns
        assert (result["uoa_score"] >= 0).all()
        assert (result["uoa_score"] <= 100).all()

    def test_compute_uoa_score_edge_cases(self):
        """Test UOA score with edge cases."""
        downloader = EODHDDataDownloader()
        # Setup
        df = pd.DataFrame([{
            "ticker": "AAPL",
            "date": "2025-12-05",
            "call_volume": 0,
            "put_volume": 0,
            "call_volume_30d_avg": 0,
            "put_volume_30d_avg": 0
        }])

        # Execute
        result = downloader.compute_uoa_score(df)

        # Verify
        assert not result.empty
        assert result["uoa_score"].iloc[0] == 0

class TestSaveToFile:
    def test_save_to_csv(self, tmp_path):
        """Test saving to CSV file."""
        downloader = EODHDDataDownloader()
        # Setup
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        filepath = tmp_path / "test.csv"

        # Execute
        result = downloader.save_to_file(df, str(tmp_path), "test.csv")

        # Verify
        assert result
        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_save_to_unsupported_format(self):
        """Test handling of unsupported file formats."""
        downloader = EODHDDataDownloader()
        # Setup
        df = pd.DataFrame({"col1": [1, 2]})

        # Execute & Verify
        with pytest.raises(ValueError):
            downloader.save_to_file(df, ".", "test.unsupported")

# Integration test
class TestIntegration:
    @patch('src.data.downloader.eodhd_downloader.EODHDDataDownloader._make_api_request')
    def test_end_to_end_flow(self, mock_api_request, tmp_path):
        """Test the complete flow from API call to file save."""
        downloader = EODHDDataDownloader()
        # Setup mock API response
        mock_api_request.return_value = SAMPLE_OPTION_CHAIN

        # 1. Fetch data
        df = downloader.fetch_chain("AAPL", "2025-12-05")
        assert not df.empty

        # 2. Compute stats
        stats = downloader.compute_30d_stats(df)
        assert not stats.empty

        # 3. Compute UOA score
        scores = downloader.compute_uoa_score(stats)
        assert "uoa_score" in scores.columns

        # 4. Save results
        output_file = tmp_path / "uoa_scores.csv"
        save_result = downloader.save_to_file(scores, str(tmp_path), "uoa_scores.csv")
        assert save_result
        assert output_file.exists()