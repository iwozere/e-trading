"""Tests for P18Reader."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.signals.p18_reader import P18Reader


class TestP18Reader:
    def test_no_p18_data_returns_zeros(self, tmp_path):
        """When the results directory doesn't exist, all counts are 0."""
        reader = P18Reader(results_base=tmp_path / "nonexistent")
        result = reader.get_high_score_tickers(date(2026, 6, 14))

        assert result["high_score_count"] == 0
        assert result["tickers"] == {}
        assert result["consensus_tickers"] == set()
        assert result["form4_buy_tickers"] == set()

    def test_reads_top_picks_csv(self, tmp_path):
        """Reads signals.csv and returns tickers scoring >= threshold."""
        run_dir = tmp_path / "2026-06-14"
        run_dir.mkdir(parents=True)

        signals_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "LOW"],
                "total_score": [80, 65, 40],
            }
        )
        signals_df.to_csv(run_dir / "signals.csv", index=False)

        reader = P18Reader(results_base=tmp_path)
        result = reader.get_high_score_tickers(date(2026, 6, 14))

        assert result["high_score_count"] == 2
        assert "AAPL" in result["tickers"]
        assert "MSFT" in result["tickers"]
        assert "LOW" not in result["tickers"]

    def test_date_fallback_finds_most_recent(self, tmp_path):
        """When no exact date match, falls back to most recent earlier date."""
        old_dir = tmp_path / "2026-06-10"
        old_dir.mkdir()
        signals_df = pd.DataFrame({"ticker": ["NVDA"], "total_score": [90]})
        signals_df.to_csv(old_dir / "signals.csv", index=False)

        reader = P18Reader(results_base=tmp_path)
        result = reader.get_high_score_tickers(date(2026, 6, 14))

        assert result["high_score_count"] == 1
        assert "NVDA" in result["tickers"]

    def test_reads_consensus(self, tmp_path):
        """consensus.csv tickers are loaded into consensus_tickers."""
        run_dir = tmp_path / "2026-06-14"
        run_dir.mkdir()

        pd.DataFrame({"ticker": ["AAPL"], "total_score": [70]}).to_csv(run_dir / "signals.csv", index=False)
        pd.DataFrame({"ticker": ["MSFT", "GOOG"]}).to_csv(run_dir / "consensus.csv", index=False)

        reader = P18Reader(results_base=tmp_path)
        result = reader.get_high_score_tickers(date(2026, 6, 14))

        assert "MSFT" in result["consensus_tickers"]
        assert "GOOG" in result["consensus_tickers"]

    def test_reads_form4_buys_when_present(self, tmp_path):
        """form4_buys.csv tickers populate form4_buy_tickers (forward-compatible)."""
        run_dir = tmp_path / "2026-06-14"
        run_dir.mkdir()

        pd.DataFrame({"ticker": ["AAPL"], "total_score": [70]}).to_csv(run_dir / "signals.csv", index=False)
        pd.DataFrame({"ticker": ["TSLA", "NVDA"]}).to_csv(run_dir / "form4_buys.csv", index=False)

        reader = P18Reader(results_base=tmp_path)
        result = reader.get_high_score_tickers(date(2026, 6, 14))

        assert result["form4_buy_tickers"] == {"TSLA", "NVDA"}

    def test_form4_buys_empty_when_file_absent(self, tmp_path):
        """With no form4_buys.csv (today's P18 reality), form4_buy_tickers is empty."""
        run_dir = tmp_path / "2026-06-14"
        run_dir.mkdir()
        pd.DataFrame({"ticker": ["AAPL"], "total_score": [70]}).to_csv(run_dir / "signals.csv", index=False)

        reader = P18Reader(results_base=tmp_path)
        result = reader.get_high_score_tickers(date(2026, 6, 14))

        assert result["form4_buy_tickers"] == set()
