"""Tests for Stage4Output."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.stages.stage4_output import Stage4Output


def _make_pick(rank: int = 1, confidence: int = 7) -> dict:
    return {
        "rank": rank,
        "ticker": f"TICK{rank}",
        "confidence": confidence,
        "bias": "long",
        "thesis": "A compelling thesis for this setup.",
        "risk_factors": ["Macro headwinds", "Earnings risk"],
        "time_horizon": "3-6 months",
        "exit_strategy": {
            "add_conditions": ["Pullback to $90"],
            "hold_conditions": ["Revenue stays positive"],
            "thesis_breakers": ["Miss guidance by >5%", "Regulatory action"],
            "profit_targets": [
                {"price_level": 120.0, "action": "Trim 25%", "note": "lock gains"},
                {"price_level": 145.0, "action": "Trim 25%", "note": "halfway de-risked"},
            ],
            "time_horizon_note": "The structural tailwind plays out over 2-3 quarters.",
        },
    }


def _make_stage2_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "TICK1",
                "total_score": 80.0,
                "fundamental_score": 20.0,
                "momentum_score": 40.0,
                "p18_score": 20.0,
                "signal_breakdown": "{}",
            }
        ]
    )


class TestWriteResults:
    def test_write_results_creates_all_files(self, tmp_path):
        """All four output files are created."""
        output = Stage4Output(results_base=tmp_path)
        picks = [_make_pick(i + 1) for i in range(5)]
        metadata = {
            "run_date": "2026-06-14",
            "trigger_reason": "P18 flagged 2 tickers",
            "p18_signals_count": 2,
            "notification_override": False,
            "stage1_out": 150,
            "stage2_out": 25,
            "llm_tokens_used": 11000,
            "llm_model": "claude-sonnet-4-6",
            "elapsed_seconds": 45.2,
            "market_context": "Markets look constructive.",
            "timestamp": "2026-06-14T10:05:00+00:00",
        }
        run_dir = output.write_results(picks, _make_stage2_df(), metadata, date(2026, 6, 14))

        assert (run_dir / "top_picks.csv").exists()
        assert (run_dir / "full_ranking.csv").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "metadata.json").exists()


class TestFormatTelegram:
    def test_telegram_format_under_3800_chars(self):
        """Telegram message is at most 3800 characters."""
        output = Stage4Output()
        picks = [_make_pick(i + 1) for i in range(5)]
        msg = output.format_telegram(picks, "P18 flagged 3 tickers", date(2026, 6, 14))
        assert len(msg) <= 3800

    def test_telegram_contains_top3_tickers(self):
        """Top 3 tickers appear in the Telegram message."""
        output = Stage4Output()
        picks = [_make_pick(i + 1) for i in range(5)]
        msg = output.format_telegram(picks, "P18 flagged 3 tickers", date(2026, 6, 14))
        assert "TICK1" in msg
        assert "TICK2" in msg
        assert "TICK3" in msg


class TestShouldNotify:
    def test_should_notify_p18_trigger(self):
        """p18_signals_count >= 1 triggers notification."""
        output = Stage4Output()
        notify, reason = output.should_notify(3, False)
        assert notify is True
        assert "P18" in reason

    def test_should_notify_override_trigger(self):
        """notification_override=True triggers notification."""
        output = Stage4Output()
        notify, reason = output.should_notify(0, True)
        assert notify is True
        assert "confidence" in reason.lower()

    def test_should_notify_neither_returns_false(self):
        """Neither condition met → no notification."""
        output = Stage4Output()
        notify, reason = output.should_notify(0, False)
        assert notify is False
