"""
Tests for Phase 6.3 — rolling memory bootstrap health check.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_scanner(results_base: Path, target_date: str = "2026-06-09"):
    from src.ml.pipeline.p06_emps2.rolling_memory import RollingMemoryScanner

    config = MagicMock()
    config.lookback_days = 14
    return RollingMemoryScanner(
        config=config,
        results_base_path=results_base,
        target_date=target_date,
        verbose=False,
    )


class TestCheckBootstrapHealth:
    def test_persists_first_run_date_on_first_call(self, tmp_path):
        scanner = _make_scanner(tmp_path, "2026-06-09")
        scanner.check_bootstrap_health(phase1_count=3)
        state = json.loads((tmp_path / ".rolling_memory_state.json").read_text())
        assert state["first_run_date"] == "2026-06-09"

    def test_accumulates_phase1_count_across_calls(self, tmp_path):
        scanner = _make_scanner(tmp_path, "2026-06-09")
        scanner.check_bootstrap_health(phase1_count=2)
        scanner.check_bootstrap_health(phase1_count=3)
        state = json.loads((tmp_path / ".rolling_memory_state.json").read_text())
        assert state["cumulative_phase1_count"] == 5

    def test_first_run_date_not_overwritten_on_subsequent_calls(self, tmp_path):
        scanner1 = _make_scanner(tmp_path, "2026-06-01")
        scanner1.check_bootstrap_health(phase1_count=0)

        scanner2 = _make_scanner(tmp_path, "2026-06-09")
        scanner2.check_bootstrap_health(phase1_count=0)

        state = json.loads((tmp_path / ".rolling_memory_state.json").read_text())
        assert state["first_run_date"] == "2026-06-01"

    def test_no_error_when_detections_present(self, tmp_path, caplog):
        import logging

        scanner = _make_scanner(tmp_path, "2026-06-09")
        # First run 20 days ago
        existing_state = {
            "first_run_date": "2026-05-20",
            "last_run_date": "2026-06-08",
            "cumulative_phase1_count": 5,
        }
        (tmp_path / ".rolling_memory_state.json").write_text(json.dumps(existing_state))

        with caplog.at_level(logging.ERROR, logger="src.ml.pipeline.p06_emps2.rolling_memory"):
            scanner.check_bootstrap_health(phase1_count=1)

        assert not any(r.levelno == logging.ERROR for r in caplog.records)

    def test_error_emitted_when_zero_detections_after_lookback(self, tmp_path, caplog):
        import logging

        scanner = _make_scanner(tmp_path, "2026-06-09")
        # First run 20 days ago (> lookback of 14), zero cumulative detections
        existing_state = {
            "first_run_date": "2026-05-20",
            "last_run_date": "2026-06-08",
            "cumulative_phase1_count": 0,
        }
        (tmp_path / ".rolling_memory_state.json").write_text(json.dumps(existing_state))

        with caplog.at_level(logging.ERROR, logger="src.ml.pipeline.p06_emps2.rolling_memory"):
            scanner.check_bootstrap_health(phase1_count=0)

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "Expected at least one ERROR log record"
        assert "health check FAILED" in error_records[0].message

    def test_no_error_when_still_within_lookback(self, tmp_path, caplog):
        import logging

        scanner = _make_scanner(tmp_path, "2026-06-09")
        # First run only 5 days ago (< lookback of 14) — too soon to flag
        existing_state = {
            "first_run_date": "2026-06-04",
            "last_run_date": "2026-06-08",
            "cumulative_phase1_count": 0,
        }
        (tmp_path / ".rolling_memory_state.json").write_text(json.dumps(existing_state))

        with caplog.at_level(logging.ERROR, logger="src.ml.pipeline.p06_emps2.rolling_memory"):
            scanner.check_bootstrap_health(phase1_count=0)

        assert not any(r.levelno == logging.ERROR for r in caplog.records)

    def test_corrupt_state_file_handled_gracefully(self, tmp_path):
        scanner = _make_scanner(tmp_path, "2026-06-09")
        (tmp_path / ".rolling_memory_state.json").write_text("NOT_JSON{{{}}")
        # Should not raise
        scanner.check_bootstrap_health(phase1_count=1)
        state = json.loads((tmp_path / ".rolling_memory_state.json").read_text())
        assert state["first_run_date"] == "2026-06-09"
