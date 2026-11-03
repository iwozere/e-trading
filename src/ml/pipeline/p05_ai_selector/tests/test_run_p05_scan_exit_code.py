"""
Tests that run_p05_scan exits nonzero on a failed run so the scheduler marks it FAILED.
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p05_ai_selector import run_p05_scan


def _run_main(argv: list[str]) -> int:
    """Run main() with the given CLI args."""
    with patch.object(sys, "argv", ["run_p05_scan.py", *argv]):
        return run_p05_scan.main()


def test_successful_run_exits_zero() -> None:
    with patch.object(run_p05_scan, "P05Pipeline") as pipeline:
        pipeline.return_value.run.return_value = {"success": True, "pick_count": 3}
        assert _run_main([]) == 0


def test_failed_run_exits_nonzero() -> None:
    with patch.object(run_p05_scan, "P05Pipeline") as pipeline:
        pipeline.return_value.run.return_value = {"success": False, "error": "credit balance is too low"}
        assert _run_main([]) == 1


def test_unhandled_exception_exits_nonzero() -> None:
    with patch.object(run_p05_scan, "P05Pipeline") as pipeline:
        pipeline.return_value.run.side_effect = RuntimeError("boom")
        assert _run_main([]) == 1


def test_invalid_date_exits_nonzero() -> None:
    assert _run_main(["--as-of-date", "not-a-date"]) == 1
