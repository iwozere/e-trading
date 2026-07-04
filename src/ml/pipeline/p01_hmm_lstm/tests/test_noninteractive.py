"""
Verifies that PipelineRunner never calls input() in non-interactive mode,
and that the default skip_stages is empty (all stages run).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
project_root = Path(__file__).resolve().parents[5]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.pipeline.p01_hmm_lstm.run_pipeline import PipelineRunner


@pytest.fixture
def mock_config(tmp_path):
    """Write a minimal valid p01.yaml so PipelineRunner can load it."""
    cfg = tmp_path / "p01.yaml"
    cfg.write_text(
        "symbols: [BTCUSDT]\n"
        "timeframes: [15m]\n"
        "optuna:\n"
        "  n_trials: 1\n"
        "  timeout: 60\n"
        "paths:\n"
        "  data_raw: /tmp/data_raw\n"
        "  data_processed: /tmp/data_processed\n"
        "  data_labeled: /tmp/data_labeled\n"
        "  results: /tmp/results\n"
        "  reports: /tmp/reports\n"
        "  models_lstm: /tmp/models\n"
    )
    return str(cfg)


def test_no_input_call_in_noninteractive_mode(mock_config):
    """
    When a stage fails and interactive=False (default), input() must NOT be called.
    """
    runner = PipelineRunner(mock_config, interactive=False)

    # Make every stage fail
    failing_result = lambda stage_num: {
        "stage": stage_num,
        "name": runner.stages[stage_num]["name"],
        "success": False,
        "execution_time": 0.0,
        "error": "simulated failure",
    }

    with patch.object(runner, "run_stage", side_effect=failing_result), patch("builtins.input") as mock_input:
        runner.run_pipeline(fail_fast=False, continue_on_optional_failures=False)

    mock_input.assert_not_called(), "input() was called in non-interactive mode!"


def test_input_called_in_interactive_mode_on_optional_failure(mock_config):
    """
    When interactive=True and an optional stage fails with
    continue_on_optional_failures=False, input() must be called.
    """
    runner = PipelineRunner(mock_config, interactive=True)

    # Stage 4 is optional; make it fail
    def side_effect(stage_num):
        success = stage_num not in (4,)
        return {
            "stage": stage_num,
            "name": runner.stages[stage_num]["name"],
            "success": success,
            "execution_time": 0.0,
            "error": "simulated" if not success else None,
        }

    with (
        patch.object(runner, "run_stage", side_effect=side_effect),
        patch("builtins.input", return_value="n") as mock_input,
    ):
        runner.run_pipeline(fail_fast=True, continue_on_optional_failures=False)

    mock_input.assert_called_once()


def test_default_skip_stages_is_empty(mock_config):
    """Default skip_stages passed to run_pipeline must be empty."""
    runner = PipelineRunner(mock_config, interactive=False)
    with patch.object(
        runner,
        "run_stage",
        return_value={"stage": 1, "name": "x", "success": True, "execution_time": 0.0, "result": None},
    ):
        results = runner.run_pipeline()

    assert results["stages_skipped"] == [], "Default skip_stages should be [] — all stages must run by default"
