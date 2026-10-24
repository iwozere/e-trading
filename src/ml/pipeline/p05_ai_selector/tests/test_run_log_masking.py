"""Tests that the P05 per-run pipeline.log masks credentials."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p05_ai_selector.pipeline import P05Pipeline


def test_run_log_masks_api_key_in_traceback(tmp_path: Path) -> None:
    """An exception whose message embeds an apikey URL is written masked."""
    pipeline = P05Pipeline.__new__(P05Pipeline)  # handler helpers need no pipeline state
    handler = pipeline._attach_run_log_handler(tmp_path)
    logger = logging.getLogger("src.ml.pipeline.p05_ai_selector.signals.earnings_calendar")
    try:
        try:
            raise RuntimeError("429 Client Error for url: https://example.com/x?apikey=SECRETKEY1234567890&from=1")
        except RuntimeError:
            logger.exception("HTTP error fetching calendar")
    finally:
        pipeline._detach_run_log_handler(handler)

    text = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "HTTP error fetching calendar" in text
    assert "SECRETKEY1234567890" not in text
    assert "[MASKED]" in text
