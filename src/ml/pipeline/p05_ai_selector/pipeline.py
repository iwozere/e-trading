"""P05 AI Selector — 4-stage pipeline orchestrator."""

import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p05_ai_selector.stages.universe_loader import UniverseLoader
from src.ml.pipeline.p05_ai_selector.signals.p18_reader import P18Reader
from src.ml.pipeline.p05_ai_selector.signals.earnings_calendar import EarningsCalendar
from src.ml.pipeline.p05_ai_selector.stages.stage1_prefilter import Stage1Prefilter
from src.ml.pipeline.p05_ai_selector.stages.stage2_scorer import Stage2Scorer
from src.ml.pipeline.p05_ai_selector.stages.stage3_llm_synthesizer import Stage3LLMSynthesizer
from src.ml.pipeline.p05_ai_selector.stages.stage4_output import Stage4Output
from src.ml.pipeline.p05_ai_selector.config import LLM_MODEL, RESULTS_BASE
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


_P05_LOGGER_NAME = "src.ml.pipeline.p05_ai_selector"
_RUN_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s"


class P05Pipeline:
    """
    Daily AI-powered equity and crypto screener.

    Stages:
      0 — Universe load (Russell 3000 + crypto)
      1 — Liquidity & momentum pre-filter (~3,020 → ~200)
      2 — Deterministic signal scoring (~200 → top-25)
      3 — LLM synthesis (top-25 → top-5 with exit strategies)
      4 — Output generation (CSV/MD/JSON + optional notifications)
    """

    def _attach_run_log_handler(self, run_dir: Path) -> logging.FileHandler:
        run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_RUN_LOG_FORMAT))
        logging.getLogger(_P05_LOGGER_NAME).addHandler(handler)
        return handler

    def _detach_run_log_handler(self, handler: logging.FileHandler) -> None:
        logging.getLogger(_P05_LOGGER_NAME).removeHandler(handler)
        handler.close()

    def run(
        self,
        user_id: Optional[str] = None,
        as_of_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full daily pipeline.

        Args:
            user_id: Injected by the scheduler; stored in result dict.
            as_of_date: Reference date. Defaults to today.
            force_refresh: Bypass all stage caches and recompute.

        Returns:
            Result dict consumed by the scheduler notification_rules engine.
        """
        run_date = as_of_date or date.today()
        t_start = time.monotonic()
        stage_times: Dict[str, float] = {}

        run_dir = RESULTS_BASE / str(run_date)
        log_handler = self._attach_run_log_handler(run_dir)
        try:
            _logger.info("=" * 60)
            _logger.info("P05 AI Selector — daily run for %s", run_date)
            _logger.info("=" * 60)

            # Stage 0 — Universe
            t0 = time.monotonic()
            tickers = UniverseLoader().load()
            stage_times["universe_s"] = round(time.monotonic() - t0, 1)
            _logger.info("Stage 0 done: %d tickers loaded in %.1fs", len(tickers), stage_times["universe_s"])

            # P18 signals (parallel with earnings)
            t0 = time.monotonic()
            p18_data = P18Reader().get_high_score_tickers(run_date)
            stage_times["p18_read_s"] = round(time.monotonic() - t0, 1)

            t0 = time.monotonic()
            earnings_flags = EarningsCalendar().get_earnings_within_days(tickers, run_date)
            stage_times["earnings_s"] = round(time.monotonic() - t0, 1)

            # Stage 1 — Pre-filter
            t0 = time.monotonic()
            stage1_df = Stage1Prefilter().run(tickers, run_date, force_refresh=force_refresh)
            stage_times["stage1_s"] = round(time.monotonic() - t0, 1)
            stage1_out = len(stage1_df)
            _logger.info("Stage 1 done: %d candidates in %.1fs", stage1_out, stage_times["stage1_s"])

            # Stage 2 — Scorer
            t0 = time.monotonic()
            stage2_df = Stage2Scorer().run(
                stage1_df, p18_data, earnings_flags, run_date, force_refresh=force_refresh
            )
            stage_times["stage2_s"] = round(time.monotonic() - t0, 1)
            stage2_out = len(stage2_df)
            _logger.info("Stage 2 done: %d candidates in %.1fs", stage2_out, stage_times["stage2_s"])

            if stage2_df.empty:
                _logger.error("Stage 2 returned empty — aborting pipeline")
                return self._failure_result(
                    "Stage 2 returned no candidates", p18_data, user_id, run_date, t_start
                )

            # Stage 3 — LLM
            t0 = time.monotonic()
            llm_result = Stage3LLMSynthesizer().run(stage2_df)
            stage_times["stage3_s"] = round(time.monotonic() - t0, 1)
            picks = llm_result["picks"]
            tokens_used = llm_result.get("tokens_used", 0)
            notification_override = llm_result.get("notification_override", False)
            _logger.info(
                "Stage 3 done: %d picks, tokens=%d in %.1fs",
                len(picks),
                tokens_used,
                stage_times["stage3_s"],
            )

            # Stage 4 — Output
            output = Stage4Output()
            notify, trigger_reason = output.should_notify(
                p18_data["high_score_count"], notification_override
            )
            _logger.info("Notification decision: %s (%s)", notify, trigger_reason)

            elapsed = round(time.monotonic() - t_start, 1)
            metadata = {
                "run_date": str(run_date),
                "trigger_reason": trigger_reason,
                "p18_signals_count": p18_data["high_score_count"],
                "notification_override": notification_override,
                "stage1_out": stage1_out,
                "stage2_out": stage2_out,
                "llm_tokens_used": tokens_used,
                "llm_model": LLM_MODEL,
                "elapsed_seconds": elapsed,
                "market_context": llm_result.get("market_context", ""),
                "stage_times": stage_times,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            results_dir = output.write_results(picks, stage2_df, metadata, run_date)

            top_ticker = picks[0]["ticker"] if picks else ""
            top_confidence = int(picks[0].get("confidence", 0)) if picks else 0

            result = {
                "success": True,
                "pick_count": len(picks),
                "p18_signals_count": p18_data["high_score_count"],
                "notification_override": 1 if notification_override else 0,
                "trigger_reason": trigger_reason,
                "top_ticker": top_ticker,
                "top_confidence": top_confidence,
                "stage1_out": stage1_out,
                "stage2_out": stage2_out,
                "llm_tokens_used": tokens_used,
                "results_dir": str(results_dir),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
            }
            _logger.info("P05 pipeline complete: %s", result)
            return result

        except Exception as exc:
            _logger.exception("P05 pipeline failed for %s", run_date)
            return self._failure_result(str(exc), {}, user_id, run_date, t_start)

        finally:
            self._detach_run_log_handler(log_handler)

    def _failure_result(
        self,
        error: str,
        p18_data: Dict[str, Any],
        user_id: Optional[str],
        run_date: date,
        t_start: float,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "pick_count": 0,
            "p18_signals_count": p18_data.get("high_score_count", 0),
            "notification_override": 0,
            "trigger_reason": "pipeline_error",
            "top_ticker": "",
            "top_confidence": 0,
            "stage1_out": 0,
            "stage2_out": 0,
            "llm_tokens_used": 0,
            "results_dir": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
        }
