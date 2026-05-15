"""
P17 Explosive Penny Stock Screener — Pipeline Orchestrator

Coordinates all pipeline stages:
  Stage 1: Universe download + hard filters  (UniverseAgent)
  Stage 2: OHLCV + fundamental enrichment   (MarketAgent)
  Stage 3: Technical indicator computation  (TechnicalAgent)
  Stage 4: Short squeeze enrichment         (ShortSqueezeAgent)
  Stage 5: Dilution risk detection          (DilutionAgent)
  Stage 6: Composite scoring + tier assign  (ScoringAgent)
  Stage 7: Report generation                (ReportingAgent)

Each stage is wrapped in _run_job() for isolated error handling — a single
failing stage does not abort the pipeline.
"""

import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17PipelineConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.ml.pipeline.p17_penny_stocks.agents.universe_agent import UniverseAgent
from src.ml.pipeline.p17_penny_stocks.agents.market_agent import MarketAgent
from src.ml.pipeline.p17_penny_stocks.agents.technical_agent import TechnicalAgent
from src.ml.pipeline.p17_penny_stocks.agents.short_squeeze_agent import ShortSqueezeAgent
from src.ml.pipeline.p17_penny_stocks.agents.dilution_agent import DilutionAgent
from src.ml.pipeline.p17_penny_stocks.agents.scoring_agent import ScoringAgent
from src.ml.pipeline.p17_penny_stocks.agents.reporting_agent import ReportingAgent

_logger = setup_logger(__name__)


def _sf(value: Any, default: float = 0.0) -> float:
    """Safe float conversion with fallback."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class P17Pipeline:
    """
    End-to-end P17 penny stock screener pipeline.

    Usage:
        config = P17PipelineConfig.create_default()
        pipeline = P17Pipeline(config)
        result = pipeline.run()
    """

    def __init__(
        self,
        config: Optional[P17PipelineConfig] = None,
        target_date: Optional[str] = None,
    ) -> None:
        self.config = config or P17PipelineConfig.create_default()

        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.target_date = target_date

        self._results_dir = Path("results") / "p17_penny_stocks" / target_date
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._setup_pipeline_logging()
        self._init_agents()

        _logger.info(
            "P17 Pipeline initialised (target_date=%s, results=%s)",
            self.target_date, self._results_dir,
        )

    # ── Agents ─────────────────────────────────────────────────────────────

    def _init_agents(self) -> None:
        cfg = self.config
        self._universe_agent = UniverseAgent(
            cfg.filter_config, cfg.universe_config,
            self._results_dir, self.target_date,
        )
        self._market_agent = MarketAgent(
            cfg.filter_config, self._results_dir, self.target_date,
        )
        self._technical_agent = TechnicalAgent(cfg.technical_config)
        self._ss_agent = ShortSqueezeAgent(
            cfg.short_squeeze_config, self._results_dir, self.target_date,
        )
        self._dilution_agent = DilutionAgent(
            cfg.scoring_config, self._results_dir, self.target_date,
        )
        self._scoring_agent = ScoringAgent(cfg.scoring_config, self._ss_agent)
        self._reporting_agent = ReportingAgent(
            cfg.scoring_config, self._results_dir, self.target_date,
        )

    # ── Run ────────────────────────────────────────────────────────────────

    def run(
        self,
        force_refresh: bool = False,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute all pipeline stages.

        Args:
            force_refresh: Bypass all caches.
            tickers: Optional explicit ticker list (skips NASDAQ FTP).

        Returns:
            Summary dict with counts, paths, and per-stage timing.
        """
        t_start = time.monotonic()
        _logger.info("=" * 70)
        _logger.info("P17 Pipeline Starting  (target_date=%s)", self.target_date)
        _logger.info("=" * 70)

        job_results: Dict[str, Dict] = {}

        # ── Stage 1: Universe ──────────────────────────────────────────────
        universe_df: pd.DataFrame = pd.DataFrame()

        def stage1() -> Dict:
            nonlocal universe_df
            universe_df = self._universe_agent.run(force_refresh, tickers)
            return {"universe_size": len(universe_df)}

        job_results["universe"] = self._run_job("Stage1 Universe", stage1)

        if universe_df.empty:
            _logger.error("Universe is empty — aborting pipeline")
            return self._abort_result(job_results, t_start)

        ticker_list = universe_df["ticker"].tolist()

        # ── Stage 2: Market data ───────────────────────────────────────────
        ohlcv: Dict = {}
        fundamentals: Dict = {}

        def stage2() -> Dict:
            nonlocal ohlcv, fundamentals
            ohlcv, fundamentals = self._market_agent.run(ticker_list, force_refresh)

            # Survival filter (cash runway, debt/cash)
            survived = self._market_agent.apply_survival_filter(
                ticker_list, fundamentals, self.config.filter_config
            )
            return {"ohlcv_tickers": len(ohlcv), "survived_survival": len(survived)}

        job_results["market"] = self._run_job("Stage2 Market", stage2)

        # Build Candidate objects from universe_df rows
        candidates = self._build_candidates(universe_df, fundamentals)

        if not candidates:
            _logger.error("No candidates after market stage — aborting")
            return self._abort_result(job_results, t_start)

        if self.config.save_intermediate_results:
            self._save_candidates_csv(candidates, "03_candidates_post_market.csv")

        # ── Stage 3: Technical ─────────────────────────────────────────────
        def stage3() -> Dict:
            self._technical_agent.run(candidates, ohlcv)
            snap = self._technical_agent.build_snapshot(candidates)
            if snap is not None and self.config.save_intermediate_results:
                snap.to_csv(self._results_dir / "04_technical_snapshot.csv", index=False)
            return {"enriched": sum(1 for c in candidates if c.relative_volume > 0)}

        job_results["technical"] = self._run_job("Stage3 Technical", stage3)

        # ── Stage 4: Short squeeze ─────────────────────────────────────────
        def stage4() -> Dict:
            self._ss_agent.run(candidates, fundamentals)
            si_known = sum(1 for c in candidates if c.short_interest_pct_float is not None)
            return {"si_data_available": si_known}

        job_results["short_squeeze"] = self._run_job("Stage4 ShortSqueeze", stage4)

        # ── Stage 5: Dilution ──────────────────────────────────────────────
        def stage5() -> Dict:
            self._dilution_agent.run(candidates, force_refresh)
            flagged = sum(1 for c in candidates if c.dilution_penalty > 0)
            return {"dilution_flagged": flagged}

        job_results["dilution"] = self._run_job("Stage5 Dilution", stage5)

        # ── Stage 6: Scoring ───────────────────────────────────────────────
        def stage6() -> Dict:
            self._scoring_agent.run(candidates)
            return {
                "tier_a": sum(1 for c in candidates if c.tier == "A"),
                "tier_b": sum(1 for c in candidates if c.tier == "B"),
                "tier_c": sum(1 for c in candidates if c.tier == "C"),
                "explosive": sum(1 for c in candidates if c.explosive_candidate),
            }

        job_results["scoring"] = self._run_job("Stage6 Scoring", stage6)

        # ── Stage 7: Reporting ─────────────────────────────────────────────
        report_meta: Dict = {}

        def stage7() -> Dict:
            nonlocal report_meta
            report_meta = self._reporting_agent.run(candidates)
            return report_meta

        job_results["reporting"] = self._run_job("Stage7 Reporting", stage7)

        elapsed = round(time.monotonic() - t_start, 1)
        _logger.info("=" * 70)
        _logger.info("P17 Pipeline Completed in %.1fs", elapsed)
        _logger.info(
            "Result: %d candidates | A=%d B=%d C=%d | explosive=%d",
            len(candidates),
            report_meta.get("tier_a", 0), report_meta.get("tier_b", 0),
            report_meta.get("tier_c", 0), report_meta.get("explosive", 0),
        )
        _logger.info("=" * 70)

        return {
            "success": True,
            "target_date": self.target_date,
            "elapsed_s": elapsed,
            "total_candidates": len(candidates),
            **report_meta,
            "stages": job_results,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_candidates(
        self,
        universe_df: pd.DataFrame,
        fundamentals: Dict,
    ) -> List[Candidate]:
        candidates = []
        for _, row in universe_df.iterrows():
            ticker = str(row.get("ticker", ""))
            if not ticker:
                continue

            f = fundamentals.get(ticker, {})
            c = Candidate(
                ticker=ticker,
                company_name=str(row.get("company_name", "")),
                exchange=str(row.get("exchange_norm", row.get("exchange", ""))),
                sector=str(row.get("sector", "")),
                industry=str(row.get("industry", "")),
                price=_sf(row.get("price")),
                market_cap=_sf(row.get("market_cap")),
                volume=_sf(row.get("volume")),
                avg_volume_30d=_sf(row.get("avg_volume")),
                float_shares=_sf(row.get("float_shares")),
                shares_outstanding=_sf(row.get("shares_outstanding")),
                high_52w=_sf(f.get("high_52w") or row.get("high_52w")),
                low_52w=_sf(f.get("low_52w") or row.get("low_52w")),
                gross_margin=f.get("gross_margin"),
                revenue_growth_yoy=f.get("revenue_growth_yoy"),
                total_cash=f.get("total_cash"),
                total_debt=f.get("total_debt"),
                cash_runway_months=f.get("cash_runway_months"),
                operating_cashflow=f.get("operating_cashflow"),
                institutional_pct=f.get("institutional_pct") or row.get("institutional_pct"),
                short_interest_pct_float=f.get("short_pct_float") or row.get("short_pct_float"),
                days_to_cover=f.get("short_ratio") or row.get("short_ratio"),
                run_date=self.target_date,
            )
            candidates.append(c)

        _logger.info("Built %d Candidate objects", len(candidates))
        return candidates

    def _save_candidates_csv(self, candidates: List[Candidate], filename: str) -> None:
        try:
            rows = [c.to_dict() for c in candidates]
            pd.DataFrame(rows).to_csv(self._results_dir / filename, index=False)
        except Exception:
            _logger.warning("Could not save intermediate CSV: %s", filename)

    @staticmethod
    def _run_job(name: str, fn: Callable[[], Optional[Dict]]) -> Dict:
        t0 = time.monotonic()
        try:
            extra = fn() or {}
            elapsed = round(time.monotonic() - t0, 1)
            _logger.info("%-30s OK   %.1fs", name, elapsed)
            return {"success": True, "elapsed_s": elapsed, **extra}
        except Exception:
            elapsed = round(time.monotonic() - t0, 1)
            _logger.exception("%-30s FAIL %.1fs", name, elapsed)
            return {"success": False, "elapsed_s": elapsed}

    @staticmethod
    def _abort_result(job_results: Dict, t_start: float) -> Dict:
        return {
            "success": False,
            "elapsed_s": round(time.monotonic() - t_start, 1),
            "total_candidates": 0,
            "stages": job_results,
        }

    # ── Logging ────────────────────────────────────────────────────────────

    def _setup_pipeline_logging(self) -> None:
        log_file = self._results_dir / "pipeline.log"
        handler = RotatingFileHandler(
            str(log_file), maxBytes=100 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(fmt)
        pipeline_logger = logging.getLogger("src.ml.pipeline")
        pipeline_logger.addHandler(handler)
        pipeline_logger.setLevel(logging.DEBUG)
        self._log_handler = handler

    def __del__(self) -> None:
        if hasattr(self, "_log_handler"):
            logging.getLogger("src.ml.pipeline").removeHandler(self._log_handler)
            self._log_handler.close()
