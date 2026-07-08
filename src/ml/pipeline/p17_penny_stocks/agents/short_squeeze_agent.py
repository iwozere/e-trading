"""
P17 Short Squeeze Agent

Enriches candidates with short interest data from two sources:
  1. yfinance .info (shortPercentOfFloat, shortRatio) — bi-monthly FINRA snapshot
  2. FINRA regShoDaily — daily short-sale volume ratio (flow metric)

The two signals are combined:
  - yfinance SI%: absolute short position size
  - FINRA short-vol ratio: daily short-selling pressure trend

FINRA credentials are optional — the agent degrades gracefully to yfinance-only.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.config import P17ShortSqueezeConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_download_trf = None
try:
    from src.ml.pipeline.shared.trf_downloader import download_trf as _download_trf

    _TRF_AVAILABLE = True
except Exception:
    _TRF_AVAILABLE = False
    _logger.debug("TRF downloader not available — FINRA short-vol will be skipped")


class ShortSqueezeAgent:
    """
    Stage 4: Enrich candidates with short interest and short-vol ratio.

    Populates on each Candidate:
      - short_interest_pct_float
      - days_to_cover
      - finra_short_vol_ratio
    """

    def __init__(
        self,
        config: P17ShortSqueezeConfig,
        results_dir: Path,
        target_date: str,
    ) -> None:
        self.config = config
        self.results_dir = results_dir
        self.target_date = target_date

    def run(
        self,
        candidates: List[Candidate],
        fundamentals: Dict[str, dict],
    ) -> List[Candidate]:
        """
        Populate short-squeeze fields on each candidate.

        Args:
            candidates: List of Candidate objects.
            fundamentals: Dict[ticker → fundamentals dict] from MarketAgent.

        Returns:
            Same list with short-squeeze fields populated.
        """
        # Pull SI data from fundamentals (yfinance already fetched it)
        self._enrich_from_fundamentals(candidates, fundamentals)

        # Optionally enrich with FINRA daily short-vol ratio
        if _TRF_AVAILABLE:
            self._enrich_from_finra(candidates)

        enriched = sum(1 for c in candidates if c.short_interest_pct_float is not None)
        _logger.info("Short squeeze agent: %d/%d candidates have SI data", enriched, len(candidates))
        return candidates

    # ── yfinance fundamentals enrichment ───────────────────────────────────

    def _enrich_from_fundamentals(
        self,
        candidates: List[Candidate],
        fundamentals: Dict[str, dict],
    ) -> None:
        for c in candidates:
            f = fundamentals.get(c.ticker, {})
            si_pct = f.get("short_pct_float")
            short_ratio = f.get("short_ratio")

            if si_pct is not None:
                c.short_interest_pct_float = float(si_pct)
            if short_ratio is not None:
                c.days_to_cover = float(short_ratio)

            # Flag extremely high SI + low liquidity as halt risk
            if (
                c.short_interest_pct_float is not None
                and c.short_interest_pct_float > self.config.si_extreme_threshold
                and c.volume < self.config.high_si_min_volume
            ):
                if "halt_risk_high_si" not in c.signals:
                    c.signals.append("halt_risk_high_si")

    # ── FINRA TRF daily short-vol enrichment ───────────────────────────────

    def _enrich_from_finra(self, candidates: List[Candidate]) -> None:
        trf_df = self._load_finra_data()
        if trf_df is None or trf_df.empty:
            _logger.debug("No FINRA TRF data available")
            return

        # Normalize column names — FINRA TRF uses different column names across versions
        col_map = self._detect_columns(trf_df)
        if col_map is None:
            _logger.warning("FINRA TRF columns not recognised — skipping short-vol enrichment")
            return

        ticker_col, short_vol_col, total_vol_col = col_map

        candidate_tickers = {c.ticker for c in candidates}

        # Build ratios using iterrows to avoid pandas boolean-indexing type issues
        sv_by_ticker: Dict[str, float] = {}
        tv_by_ticker: Dict[str, float] = {}
        for _, row in trf_df.iterrows():
            t = str(row.get(ticker_col, ""))
            if t not in candidate_tickers:
                continue
            sv_by_ticker[t] = sv_by_ticker.get(t, 0.0) + float(row.get(short_vol_col, 0) or 0)
            tv_by_ticker[t] = tv_by_ticker.get(t, 0.0) + float(row.get(total_vol_col, 0) or 0)

        ratios: Dict[str, float] = {
            t: sv_by_ticker[t] / tv_by_ticker[t] for t in sv_by_ticker if tv_by_ticker.get(t, 0.0) > 0
        }

        enriched_n = 0
        for c in candidates:
            ratio = ratios.get(c.ticker)
            if ratio is not None:
                c.finra_short_vol_ratio = ratio
                enriched_n += 1

        _logger.info("FINRA short-vol ratio: enriched %d/%d candidates", enriched_n, len(candidates))

    def _load_finra_data(self) -> pd.DataFrame | None:
        """Download or load cached FINRA TRF data for target_date."""
        cache_file = self.results_dir / "finra_trf_cache.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                _logger.debug("FINRA TRF loaded from cache: %d rows", len(df))
                return df
            except Exception:
                _logger.warning("FINRA TRF cache unreadable")

        try:
            target_dt = datetime.strptime(self.target_date, "%Y-%m-%d")
            # Use previous trading day if target is weekend
            if target_dt.weekday() >= 5:
                target_dt -= timedelta(days=target_dt.weekday() - 4)

            if _download_trf is None:
                return None
            trf_file = _download_trf(target_date=target_dt)
            if trf_file is None:
                return None

            df = pd.read_csv(trf_file, low_memory=False)
            df.to_parquet(cache_file, index=False)
            _logger.info("FINRA TRF downloaded: %d rows", len(df))
            return df
        except Exception:
            _logger.exception("FINRA TRF download failed")
            return None

    @staticmethod
    def _detect_columns(
        df: pd.DataFrame,
    ) -> tuple | None:
        """
        Map FINRA TRF columns to (ticker, short_vol, total_vol).
        Column names vary across FINRA data versions.
        """
        cols = {c.upper().strip(): c for c in df.columns}

        ticker_candidates = ["SYMBOL", "TICKER", "STOCK SYMBOL"]
        short_candidates = ["SHORTSALEVOLUME", "SHORT VOLUME", "SHORT_VOLUME"]
        total_candidates = ["TOTALVOLUME", "TOTAL VOLUME", "TOTAL_VOLUME"]

        ticker_col = next((cols[k] for k in ticker_candidates if k in cols), None)
        short_col = next((cols[k] for k in short_candidates if k in cols), None)
        total_col = next((cols[k] for k in total_candidates if k in cols), None)

        if not all([ticker_col, short_col, total_col]):
            return None
        return ticker_col, short_col, total_col

    # ── Scoring helper ─────────────────────────────────────────────────────

    def compute_score(self, c: Candidate) -> float:
        """
        Compute short_squeeze_score (0–100) for one candidate.

        Thresholds per spec:
          SI < 5%  float → 10
          SI 5–10% float → 20
          SI 10–20%      → 50   (linear interpolation)
          SI > 20%       → 75
          SI > 25%       → 90
          SI ≥ 30%       → 100

        Boosted by +10 if days_to_cover ≥ threshold AND rvol confirms.
        """
        si = c.short_interest_pct_float
        if si is None:
            return 0.0

        cfg = self.config
        if si < 0.05:
            base = 10.0
        elif si < cfg.si_moderate_threshold:
            base = 10.0 + 10.0 * (si - 0.05) / (cfg.si_moderate_threshold - 0.05)
        elif si < cfg.si_high_threshold:
            base = 20.0 + 30.0 * (si - cfg.si_moderate_threshold) / (cfg.si_high_threshold - cfg.si_moderate_threshold)
        elif si < cfg.si_extreme_threshold:
            base = 50.0 + 40.0 * (si - cfg.si_high_threshold) / (cfg.si_extreme_threshold - cfg.si_high_threshold)
        else:
            base = 100.0

        # Volume-confirmed squeeze bonus
        if (
            c.days_to_cover is not None
            and c.days_to_cover >= cfg.days_to_cover_threshold
            and c.relative_volume >= cfg.min_rvol_for_trigger
        ):
            base = min(100.0, base + 10.0)

        # FINRA daily short-vol ratio supplement
        if c.finra_short_vol_ratio is not None and c.finra_short_vol_ratio > 0.50:
            base = min(100.0, base + 5.0)

        return round(base, 1)
