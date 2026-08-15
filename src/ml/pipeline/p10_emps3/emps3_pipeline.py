"""
EMPS3Pipeline — deprecated.

AccumulationAnalyzer has been merged into EMPS2Pipeline (P06) as a configurable
Stage 3 analyzer.  Use EMPS2Pipeline with analyzer_type='accumulation' instead.

This shim is kept for backward-compatibility during the deprecation window.
Remove once no scheduler or caller references this entrypoint directly.
"""

import warnings

import pandas as pd

from src.ml.pipeline.p06_emps2.config import EMPS2PipelineConfig, RollingMemoryConfig
from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DEPRECATION_MSG = (
    "EMPS3Pipeline is deprecated. "
    "Use EMPS2Pipeline with analyzer_type='accumulation' instead: "
    "cfg = EMPS2PipelineConfig.create_default(); cfg.analyzer_type = 'accumulation'; "
    "EMPS2Pipeline(cfg)."
)


def _p10_rolling_memory_config() -> RollingMemoryConfig:
    """
    Phase 2 gate config for p10_emps3, decoupled from p06_emps2's RollingMemoryConfig.

    Was silently sharing p06_emps2's defaults (see docs/TIMING_ANALYSIS.md 2026-08-14
    re-measurement). `max_phase2_lag_days` inherits p06's fix — with
    `phase1_min_appearances=5` inside a 14-day lookback, a ticker typically doesn't
    qualify for Phase 1 until 8-14 days after first_seen, so a 7-day lag cap made
    Phase 2 almost unreachable for both pipelines (mechanical bug, not a quality issue).

    The remaining gates (vol_zscore/vol_acceleration/drift) are left at p06's defaults
    on purpose: a grid-sweep across p10's own production history (2026-06-02 ->
    2026-08-13) found no combination that beat these defaults on 10d forward returns —
    every looser variant traded volume for a worse (more negative) mean return. p10's
    "coiled spring" thesis is fundamentally about catching a breakout *before* it
    spikes, while this gate confirms one *after* it spikes (vol_zscore >= 3.0); no
    amount of threshold tuning closes that gap. See p10_emps3/docs/Tasks.md.
    """
    return RollingMemoryConfig(max_phase2_lag_days=10)


class EMPS3Pipeline:
    """
    Deprecated shim — delegates to EMPS2Pipeline(analyzer_type='accumulation').
    """

    def __init__(self, config: EMPS2PipelineConfig | None = None, target_date: str | None = None):
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        _logger.warning("EMPS3Pipeline is deprecated. %s", _DEPRECATION_MSG)

        from pathlib import Path

        emps2_config = config if config is not None else EMPS2PipelineConfig.create_default()
        emps2_config.analyzer_type = "accumulation"
        emps2_config.rolling_memory_config = _p10_rolling_memory_config()

        self._delegate = EMPS2Pipeline(
            emps2_config,
            target_date=target_date,
            results_base=Path("results") / "p10_emps3",
        )

    @property
    def target_date(self) -> str:
        return self._delegate.target_date

    @property
    def results_dir(self):
        return self._delegate._results_dir

    def run(self, force_refresh: bool = False, tickers: list | None = None) -> pd.DataFrame:
        if tickers is not None:
            _logger.warning(
                "EMPS3Pipeline shim: 'tickers' filter is not supported by the EMPS2 delegate "
                "and will be ignored (%d tickers passed).",
                len(tickers),
            )
        return self._delegate.run(force_refresh=force_refresh)
