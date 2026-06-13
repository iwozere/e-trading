"""
EMPS3Pipeline — deprecated.

AccumulationAnalyzer has been merged into EMPS2Pipeline (P06) as a configurable
Stage 3 analyzer.  Use EMPS2Pipeline with analyzer_type='accumulation' instead.

This shim is kept for backward-compatibility during the deprecation window.
Remove once no scheduler or caller references this entrypoint directly.
"""

import warnings
from typing import Optional

import pandas as pd

from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline
from src.ml.pipeline.p06_emps2.config import EMPS2PipelineConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DEPRECATION_MSG = (
    "EMPS3Pipeline is deprecated. "
    "Use EMPS2Pipeline with analyzer_type='accumulation' instead: "
    "cfg = EMPS2PipelineConfig.create_default(); cfg.analyzer_type = 'accumulation'; "
    "EMPS2Pipeline(cfg)."
)


class EMPS3Pipeline:
    """
    Deprecated shim — delegates to EMPS2Pipeline(analyzer_type='accumulation').
    """

    def __init__(self, config=None, target_date: Optional[str] = None):
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        _logger.warning("EMPS3Pipeline is deprecated. %s", _DEPRECATION_MSG)

        from pathlib import Path
        emps2_config = EMPS2PipelineConfig.create_default()
        emps2_config.analyzer_type = "accumulation"

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

    def run(self, force_refresh: bool = False, tickers: Optional[list] = None) -> pd.DataFrame:
        if tickers is not None:
            _logger.warning(
                "EMPS3Pipeline shim: 'tickers' filter is not supported by the EMPS2 delegate "
                "and will be ignored (%d tickers passed).",
                len(tickers),
            )
        return self._delegate.run(force_refresh=force_refresh)
