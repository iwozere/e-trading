"""
Tests for EMPS2Pipeline in accumulation mode (analyzer_type='accumulation').

Verifies:
1. _build_analyzer() returns AccumulationAnalyzer when analyzer_type='accumulation'
2. _build_analyzer() returns VolatilityFilter when analyzer_type='volatility'
3. Stage 3 calls apply_filters() on whichever analyzer was built
4. EMPS3Pipeline emits DeprecationWarning and delegates to EMPS2Pipeline
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p06_emps2.config import EMPS2PipelineConfig


from typing import Literal


def _make_config(analyzer_type: Literal["volatility", "accumulation"] = "volatility") -> EMPS2PipelineConfig:
    cfg = EMPS2PipelineConfig.create_default()
    cfg.analyzer_type = analyzer_type
    return cfg


# ---------------------------------------------------------------------------
# _build_analyzer factory
# ---------------------------------------------------------------------------


def test_build_analyzer_returns_volatility_filter_by_default():
    from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline

    with (
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.NasdaqUniverseDownloader"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.DataManager"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.FundamentalFilter"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.VolatilityFilter") as MockVF,
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.RollingMemoryScanner"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.EMPS2AlertSender"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.SentimentFilter"),
        patch.object(EMPS2Pipeline, "_setup_pipeline_logging"),
    ):
        cfg = _make_config("volatility")
        pipeline = EMPS2Pipeline(cfg)

    MockVF.assert_called_once()


def test_build_analyzer_returns_accumulation_analyzer_when_configured():
    from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline

    with (
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.NasdaqUniverseDownloader"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.DataManager"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.FundamentalFilter"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.VolatilityFilter"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.RollingMemoryScanner"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.EMPS2AlertSender"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.SentimentFilter"),
        patch.object(EMPS2Pipeline, "_setup_pipeline_logging"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.AccumulationAnalyzer", autospec=True) as MockAA,
    ):
        cfg = _make_config("accumulation")
        pipeline = EMPS2Pipeline(cfg)

    MockAA.assert_called_once()


# ---------------------------------------------------------------------------
# Stage 3 delegation
# ---------------------------------------------------------------------------


def test_stage3_calls_apply_filters_on_built_analyzer():
    """_stage3_volatility_filter must delegate to _stage3_analyzer.apply_filters()."""
    import pandas as pd

    from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline

    mock_analyzer = MagicMock()
    mock_analyzer.apply_filters.return_value = pd.DataFrame({"ticker": ["AAPL"]})

    with (
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.NasdaqUniverseDownloader"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.DataManager"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.FundamentalFilter"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.VolatilityFilter"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.RollingMemoryScanner"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.EMPS2AlertSender"),
        patch("src.ml.pipeline.p06_emps2.emps2_pipeline.SentimentFilter"),
        patch.object(EMPS2Pipeline, "_setup_pipeline_logging"),
        patch.object(EMPS2Pipeline, "_build_analyzer", return_value=mock_analyzer),
    ):
        pipeline = EMPS2Pipeline(_make_config())
        result = pipeline._stage3_volatility_filter(["AAPL", "MSFT"])

    mock_analyzer.apply_filters.assert_called_once_with(["AAPL", "MSFT"])
    assert list(result["ticker"]) == ["AAPL"]


# ---------------------------------------------------------------------------
# P10 deprecation shim
# ---------------------------------------------------------------------------


def test_emps3_pipeline_emits_deprecation_warning():
    """Instantiating EMPS3Pipeline must raise DeprecationWarning."""
    with (
        patch("src.ml.pipeline.p10_emps3.emps3_pipeline.EMPS2Pipeline"),
        pytest.warns(DeprecationWarning, match="EMPS3Pipeline is deprecated"),
    ):
        from src.ml.pipeline.p10_emps3.emps3_pipeline import EMPS3Pipeline

        EMPS3Pipeline()


def test_emps3_pipeline_run_delegates_to_emps2():
    """EMPS3Pipeline.run() must call the delegate EMPS2Pipeline.run()."""
    import pandas as pd

    mock_emps2 = MagicMock()
    mock_emps2.run.return_value = pd.DataFrame({"ticker": ["X"]})

    with (
        patch("src.ml.pipeline.p10_emps3.emps3_pipeline.EMPS2Pipeline", return_value=mock_emps2),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        from src.ml.pipeline.p10_emps3.emps3_pipeline import EMPS3Pipeline

        p = EMPS3Pipeline()
        result = p.run(force_refresh=True)

    mock_emps2.run.assert_called_once_with(force_refresh=True)
    assert list(result["ticker"]) == ["X"]
