"""
Backward-compatible re-export.

AccumulationAnalyzer has moved to src.ml.pipeline.p06_emps2.accumulation_analyzer.
This module re-exports it so existing P10 imports keep working during the
deprecation window.
"""

import warnings

warnings.warn(
    "Importing AccumulationAnalyzer from p10_emps3 is deprecated. "
    "Use 'from src.ml.pipeline.p06_emps2.accumulation_analyzer import AccumulationAnalyzer' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.ml.pipeline.p06_emps2.accumulation_analyzer import AccumulationAnalyzer  # noqa: F401, E402

__all__ = ["AccumulationAnalyzer"]
