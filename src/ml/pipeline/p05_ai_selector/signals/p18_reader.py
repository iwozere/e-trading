"""P18 signal reader — loads today's P18 output CSVs for Stage 2 score boost."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.config import P18_HIGH_SCORE_THRESHOLD, P18_RESULTS_BASE
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_EMPTY_RESULT: Dict[str, Any] = {
    "high_score_count": 0,
    "tickers": {},
    "consensus_tickers": set(),
    "form4_buy_tickers": set(),
    "13dg_tickers": set(),
}


class P18Reader:
    """Reads P18 Institutional Flow Tracker outputs for a given date."""

    def __init__(self, results_base: Path | None = None):
        self._results_base = results_base or P18_RESULTS_BASE

    def get_high_score_tickers(self, as_of_date: date) -> Dict[str, Any]:
        """
        Load P18 outputs for the most recent run on or before as_of_date.

        Returns:
            Dict with keys: high_score_count, tickers, consensus_tickers,
            form4_buy_tickers, 13dg_tickers. All empty/zero when no data found.
        """
        run_dir = self._find_most_recent_dir(as_of_date)
        if run_dir is None:
            _logger.info("P18Reader: no results found on or before %s", as_of_date)
            return dict(_EMPTY_RESULT)

        _logger.info("P18Reader: reading from %s", run_dir)
        result: Dict[str, Any] = {
            "high_score_count": 0,
            "tickers": {},
            "consensus_tickers": set(),
            "form4_buy_tickers": set(),
            "13dg_tickers": set(),
        }

        signals_file = run_dir / "signals.csv"
        if signals_file.exists():
            try:
                df = pd.read_csv(signals_file)
                if "ticker" in df.columns and "total_score" in df.columns:
                    high = df[df["total_score"] >= P18_HIGH_SCORE_THRESHOLD]
                    result["tickers"] = dict(zip(high["ticker"], high["total_score"]))
                    result["high_score_count"] = len(high)
                    _logger.info("P18Reader: %d high-score tickers (threshold=%d)", len(high), P18_HIGH_SCORE_THRESHOLD)
            except Exception:
                _logger.exception("P18Reader: error reading signals.csv")

        consensus_file = run_dir / "consensus.csv"
        if consensus_file.exists():
            try:
                df = pd.read_csv(consensus_file)
                if "ticker" in df.columns:
                    result["consensus_tickers"] = set(df["ticker"].dropna().tolist())
            except Exception:
                _logger.exception("P18Reader: error reading consensus.csv")

        # form4_buy_tickers: bullish insider-buy signal for Stage 2 (+form4_insider_buy).
        # Sourced from a dedicated form4_buys.csv. P18 currently persists only
        # significant SELLS (form4_sells.csv), so this file is absent in production
        # today and the set stays empty until P18 is extended to capture buy-side
        # Form 4 filings — but the reader is ready for it.
        form4_buys_file = run_dir / "form4_buys.csv"
        if form4_buys_file.exists():
            try:
                df = pd.read_csv(form4_buys_file)
                if "ticker" in df.columns:
                    result["form4_buy_tickers"] = set(df["ticker"].dropna().tolist())
            except Exception:
                _logger.exception("P18Reader: error reading form4_buys.csv")

        _logger.info(
            "P18Reader: loaded — scores=%d, consensus=%d",
            result["high_score_count"],
            len(result["consensus_tickers"]),
        )
        return result

    def _find_most_recent_dir(self, as_of_date: date) -> Path | None:
        """Find the most recent P18 results directory on or before as_of_date."""
        if not self._results_base.exists():
            return None

        candidates = []
        for entry in self._results_base.iterdir():
            if not entry.is_dir():
                continue
            try:
                dir_date = date.fromisoformat(entry.name)
                if dir_date <= as_of_date:
                    candidates.append(dir_date)
            except ValueError:
                pass

        if not candidates:
            return None

        best = max(candidates)
        return self._results_base / str(best)
