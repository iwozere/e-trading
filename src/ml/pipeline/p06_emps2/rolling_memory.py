"""
Rolling Memory Module - 14-Day Accumulation Tracker

Scans historical daily results to identify:
- Phase 1: Quiet Accumulation (5+ appearances in 14 days)
- Phase 2: Early Public Signal (Phase 1 + volume/sentiment acceleration)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

from src.ml.pipeline.p06_emps2.config import RollingMemoryConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class RollingMemoryScanner:
    """
    Scans historical EMPS2 results to detect accumulation patterns.

    Workflow:
    1. Scan last N days of results folders
    2. Aggregate tickers by appearance frequency
    3. Detect Phase 1 (persistent accumulation)
    4. Detect Phase 2 (acceleration signals)
    5. Generate watchlists and alerts
    """

    def __init__(self, config: RollingMemoryConfig, results_base_path: Path, target_date: str, verbose: bool = True):
        """
        Initialize rolling memory scanner.

        Args:
            config: Rolling memory configuration
            results_base_path: Base path to results/emps2/ folder
            target_date: Target trading date (YYYY-MM-DD) to use as reference
            verbose: Enable verbose logging
        """
        self.config = config
        self.results_base_path = results_base_path
        self.target_date = target_date
        self.verbose = verbose

    def scan_historical_results(self, lookback_days: int | None = None) -> pd.DataFrame:
        """
        Scan historical results from last N days.

        Args:
            lookback_days: Override config lookback period

        Returns:
            DataFrame with columns:
            - scan_date, ticker, market_cap, vol_zscore, vol_rv_ratio,
              atr_ratio, last_price, avg_volume, etc.
        """
        lookback = lookback_days or self.config.lookback_days

        # Calculate date range using target_date as reference
        from datetime import datetime as dt

        today = dt.strptime(self.target_date, "%Y-%m-%d").date()
        start_date = today - timedelta(days=lookback)

        _logger.info("Scanning historical results from %s to %s (%d days)", start_date, today, lookback)

        all_results = []

        # Scan each day's folder
        for days_back in range(lookback + 1):
            scan_date = today - timedelta(days=days_back)
            date_str = scan_date.strftime("%Y-%m-%d")
            day_folder = self.results_base_path / date_str

            if not day_folder.exists():
                _logger.debug("No results for %s", date_str)
                continue

            # P06 (VolatilityFilter) writes 05_volatility_filtered.csv;
            # P10 / accumulation mode writes 07_prebreakout_watchlist.csv.
            vol_file = day_folder / "05_volatility_filtered.csv"
            if not vol_file.exists():
                vol_file = day_folder / "07_prebreakout_watchlist.csv"

            if vol_file.exists():
                try:
                    df = pd.read_csv(vol_file)
                    df["scan_date"] = scan_date
                    all_results.append(df)
                    _logger.debug("Loaded %d tickers from %s (%s)", len(df), date_str, vol_file.name)
                except Exception:
                    _logger.exception("Error loading %s", vol_file)

        if not all_results:
            _logger.warning("No historical results found in last %d days", lookback)
            return pd.DataFrame()

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        _logger.info("Loaded %d total records from %d days", len(combined_df), len(all_results))

        return combined_df

    def calculate_appearance_frequency(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how many times each ticker appeared in lookback period.

        Args:
            historical_df: Combined historical data

        Returns:
            DataFrame with:
            - ticker, appearance_count, first_seen, last_seen,
              avg_vol_zscore, avg_vol_rv_ratio, latest_price, etc.
        """
        if historical_df.empty:
            return pd.DataFrame()

        # Group by ticker and aggregate
        agg_dict = {
            "scan_date": ["count", "min", "max"],
        }

        # Add metrics if they exist
        if "vol_zscore" in historical_df.columns:
            agg_dict["vol_zscore"] = ["mean", "max", "last"]
        if "vol_rv_ratio" in historical_df.columns:
            agg_dict["vol_rv_ratio"] = ["mean", "max", "last"]
        if "atr_ratio" in historical_df.columns:
            agg_dict["atr_ratio"] = ["mean", "last"]
        if "last_price" in historical_df.columns:
            agg_dict["last_price"] = "last"
        if "market_cap" in historical_df.columns:
            agg_dict["market_cap"] = "last"
        if "avg_volume" in historical_df.columns:
            agg_dict["avg_volume"] = "last"

        freq_df = historical_df.groupby("ticker").agg(agg_dict).reset_index()

        # Flatten column names
        new_columns = ["ticker"]
        for col in freq_df.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == "count":
                    new_columns.append("appearance_count")
                elif col[1] == "min":
                    new_columns.append("first_seen")
                elif col[1] == "max" and col[0] == "scan_date":
                    new_columns.append("last_seen")
                elif col[1] == "mean":
                    new_columns.append(f"avg_{col[0]}")
                elif col[1] == "max":
                    new_columns.append(f"max_{col[0]}")
                elif col[1] == "last":
                    new_columns.append(f"latest_{col[0]}")
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)

        freq_df.columns = new_columns

        # Sort by appearance count
        freq_df = freq_df.sort_values("appearance_count", ascending=False)

        _logger.info("Calculated frequency for %d unique tickers", len(freq_df))

        if len(freq_df) > 0:
            _logger.info(
                "Top ticker: %s with %d appearances", freq_df.iloc[0]["ticker"], freq_df.iloc[0]["appearance_count"]
            )

        return freq_df

    def detect_phase1_candidates(self, frequency_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Phase 1: Quiet Accumulation.

        Criteria:
        - Appeared 5+ times (configurable)
        - Persistent pattern over lookback period

        Args:
            frequency_df: Frequency analysis results

        Returns:
            Phase 1 watchlist DataFrame
        """
        if frequency_df.empty:
            return pd.DataFrame()

        phase1_df = frequency_df[frequency_df["appearance_count"] >= self.config.phase1_min_appearances].copy()

        phase1_df["phase"] = "Phase 1: Quiet Accumulation"

        _logger.info(
            "Detected %d Phase 1 candidates (%d+ appearances)", len(phase1_df), self.config.phase1_min_appearances
        )

        return phase1_df

    def detect_phase2_candidates(self, phase1_df: pd.DataFrame, current_scan_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Phase 2: Early Public Signal.

        Criteria (in order applied):
        1. Already in Phase 1 watchlist AND appears in today's scan
        2. [Gate] lag_days <= max_phase2_lag_days  — drop stale signals
        3. Today's vol Z-Score >= phase2_min_vol_zscore  — strong absolute volume
        4. [Gate] vol_acceleration >= phase2_min_vol_acceleration  — volume rising vs avg
        5. Sentiment or virality rising (if data available)
        6. [Gate] pre_alert_drift_pct <= max_pre_alert_drift_pct  — price not already run up

        Priority labels on output:
        - PREMIUM: pre_alert_drift_pct < 0 (price pulled back during accumulation — best signal)
        - HIGH:    all other passing tickers

        Args:
            phase1_df: Phase 1 watchlist (from detect_phase1_candidates)
            current_scan_df: Today's volatility filter output

        Returns:
            Phase 2 alerts DataFrame with lag_days, vol_acceleration, pre_alert_drift_pct columns.
        """
        if phase1_df.empty or current_scan_df.empty:
            _logger.info("No Phase 1 candidates or current scan data for Phase 2 detection")
            return pd.DataFrame()

        # Merge Phase 1 tickers with today's scan results
        merge_cols = ["ticker"]
        for col in ("vol_zscore", "last_price", "sentiment_score", "mentions_24h", "virality_index"):
            if col in current_scan_df.columns:
                merge_cols.append(col)

        phase2_df = phase1_df.merge(
            current_scan_df[merge_cols], on="ticker", how="inner", suffixes=("_history", "_today")
        )

        if phase2_df.empty:
            _logger.info("No Phase 1 tickers found in current scan")
            return pd.DataFrame()

        # ── Gate 1: Lag filter ────────────────────────────────────────────────
        # Stale signals (ticker has been accumulating too long without triggering)
        # perform worse. See docs/TIMING_ANALYSIS.md 2026-05-20.
        from datetime import datetime as _dt

        alert_date = _dt.strptime(self.target_date, "%Y-%m-%d").date()
        phase2_df["lag_days"] = phase2_df["first_seen"].apply(
            lambda fs: (alert_date - fs).days if not pd.isna(fs) else 999
        )
        before_lag = len(phase2_df)
        if self.config.max_phase2_lag_days > 0:
            phase2_df = phase2_df[phase2_df["lag_days"] <= self.config.max_phase2_lag_days].copy()
        _logger.info(
            "Lag gate (<= %d days): %d -> %d tickers", self.config.max_phase2_lag_days, before_lag, len(phase2_df)
        )
        if phase2_df.empty:
            _logger.info("All Phase 2 candidates filtered out by lag gate")
            return pd.DataFrame()

        # ── Derived metrics ───────────────────────────────────────────────────
        # Vol acceleration: today's zscore relative to the historical average.
        # latest_last_price holds the price on first_seen date (oldest entry in the
        # reversed-iteration rolling window — see rolling_memory.scan_historical_results).
        if "vol_zscore" in phase2_df.columns and "avg_vol_zscore" in phase2_df.columns:
            safe_avg = phase2_df["avg_vol_zscore"].replace(0.0, float("nan"))
            phase2_df["vol_acceleration"] = phase2_df["vol_zscore"] / safe_avg

        if "last_price" in phase2_df.columns and "latest_last_price" in phase2_df.columns:
            safe_first = phase2_df["latest_last_price"].replace(0.0, float("nan"))
            phase2_df["pre_alert_drift_pct"] = (phase2_df["last_price"] / safe_first - 1.0) * 100.0

        # ── Signal conditions ─────────────────────────────────────────────────
        conditions = []

        # Absolute vol level: today's zscore must be strong
        if "vol_zscore" in phase2_df.columns:
            conditions.append(phase2_df["vol_zscore"] >= self.config.phase2_min_vol_zscore)

        # Gate 2: Vol acceleration — volume must be rising vs its own history
        # Filters out tickers that had one spike days ago but are now cooling.
        if "vol_acceleration" in phase2_df.columns:
            conditions.append(phase2_df["vol_acceleration"].fillna(0.0) >= self.config.phase2_min_vol_acceleration)

        # Sentiment OR virality (optional enrichment if data is available)
        sentiment_conditions = []
        if "sentiment_score" in phase2_df.columns:
            sentiment_conditions.append(phase2_df["sentiment_score"] >= self.config.phase2_min_sentiment)
        if "virality_index" in phase2_df.columns:
            sentiment_conditions.append(phase2_df["virality_index"] >= self.config.phase2_min_virality)

        # Combine: all hard conditions AND (sentiment OR virality if available)
        if conditions:
            combined = conditions[0]
            for cond in conditions[1:]:
                combined = combined & cond
            if sentiment_conditions:
                sentiment_combined = sentiment_conditions[0]
                for sc in sentiment_conditions[1:]:
                    sentiment_combined = sentiment_combined | sc
                combined = combined & sentiment_combined
            phase2_df = phase2_df[combined].copy()
        elif sentiment_conditions:
            sentiment_combined = sentiment_conditions[0]
            for sc in sentiment_conditions[1:]:
                sentiment_combined = sentiment_combined | sc
            phase2_df = phase2_df[sentiment_combined].copy()
        else:
            _logger.warning("No valid Phase 2 detection conditions available")
            return pd.DataFrame()

        if phase2_df.empty:
            _logger.info("No tickers passed vol/sentiment conditions")
            return pd.DataFrame()

        # ── Gate 3: Price drift filter ────────────────────────────────────────
        # Tickers that have already run up >5% before the alert fire with much lower
        # win rates. See docs/TIMING_ANALYSIS.md 2026-05-20.
        if "pre_alert_drift_pct" in phase2_df.columns and self.config.max_pre_alert_drift_pct > 0:
            before_drift = len(phase2_df)
            drift_ok = (phase2_df["pre_alert_drift_pct"] <= self.config.max_pre_alert_drift_pct) | phase2_df[
                "pre_alert_drift_pct"
            ].isna()
            phase2_df = phase2_df[drift_ok].copy()
            _logger.info(
                "Price drift gate (<= +%.0f%%): %d -> %d tickers",
                self.config.max_pre_alert_drift_pct,
                before_drift,
                len(phase2_df),
            )

        if phase2_df.empty:
            _logger.info("All Phase 2 candidates filtered out by price drift gate")
            return pd.DataFrame()

        phase2_df["phase"] = "Phase 2: Early Public Signal"

        # PREMIUM priority: price pulled back during accumulation (strongest signal)
        if "pre_alert_drift_pct" in phase2_df.columns:
            phase2_df["alert_priority"] = phase2_df["pre_alert_drift_pct"].apply(
                lambda x: "PREMIUM" if (not pd.isna(x) and float(x) < 0.0) else "HIGH"
            )
        else:
            phase2_df["alert_priority"] = "HIGH"

        premium_count = (phase2_df["alert_priority"] == "PREMIUM").sum()
        _logger.info(
            "Phase 2 detection complete: %d candidates (%d PREMIUM, %d HIGH)",
            len(phase2_df),
            premium_count,
            len(phase2_df) - premium_count,
        )

        return phase2_df

    # ── Bootstrap health check ────────────────────────────────────────────────

    @property
    def _state_path(self) -> Path:
        return self.results_base_path / ".rolling_memory_state.json"

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception:
                _logger.warning("Could not read rolling memory state from %s", self._state_path)
        return {}

    def _save_state(self, state: dict) -> None:
        try:
            self._state_path.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception:
            _logger.warning("Could not persist rolling memory state to %s", self._state_path)

    def check_bootstrap_health(
        self,
        phase1_count: int,
        lookback_days: int | None = None,
    ) -> None:
        """
        Persist run state and emit error if Phase 1 has never fired after lookback_days.

        Call this once per pipeline run, passing the number of Phase 1 candidates
        detected in this run.  The method accumulates a lifetime detection count and
        raises an error log when the scanner has been active longer than the lookback
        window without ever producing a Phase 1 signal — which strongly indicates a
        data quality or configuration problem.

        Args:
            phase1_count: Number of Phase 1 candidates detected this run.
            lookback_days: Override config lookback period.
        """
        lookback = lookback_days or self.config.lookback_days

        state = self._load_state()
        if "first_run_date" not in state:
            state["first_run_date"] = self.target_date
        state["last_run_date"] = self.target_date
        state["cumulative_phase1_count"] = state.get("cumulative_phase1_count", 0) + phase1_count
        self._save_state(state)

        first_run = datetime.strptime(state["first_run_date"], "%Y-%m-%d").date()
        today = datetime.strptime(self.target_date, "%Y-%m-%d").date()
        active_days = (today - first_run).days

        _logger.debug(
            "Rolling memory health: active_days=%d, lookback=%d, cumulative_phase1=%d (this_run=%d)",
            active_days,
            lookback,
            state["cumulative_phase1_count"],
            phase1_count,
        )

        if active_days > lookback and state["cumulative_phase1_count"] == 0:
            _logger.error(
                "Rolling memory health check FAILED: scanner has been active %d days "
                "(> lookback %d days) but cumulative Phase 1 detections = 0. "
                "Check data quality, filter thresholds, and results path: %s",
                active_days,
                lookback,
                self.results_base_path,
            )

    def generate_outputs(
        self, frequency_df: pd.DataFrame, phase1_df: pd.DataFrame, phase2_df: pd.DataFrame, output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generate output files.

        Args:
            frequency_df: Rolling candidates frequency data
            phase1_df: Phase 1 watchlist
            phase2_df: Phase 2 alerts
            output_dir: Output directory

        Returns:
            Dict of {file_type: file_path}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = {}

        # 1. Rolling candidates (all tickers in 14-day window)
        if self.config.save_rolling_candidates and not frequency_df.empty:
            rolling_file = output_dir / "06_rolling_candidates.csv"
            frequency_df.to_csv(rolling_file, index=False)
            output_files["rolling_candidates"] = rolling_file
            _logger.info("Saved rolling candidates: %s", rolling_file)

        # 2. Phase 1 watchlist
        if self.config.save_phase1_watchlist and not phase1_df.empty:
            phase1_file = output_dir / "07_phase1_watchlist.csv"
            phase1_df.to_csv(phase1_file, index=False)
            output_files["phase1_watchlist"] = phase1_file
            _logger.info("Saved Phase 1 watchlist: %s", phase1_file)

        # 3. Phase 2 alerts (HOT)
        if self.config.save_phase2_alerts and not phase2_df.empty:
            phase2_file = output_dir / "08_phase2_alerts.csv"
            phase2_df.to_csv(phase2_file, index=False)
            output_files["phase2_alerts"] = phase2_file
            _logger.info("Saved Phase 2 alerts: %s ⚠️", phase2_file)

        return output_files
