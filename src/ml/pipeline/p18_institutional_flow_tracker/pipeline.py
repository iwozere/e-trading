"""
P18 Institutional Flow Tracker — Daily Pipeline Orchestrator

Coordinates all three signal layers:
  Layer 1: New 13F-HR filings today → delta/consensus update
  Layer 2: Form 4 and Schedule 13D/G daily events
  Layer 3: Volume anomaly detection on current watchlist tickers
"""

import logging
import logging.handlers
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.data.data_manager import DataManager
from src.data.downloader.edgar_downloader import EdgarDownloader, EftsUnavailableError
from src.data.downloader.openfigi_mapper import OpenFigiMapper
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config
from src.ml.pipeline.p18_institutional_flow_tracker.processors.position_delta_calculator import PositionDeltaCalculator
from src.ml.pipeline.p18_institutional_flow_tracker.processors.exit_screener import ExitScreener
from src.ml.pipeline.p18_institutional_flow_tracker.processors.consensus_detector import ConsensusDetector
from src.ml.pipeline.p18_institutional_flow_tracker.processors.volume_anomaly_detector import VolumeAnomalyDetector
from src.ml.pipeline.p18_institutional_flow_tracker.processors.form4_monitor import Form4Monitor
from src.ml.pipeline.p18_institutional_flow_tracker.scoring.composite_scorer import CompositeScorer
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class InstitutionalFlowPipeline:
    """
    Daily orchestrator for P18 Institutional Flow Tracker.

    Designed to be run every day at 07:00 UTC by the scheduler.
    During quarterly 13F filing windows the 13F consensus picture grows sharper
    as institutions file.  On non-filing days the pipeline still delivers value
    through Form 4 and volume-anomaly signals.
    """

    def __init__(self, config: Optional[P18Config] = None):
        """
        Args:
            config: Pipeline configuration. Defaults to P18Config.create_default().
        """
        self.config = config or P18Config.create_default()
        self._results_dir = self.config.results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._edgar = EdgarDownloader()
        self._figi = OpenFigiMapper()
        self._delta_calc = PositionDeltaCalculator(results_dir=self._results_dir)
        self._exit_screener = ExitScreener(
            exit_threshold_pct=self.config.exit_threshold_pct,
            min_position_pct_of_portfolio=self.config.min_position_pct_of_portfolio,
            min_position_value_usd=self.config.min_position_value_usd,
        )
        self._consensus = ConsensusDetector(min_institutions=self.config.consensus_min_institutions)
        self._volume = VolumeAnomalyDetector(
            lookback_days=self.config.volume_lookback_days,
            spike_recent_days=self.config.volume_spike_recent_days,
            spike_multiplier=self.config.volume_spike_multiplier,
        )
        self._form4 = Form4Monitor(edgar_downloader=self._edgar)
        self._scorer = CompositeScorer(
            signal_weights=self.config.signal_weights,
            alert_threshold=self.config.score_alert_threshold,
        )

        self._setup_file_logging()

    def run(
        self,
        user_id: Optional[str] = None,
        as_of_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full daily pipeline run.

        Args:
            user_id: User ID passed by the scheduler (stored in result dict).
            as_of_date: Reference date. Defaults to today.
            force_refresh: Force re-download of all cached data.

        Returns:
            Result dict consumed by the scheduler notification_rules engine.
            Keys: success, high_score_count, new_13f_filings_today,
            form4_sells_count, top_ticker, top_score, results_dir, timestamp.
        """
        run_date = as_of_date or date.today()
        run_dir = self._results_dir / str(run_date)
        run_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("=" * 60)
        _logger.info("P18 Institutional Flow — daily run for %s", run_date)
        _logger.info("=" * 60)

        try:
            # ------------------------------------------------------------------
            # Layer 1: New 13F filings today → update quarterly consensus
            # ------------------------------------------------------------------
            efts_available = True
            try:
                new_filings = self._edgar.get_new_13f_filings_today(as_of_date=run_date)
            except EftsUnavailableError:
                _logger.error(
                    "EDGAR EFTS unavailable for %s — will fall back to prior consensus if possible",
                    run_date,
                )
                new_filings = pd.DataFrame(columns=["cik", "institution_name", "accession_number", "filed_date"])
                efts_available = False

            new_13f_count = len(new_filings)
            _logger.info("New 13F-HR filings today: %d", new_13f_count)

            quarter_str, year, quarter = _resolve_current_quarter(run_date)
            consensus_df = self._build_quarterly_consensus(year, quarter, new_filings, force_refresh)

            if consensus_df.empty and not efts_available and run_date.weekday() < 5:
                consensus_df = self._load_consensus_from_results(run_date)
                if not consensus_df.empty:
                    _logger.warning(
                        "EFTS was unavailable — using prior consensus loaded from results directory"
                    )
                else:
                    _logger.error(
                        "EDGAR EFTS unavailable and no prior consensus found in results directory "
                        "for %s — pipeline will produce no signals today",
                        run_date,
                    )

            # ------------------------------------------------------------------
            # Layer 2: Form 4 and 13D/G daily events
            # ------------------------------------------------------------------
            yesterday = run_date - timedelta(days=1)
            form4_df = self._form4.get_significant_sells(as_of_date=yesterday, force_refresh=force_refresh)
            dg_df = self._form4.get_13dg_drops(
                watchlist_tickers=consensus_df["ticker"].tolist() if not consensus_df.empty else [],
                as_of_date=yesterday,
                force_refresh=force_refresh,
            )
            _logger.info("Form 4 sells: %d | 13D/G amendments: %d", len(form4_df), len(dg_df))

            # ------------------------------------------------------------------
            # Layer 3: Volume anomaly on watchlist tickers
            # ------------------------------------------------------------------
            watchlist = _collect_watchlist_tickers(consensus_df, form4_df, self.config.max_tickers_for_volume_check)
            volume_df = self._volume.detect(tickers=watchlist)

            # ------------------------------------------------------------------
            # Scoring
            # ------------------------------------------------------------------
            price_proximity_df = self._build_price_proximity(watchlist, run_date)
            scored_df = self._scorer.score(
                consensus_df=consensus_df,
                volume_df=volume_df,
                form4_df=form4_df,
                dg_df=dg_df,
                as_of_date=run_date,
                price_proximity_df=price_proximity_df,
            )

            # ------------------------------------------------------------------
            # Persist results
            # ------------------------------------------------------------------
            if not scored_df.empty:
                scored_df.to_csv(run_dir / "signals.csv", index=False)
            if not consensus_df.empty:
                consensus_df.to_csv(run_dir / "consensus.csv", index=False)
            if not form4_df.empty:
                form4_df.to_csv(run_dir / "form4_sells.csv", index=False)

            top_ticker = scored_df.iloc[0]["ticker"] if not scored_df.empty else ""
            top_score = int(scored_df.iloc[0]["total_score"]) if not scored_df.empty else 0

            _logger.info(
                "Run complete: %d alerts, top=%s score=%d", len(scored_df), top_ticker, top_score
            )

            return {
                "success": True,
                "high_score_count": len(scored_df),
                "new_13f_filings_today": new_13f_count,
                "form4_sells_count": len(form4_df),
                "top_ticker": top_ticker,
                "top_score": top_score,
                "efts_available": efts_available,
                "results_dir": str(run_dir),
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
            }

        except Exception:
            _logger.exception("P18 pipeline failed for %s", run_date)
            return {
                "success": False,
                "error": "Pipeline execution failed",
                "high_score_count": 0,
                "timestamp": datetime.now().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_quarterly_consensus(
        self,
        year: int,
        quarter: int,
        new_filings: pd.DataFrame,
        force_refresh: bool,
    ) -> pd.DataFrame:
        """
        Download any new 13F filings, compute position deltas, run exit screener
        and consensus detector for the current quarter.

        Returns the consensus DataFrame (may be empty on non-filing days).
        """
        if new_filings.empty:
            # No new filings today — load cached consensus from disk if available
            return self._load_cached_consensus(year, quarter)

        current_holdings_frames = []
        prior_holdings_frames = []
        prior_year, prior_quarter = _prev_quarter(year, quarter)

        for _, row in new_filings.iterrows():
            cik_str = str(row.get("cik", ""))
            acc = str(row.get("accession_number", ""))
            name = str(row.get("institution_name", ""))

            if not cik_str or not acc:
                continue

            try:
                cik_int = int(cik_str)
            except ValueError:
                continue

            # Download current quarter infotable
            path = self._edgar.download_13f_infotable(
                cik=cik_int,
                accession_number=acc,
                year=year,
                quarter=quarter,
                institution_name=name,
                force=force_refresh,
            )
            if path and path.exists():
                df = pd.read_csv(path, compression="gzip")
                if not df.empty and df["value_usd"].sum() >= self.config.min_aum_usd:
                    current_holdings_frames.append(df)

                    # Also try to load prior quarter for delta calc
                    prior_df = self._edgar.load_13f_holdings(cik_int, prior_year, prior_quarter)
                    if prior_df is not None and not prior_df.empty:
                        prior_holdings_frames.append(prior_df)

        if not current_holdings_frames or not prior_holdings_frames:
            return self._load_cached_consensus(year, quarter)

        curr_all = pd.concat(current_holdings_frames, ignore_index=True)
        prev_all = pd.concat(prior_holdings_frames, ignore_index=True)

        # CUSIP → ticker mapping
        cusips = curr_all["cusip"].dropna().unique().tolist()
        mapping = self._figi.map_cusips(cusips)
        curr_all["ticker"] = curr_all["cusip"].map(mapping)
        prev_all["ticker"] = prev_all["cusip"].map(
            self._figi.map_cusips(prev_all["cusip"].dropna().unique().tolist())
        )

        # Drop rows where CUSIP couldn't be mapped
        curr_all = curr_all.dropna(subset=["ticker"])
        prev_all = prev_all.dropna(subset=["ticker"])

        delta_df = self._delta_calc.calculate(curr_all, prev_all)
        exits_df = self._exit_screener.screen(delta_df)
        consensus_df = self._consensus.detect(exits_df)

        # Persist updated consensus for the day
        if not consensus_df.empty:
            cache_path = (
                self._edgar._13f_dir / "consensus" / f"{year}_Q{quarter}.csv.gz"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            consensus_df.to_csv(cache_path, index=False, compression="gzip")

        return consensus_df

    def _build_price_proximity(self, tickers: List[str], as_of_date: date) -> pd.DataFrame:
        """
        Return tickers where the current price is within 15% below the 52-week high.

        A ticker passes when: current_price >= 52w_high * 0.85.  This means
        institutions are exiting before the stock has meaningfully declined —
        a stronger distribution signal than a ticker already down 30%+.

        Args:
            tickers: Watchlist tickers to evaluate.
            as_of_date: Reference date for OHLCV end date.

        Returns:
            DataFrame with columns: ticker, below_52w_high_15pct, current_price, high_52w.
        """
        if not tickers:
            return pd.DataFrame()

        dm = DataManager()
        end = datetime(as_of_date.year, as_of_date.month, as_of_date.day, tzinfo=timezone.utc)
        start = end - timedelta(days=260)

        rows = []
        for ticker in tickers:
            try:
                df = dm.get_ohlcv(ticker, "1d", start, end)
                if df is None or df.empty or "close" not in df.columns or "high" not in df.columns:
                    continue
                high_52w = float(df["high"].max())
                current_price = float(df["close"].iloc[-1])
                if high_52w > 0 and current_price >= high_52w * 0.85:
                    rows.append({
                        "ticker": ticker,
                        "below_52w_high_15pct": True,
                        "current_price": round(current_price, 4),
                        "high_52w": round(high_52w, 4),
                    })
            except Exception:
                _logger.warning("Could not compute 52w proximity for %s", ticker)

        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        _logger.info("Price proximity: %d/%d tickers within 15%% of 52w high", len(result), len(tickers))
        return result

    def _load_cached_consensus(self, year: int, quarter: int) -> pd.DataFrame:
        """Load the most recently computed consensus CSV.gz for a quarter."""
        path = self._edgar._13f_dir / "consensus" / f"{year}_Q{quarter}.csv.gz"
        if path.exists():
            _logger.info("Loading cached consensus from %s", path)
            return pd.read_csv(path, compression="gzip")
        return pd.DataFrame()

    def _load_consensus_from_results(self, before_date: date) -> pd.DataFrame:
        """
        Find the most recent consensus.csv written to a dated results directory.

        Used as a last-resort fallback when the quarterly consensus cache has
        not yet been built (e.g. EFTS has been down since the quarter started).

        Args:
            before_date: Only consider result directories strictly before this date.

        Returns:
            The most recently saved consensus DataFrame, or an empty DataFrame.
        """
        candidates = []
        for entry in self._results_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                dir_date = date.fromisoformat(entry.name)
            except ValueError:
                continue
            if dir_date >= before_date:
                continue
            consensus_path = entry / "consensus.csv"
            if consensus_path.exists():
                candidates.append((dir_date, consensus_path))

        if not candidates:
            return pd.DataFrame()

        best_date, best_path = max(candidates, key=lambda x: x[0])
        age_days = (before_date - best_date).days
        _logger.warning(
            "Loaded fallback consensus from %s (aged %d day(s))", best_path, age_days
        )
        return pd.read_csv(best_path)

    def _setup_file_logging(self) -> None:
        log_file = self._results_dir / "pipeline.log"
        handler = logging.handlers.RotatingFileHandler(
            str(log_file), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        ))
        logging.getLogger("src.ml.pipeline.p18").addHandler(handler)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_current_quarter(ref_date: date) -> tuple:
    """Return (quarter_str, year, quarter_int) for the most recently completed quarter."""
    month = ref_date.month
    year = ref_date.year
    if month <= 3:
        # Q4 of prior year
        return (f"{year - 1}Q4", year - 1, 4)
    if month <= 6:
        return (f"{year}Q1", year, 1)
    if month <= 9:
        return (f"{year}Q2", year, 2)
    return (f"{year}Q3", year, 3)


def _prev_quarter(year: int, quarter: int) -> tuple:
    """Return (year, quarter) for the quarter preceding the given one."""
    if quarter == 1:
        return (year - 1, 4)
    return (year, quarter - 1)


def _collect_watchlist_tickers(
    consensus_df: pd.DataFrame,
    form4_df: pd.DataFrame,
    max_tickers: int,
) -> List[str]:
    """Collect the union of consensus and Form 4 tickers, capped at max_tickers."""
    tickers: List[str] = []
    if not consensus_df.empty and "ticker" in consensus_df.columns:
        tickers += consensus_df["ticker"].dropna().unique().tolist()
    if not form4_df.empty and "ticker" in form4_df.columns:
        tickers += form4_df["ticker"].dropna().unique().tolist()
    return list(dict.fromkeys(t for t in tickers if t))[:max_tickers]
