"""
Accumulation Analyzer (Stage C)

Replaces VolatilityFilter for the EMPS3 pipeline.
Implements the Coiled Spring strategy formula.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import talib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.data_manager import DataManager
from src.ml.pipeline.p10_emps3.config import EMPS3FilterConfig
from src.ml.pipeline.shared.ohlcv_timestamp import coerce_ohlcv_timestamp_column
from src.ml.pipeline.shared.trf_downloader import download_trf

_logger = setup_logger(__name__)


class AccumulationAnalyzer:
    """
    Identifies the 'Coiled Spring' effect.
    High volume buying hidden by low price volatility.
    """

    def __init__(self, data_manager: DataManager, config: EMPS3FilterConfig, results_dir: Path,
                 target_date: Optional[str] = None,
                 chunk_size: int = 50,
                 checkpoint_enabled: bool = True):
        """
        Args:
            chunk_size: number of tickers to download+process per chunk. Keeps
                peak memory bounded on constrained hosts (Pi) and allows
                incremental checkpointing after every chunk.
            checkpoint_enabled: if True, diagnostics are persisted after each
                chunk to ``accumulation_checkpoint.csv`` so an interrupted run
                (OOM-kill, SIGTERM, crash) can resume and skip already-processed
                tickers on the next run.
        """
        self.data_manager = data_manager
        self.config = config

        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        self.target_date = target_date

        self._results_dir = results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = max(1, int(chunk_size))
        self.checkpoint_enabled = bool(checkpoint_enabled)
        self._checkpoint_path = self._results_dir / "accumulation_checkpoint.csv"

        _logger.info(
            "Accumulation Analyzer initialized: AR>%.1f, Vol Z-Score>%.1f, "
            "chunk_size=%d, checkpoint=%s",
            config.min_vol_rv_ratio, config.min_vol_zscore,
            self.chunk_size, self.checkpoint_enabled,
        )

    def apply_filters(self, tickers: List[str]) -> pd.DataFrame:
        _logger.info("Applying Accumulation Analyzer to %d tickers", len(tickers))

        end_date = datetime.strptime(self.target_date, '%Y-%m-%d') + timedelta(days=1)
        # Intraday (5m/15m/30m/1h) is typically capped to ~60 days on Yahoo.
        start_date_intraday = end_date - timedelta(days=60)
        # Daily needs ~1y for 52w high and 12m BB min.
        start_date_daily = end_date - timedelta(days=365)
        trf_date = datetime.strptime(self.target_date, '%Y-%m-%d')

        # Resume from checkpoint if present.
        diagnostic_data, processed_tickers = self._load_checkpoint()
        if processed_tickers:
            _logger.info(
                "Resuming from checkpoint: %d tickers already processed (%d remaining)",
                len(processed_tickers), len(tickers) - len(processed_tickers),
            )
        remaining = [t for t in tickers if t not in processed_tickers]

        # Load TRF data once for all tickers — avoids per-ticker re-reads and re-downloads.
        trf_factors = self._load_trf_factors(trf_date)

        total = len(tickers)
        chunks = [remaining[i:i + self.chunk_size] for i in range(0, len(remaining), self.chunk_size)]

        try:
            for chunk_idx, chunk in enumerate(chunks, start=1):
                _logger.info(
                    "Chunk %d/%d: processing %d tickers (%d/%d done so far)",
                    chunk_idx, len(chunks), len(chunk), len(processed_tickers), total,
                )

                # Download OHLCV for just this chunk. A chunk failure is
                # contained: all of its tickers are marked ERROR, pipeline
                # continues.
                try:
                    ohlcv_intraday = self.data_manager.get_ohlcv_batch(
                        chunk, self.config.interval, start_date_intraday, end_date,
                    )
                    ohlcv_daily = self.data_manager.get_ohlcv_batch(
                        chunk, "1d", start_date_daily, end_date,
                    )
                except Exception as e:
                    _logger.exception(
                        "OHLCV batch download failed for chunk %d (size=%d); marking as errors",
                        chunk_idx, len(chunk),
                    )
                    for ticker in chunk:
                        diagnostic_data.append({
                            'ticker': ticker,
                            'status': 'ERROR',
                            'reason': f'ohlcv_download_failed: {e.__class__.__name__}',
                        })
                        processed_tickers.add(ticker)
                    self._save_checkpoint(diagnostic_data)
                    continue

                for ticker in chunk:
                    try:
                        df_intra = self._coerce_ohlcv_timestamp_column(ohlcv_intraday.get(ticker))
                        df_daily = self._coerce_ohlcv_timestamp_column(ohlcv_daily.get(ticker))

                        if df_intra is None or df_intra.empty or df_daily is None or df_daily.empty:
                            diagnostic_data.append({'ticker': ticker, 'status': 'FAILED', 'reason': 'no_data'})
                            continue

                        if len(df_intra) < 20 or len(df_daily) < 20:
                            diagnostic_data.append({'ticker': ticker, 'status': 'FAILED', 'reason': 'insufficient_bars'})
                            continue

                        trf_factor = trf_factors.get(ticker.upper(), 1.0)
                        if trf_factor != 1.0:
                            df_intra = self._apply_trf_volume_correction(df_intra, trf_factor)
                            df_daily = self._apply_trf_volume_correction(df_daily, trf_factor)
                        else:
                            trf_factor = None

                        passed, metrics, reason = self._check_accumulation(ticker, df_intra, df_daily)
                        metrics['trf_correction_factor'] = trf_factor

                        diag_entry = metrics.copy()
                        if passed:
                            diag_entry['status'] = 'PASSED'
                            diag_entry['reason'] = 'all_filters_passed'
                        else:
                            diag_entry['status'] = 'FAILED'
                            diag_entry['reason'] = reason
                        diagnostic_data.append(diag_entry)

                    except Exception as e:
                        _logger.exception("Error processing %s", ticker)
                        diagnostic_data.append({'ticker': ticker, 'status': 'ERROR', 'reason': str(e)})
                    finally:
                        processed_tickers.add(ticker)

                # Release chunk-scoped references before next iteration (Pi RAM hygiene).
                ohlcv_intraday = None
                ohlcv_daily = None

                # Per-chunk checkpoint + progress log.
                self._save_checkpoint(diagnostic_data)
                passed_so_far = sum(1 for d in diagnostic_data if d.get('status') == 'PASSED')
                _logger.info(
                    "Progress: %d/%d tickers (%.1f%%), passed so far: %d",
                    len(processed_tickers), total,
                    100.0 * len(processed_tickers) / max(1, total),
                    passed_so_far,
                )

        except KeyboardInterrupt:
            _logger.warning("Interrupted by user; flushing checkpoint so the next run can resume.")
            self._save_checkpoint(diagnostic_data)
            raise
        except Exception:
            _logger.exception("Unhandled error in accumulation analyzer; flushing checkpoint.")
            self._save_checkpoint(diagnostic_data)
            raise

        # Normal completion — persist final outputs, then clear checkpoint.
        results_data = [d for d in diagnostic_data if d.get('status') == 'PASSED']
        self._save_diagnostics(diagnostic_data)
        self._save_results(results_data)
        self._clear_checkpoint()

        return pd.DataFrame(results_data)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _save_checkpoint(self, diagnostic_data: list) -> None:
        if not self.checkpoint_enabled or not diagnostic_data:
            return
        try:
            pd.DataFrame(diagnostic_data).to_csv(self._checkpoint_path, index=False)
        except Exception:
            _logger.exception("Failed to write accumulation checkpoint (non-fatal)")

    def _load_checkpoint(self) -> tuple:
        """Return (diagnostic_data list, processed_tickers set)."""
        if not self.checkpoint_enabled or not self._checkpoint_path.exists():
            return [], set()
        try:
            df = pd.read_csv(self._checkpoint_path)
            if df.empty or 'ticker' not in df.columns:
                return [], set()
            records = df.to_dict(orient='records')
            processed = {str(t).upper() for t in df['ticker'].dropna().astype(str)}
            return records, processed
        except Exception:
            _logger.exception("Failed to read accumulation checkpoint; starting fresh")
            return [], set()

    def _clear_checkpoint(self) -> None:
        if not self.checkpoint_enabled:
            return
        try:
            if self._checkpoint_path.exists():
                self._checkpoint_path.unlink()
        except Exception:
            _logger.exception("Failed to remove accumulation checkpoint (non-fatal)")

    def _check_accumulation(self, ticker: str, df_intra: pd.DataFrame, df_daily: pd.DataFrame) -> tuple:
        df_intra = df_intra.sort_values('timestamp').copy()
        df_daily = df_daily.sort_values('timestamp').copy()
        
        last_price = df_intra['close'].iloc[-1]
        
        # Calculate ATR and Price Range 1d
        atr_14 = talib.ATR(df_daily['high'].values, df_daily['low'].values, df_daily['close'].values, timeperiod=self.config.atr_period)
        atr = float(atr_14[-1]) if len(atr_14) > 0 and not np.isnan(atr_14[-1]) else 0.0
        atr_ratio = atr / last_price if last_price > 0 else 0.0
        
        price_range_1d = (df_daily['high'].iloc[-1] - df_daily['low'].iloc[-1]) / df_daily['low'].iloc[-1] if df_daily['low'].iloc[-1] > 0 else 0.0

        # Calculate Vol Z-Score on daily
        vol_zscore = self._compute_zscore(df_daily['volume'].values, 20)
        
        # Calculate RV using intraday (last 5 days)
        # Approximate 5 days in intraday (e.g. 1h interval = 6.5 bars * 5 approx 33 bars)
        interval_map = {'5m': 78, '15m': 26, '30m': 13, '1h': 6.5}
        bars_per_day = interval_map.get(self.config.interval, 26)
        rv_bars_target = int(bars_per_day * 5)
        if len(df_intra) >= rv_bars_target:
            close_intra = df_intra['close'].values[-rv_bars_target:]
            log_returns = np.log(close_intra[1:] / close_intra[:-1])
            rv = np.std(log_returns) * np.sqrt(252 * bars_per_day)
        else:
            rv = 0.0
            
        ar = vol_zscore / rv if (rv > 0 and vol_zscore > 0) else 0.0
        
        # Squeeze Logic
        daily_prev = df_daily.iloc[-2]
        daily_curr = df_daily.iloc[-1]
        
        inside_day = daily_curr['high'] < daily_prev['high'] and daily_curr['low'] > daily_prev['low']
        
        upper, middle, lower = talib.BBANDS(df_daily['close'].values, timeperiod=20)
        bb_width = (upper - lower) / middle
        bb_current_width = float(bb_width[-1]) if len(bb_width) > 0 and not np.isnan(bb_width[-1]) else 1.0
        bb_12m_min = float(np.nanmin(bb_width[-252:])) if len(bb_width) > 0 else 0.0
        bb_squeeze = bb_current_width <= (bb_12m_min * 1.1)  # allow 10% tolerance for "minimum"
        
        avg_vol_20 = np.mean(df_daily['volume'].values[-20:])
        vol_ratio = daily_curr['volume'] / avg_vol_20 if avg_vol_20 > 0 else 0.0
        
        ranges = df_daily['high'].values - df_daily['low'].values
        avg_range_20 = np.mean(ranges[-20:])
        range_ratio = (daily_curr['high'] - daily_curr['low']) / avg_range_20 if avg_range_20 > 0 else 0.0
        
        vr_divergence = vol_ratio > 1.5 and range_ratio < 0.8
        
        squeeze_state = inside_day or bb_squeeze or vr_divergence
        
        # TRF Surge Logic (Mocked approximation as we assume it's applied to volume)
        trf_surge = False # Needs actual rolling 3-day window of trf factor check. For now, we will flag it if ar is very high.
        
        # Pre-breakout exclusion rules
        price_change_1d = abs(daily_curr['close'] - daily_prev['close']) / daily_prev['close']
        sma_20 = np.mean(df_daily['close'].values[-20:])
        dist_sma_20 = abs(daily_curr['close'] - sma_20) / sma_20
        high_52w = np.max(df_daily['high'].values[-252:]) if len(df_daily) > 0 else daily_curr['high']
        dist_52w_high = (high_52w - daily_curr['close']) / high_52w if high_52w > 0 else 0.0
        
        # Scoring
        score = 0
        if ar > 2.5: score += 30
        if bb_current_width < 0.05: score += 20
        # Resistance pressing: Price within 2% of 20-day High
        high_20 = np.max(df_daily['high'].values[-20:])
        if (high_20 - daily_curr['close']) / high_20 < 0.02: score += 30
        # Virality mocked or fetched: missing sentiment here so +0 for now.

        dist_local_high = (high_20 - daily_curr['close']) / high_20 if high_20 > 0 else 0.0

        metrics = {
            'ticker': ticker,
            'last_price': float(last_price),
            'vol_zscore': float(vol_zscore),
            'rv': float(rv),
            'absorption_ratio': float(ar),
            'atr_ratio': atr_ratio,
            'price_range_1d': price_range_1d,
            'inside_day': bool(inside_day),
            'bb_squeeze': bool(bb_squeeze),
            'vr_divergence': bool(vr_divergence),
            'squeeze_state': bool(squeeze_state),
            'price_change_1d': float(price_change_1d),
            'dist_sma_20': float(dist_sma_20),
            'dist_52w_high': float(dist_52w_high),
            'dist_local_high': float(dist_local_high),
            'prebreakout_score': score,
            # for rolling memory Trend tracking
            'volume': float(daily_curr['volume'])
        }

        # NaN guard — reject before any comparison (NaN <= X evaluates to False in Python,
        # causing tickers with missing data to silently pass all filters).
        if any(np.isnan(v) for v in [vol_zscore, rv, ar, price_range_1d, atr_ratio]):
            return False, metrics, 'nan_metrics'

        # Validate conditions
        # 1. Volume Presence
        if vol_zscore <= self.config.min_vol_zscore:
            return False, metrics, 'low_volume_zscore'

        # 2. Price Compression
        if price_range_1d >= self.config.max_price_impact or atr_ratio >= self.config.max_atr_ratio:
            return False, metrics, 'poor_price_compression'

        # 3. Absorption
        if ar <= self.config.min_vol_rv_ratio:
            return False, metrics, 'low_absorption_ratio'

        # Exclusions
        if price_change_1d > 0.035:
            return False, metrics, 'price_change_too_high'
        if dist_sma_20 > self.config.max_distance_from_sma20:
            return False, metrics, 'too_far_from_sma20'

        # Inclusions: must be pressing local resistance (20-day high)
        if dist_local_high > self.config.max_distance_from_resistance:
            return False, metrics, 'too_far_from_local_high'
            
        if score > 70:
            metrics['prebreakout_watchlist'] = True

        return True, metrics, 'passed'

    def _compute_zscore(self, series: np.ndarray, window: int) -> float:
        if len(series) < window: return 0.0
        mean = np.mean(series[-window:])
        std = np.std(series[-window:])
        if std == 0: return 0.0
        return (series[-1] - mean) / std

    def _coerce_ohlcv_timestamp_column(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Normalize OHLCV DataFrame to include a canonical timestamp column."""
        return coerce_ohlcv_timestamp_column(df)

    def _load_trf_factors(self, trf_date: datetime) -> Dict[str, float]:
        """Load TRF correction factors for all tickers into memory once."""
        try:
            trf_path = download_trf(target_date=trf_date)
            if not trf_path.exists():
                return {}
            df = pd.read_csv(trf_path)
            if df.empty or "ticker" not in df.columns:
                return {}
            factors: Dict[str, float] = {}
            for row in df.itertuples(index=False):
                ticker = str(getattr(row, "ticker", "")).upper()
                total = getattr(row, "total_volume", 0)
                short = getattr(row, "short_volume", 0)
                if total and total > 0 and short < total:
                    factors[ticker] = total / (total - short)
            _logger.info("Loaded TRF correction factors for %d tickers", len(factors))
            return factors
        except Exception:
            _logger.exception("Failed to load TRF data — proceeding without volume corrections")
            return {}

    def _load_trf_volume_corrections(self) -> Dict[str, float]:
        trf_date = datetime.now() - timedelta(days=self.config.lookback_days)
        return self._load_trf_factors(trf_date)

    def _apply_trf_volume_correction(self, df: pd.DataFrame, correction_factor: float) -> pd.DataFrame:
        df = coerce_ohlcv_timestamp_column(df.copy())
        if df is None or df.empty or "timestamp" not in df.columns:
            return df
        today = datetime.now().date()
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        historical_mask = ts.dt.date < today
        if historical_mask.any():
            df.loc[historical_mask, "volume"] = df.loc[historical_mask, "volume"] * correction_factor
        return df

    def _save_diagnostics(self, diagnostic_data: list):
        if not diagnostic_data: return
        df = pd.DataFrame(diagnostic_data)
        # Ensure 'reason' and 'status' are present
        out = self._results_dir / "08_absorption_diagnostics.csv"
        df.to_csv(out, index=False)
        _logger.info("Saved diagnostics to %s", out)

    def _save_results(self, results_data: list):
        if not results_data: return
        df = pd.DataFrame(results_data)
        out = self._results_dir / "07_prebreakout_watchlist.csv"
        # Save full results - don't silently drop rows here
        df.to_csv(out, index=False)
        
        high_priority_count = len(df[df.get('prebreakout_score', 0) > 70])
        _logger.info("Saved %d candidates to %s (%d high priority)", len(df), out, high_priority_count)
