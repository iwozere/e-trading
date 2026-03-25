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
from src.ml.pipeline.shared.trf_downloader import download_trf, get_trf_correction_factor

_logger = setup_logger(__name__)


class AccumulationAnalyzer:
    """
    Identifies the 'Coiled Spring' effect.
    High volume buying hidden by low price volatility.
    """

    def __init__(self, data_manager: DataManager, config: EMPS3FilterConfig, results_dir: Path, target_date: Optional[str] = None):
        self.data_manager = data_manager
        self.config = config
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        self.target_date = target_date

        self._results_dir = results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Accumulation Analyzer initialized: AR>%.1f, Vol Z-Score>%.1f",
                     config.min_vol_rv_ratio, config.min_vol_zscore)

    def apply_filters(self, tickers: List[str]) -> pd.DataFrame:
        try:
            _logger.info("Applying Accumulation Analyzer to %d tickers", len(tickers))

            from datetime import datetime as dt
            # Need a longer lookback for 52w high and 20d SMA, and 12-m BB minimum.
            # Intraday data won't cover 52 weeks. For 52w high and SMA20, we may need daily data, 
            # but we can try to get it via Yahoo downloader.
            # However, the downloader.get_ohlcv_batch interval is 1h usually. 
            # We will use intraday for recent logic and rely on the downloaded data.
            # Since 12-month BB Width minimum requires 1 year of daily data, let's fetch daily data as well.
            
            # Fetch intraday data for recent metrics
            end_date = dt.strptime(self.target_date, '%Y-%m-%d') + timedelta(days=1)
            # Intraday 15m/1h limited to 60 days usually in yfinance. 
            # We will use 60 days for intraday.
            start_date_intraday = end_date - timedelta(days=60)
            
            _logger.info("Downloading %s data for %d tickers", self.config.interval, len(tickers))
            ohlcv_intraday = self.data_manager.get_ohlcv_batch(tickers, self.config.interval, start_date_intraday, end_date)
            
            # Fetch daily data for 52-week High, 20-day SMA, 12-month BB
            start_date_daily = end_date - timedelta(days=365)
            ohlcv_daily = self.data_manager.get_ohlcv_batch(tickers, "1d", start_date_daily, end_date)
            
            # target_date for TRF is usually yesterday
            trf_date = dt.strptime(self.target_date, '%Y-%m-%d')
            
            passed_tickers = []
            results_data = []
            diagnostic_data = []

            for ticker in tickers:
                try:
                    df_intra = ohlcv_intraday.get(ticker)
                    df_daily = ohlcv_daily.get(ticker)

                    if df_intra is None or df_intra.empty or df_daily is None or df_daily.empty:
                        # Log failure
                        diagnostic_data.append({'ticker': ticker, 'status': 'FAILED', 'reason': 'no_data'})
                        continue

                    if len(df_intra) < 20 or len(df_daily) < 20:
                        diagnostic_data.append({'ticker': ticker, 'status': 'FAILED', 'reason': 'insufficient_bars'})
                        continue

                    trf_factor = get_trf_correction_factor(ticker, trf_date)
                    if trf_factor != 1.0:
                        df_intra = self._apply_trf_volume_correction(df_intra, trf_factor)
                        df_daily = self._apply_trf_volume_correction(df_daily, trf_factor)
                    else:
                        trf_factor = None

                    passed, metrics, reason = self._check_accumulation(ticker, df_intra, df_daily)
                    metrics['trf_correction_factor'] = trf_factor

                    if passed:
                        passed_tickers.append(ticker)
                        results_data.append(metrics)
                        diag_entry = metrics.copy()
                        diag_entry['status'] = 'PASSED'
                        diag_entry['reason'] = 'all_filters_passed'
                        diagnostic_data.append(diag_entry)
                    else:
                        diag_entry = metrics.copy()
                        diag_entry['status'] = 'FAILED'
                        diag_entry['reason'] = reason
                        diagnostic_data.append(diag_entry)
                        
                except Exception as e:
                    _logger.exception("Error processing %s", ticker)
                    diagnostic_data.append({'ticker': ticker, 'status': 'ERROR', 'reason': str(e)})

            # Save results
            self._save_diagnostics(diagnostic_data)
            self._save_results(results_data)

            # Return the DataFrame of results
            return pd.DataFrame(results_data)

        except Exception:
            _logger.exception("Error applying accumulation filters:")
            return []

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
            
        ar = vol_zscore / rv if rv > 0 else 0.0
        
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
            'prebreakout_score': score,
            # for rolling memory Trend tracking
            'volume': float(daily_curr['volume'])
        }

        # Validate conditions
        # 1. Volume Presence
        if vol_zscore <= self.config.min_vol_zscore:
            return False, metrics, 'low_volume_zscore'
        
        # 2. Price Compression
        if price_range_1d >= self.config.max_price_impact or atr_ratio >= 0.02:
            return False, metrics, 'poor_price_compression'
            
        # 3. Absorption
        if ar <= self.config.min_vol_rv_ratio:
            return False, metrics, 'low_absorption_ratio'
            
        # Exclusions
        if price_change_1d > 0.035:
            return False, metrics, 'price_change_too_high'
        if dist_sma_20 > self.config.max_distance_from_sma20:
            return False, metrics, 'too_far_from_sma20'
        
        # Inclusions: must be close to resistance (we use 3% of 52-week high here)
        if dist_52w_high > self.config.max_distance_from_resistance:
            return False, metrics, 'too_far_from_52w_high'
            
        if score > 70:
            metrics['prebreakout_watchlist'] = True

        return True, metrics, 'passed'

    def _compute_zscore(self, series: np.ndarray, window: int) -> float:
        if len(series) < window: return 0.0
        mean = np.mean(series[-window:])
        std = np.std(series[-window:])
        if std == 0: return 0.0
        return (series[-1] - mean) / std

    def _load_trf_volume_corrections(self) -> Dict[str, float]:
        try:
            target_date = datetime.now() - timedelta(days=self.config.lookback_days)
            correction_factor = get_trf_correction_factor("", target_date)
            if correction_factor != 1.0:
                return {"*": correction_factor}
            return {}
        except Exception:
            return {}

    def _apply_trf_volume_correction(self, df: pd.DataFrame, correction_factor: float) -> pd.DataFrame:
        df = df.copy()
        today = datetime.now().date()
        historical_mask = df['timestamp'].dt.date < today
        if historical_mask.any():
            df.loc[historical_mask, 'volume'] = df.loc[historical_mask, 'volume'] * correction_factor
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
