import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P07FeatureEngine:
    """
    Shared Feature Engineering logic for p07_combined.
    Ensures parity between Optimization (Batch) and Backtrader/Live (Point-in-time).
    Supports optional MTF anchor features via enable_mtf flag.
    """

    @staticmethod
    def build_features(df: pd.DataFrame, config: Dict[str, Any], enable_mtf: bool = False) -> pd.DataFrame:
        """
        Builds a consistent feature set from OHLCV + Macro data.

        Args:
            df: DataFrame with OHLCV + optional macro and anchor columns.
            config: Parameter dict for indicator periods.
            enable_mtf: When True, adds anchor timeframe features (from merged P08 logic).

        Returns:
            Feature DataFrame with NaN rows dropped.
        """
        X = pd.DataFrame(index=df.index)

        # 1. Price-based stationary features (Log Returns)
        close = df['close']
        X['log_ret_1'] = np.log(close / close.shift(1))
        X['log_ret_5'] = np.log(close / close.shift(5))

        # 2. Macro Features (already injected by DataLoader)
        # Fill NaNs with 0 to prevent dropping entire dataset if macro is missing
        if 'vix' in df.columns:
            X['vix'] = df['vix'].fillna(0)
        else:
            X['vix'] = 0.0

        if 'btc_mc' in df.columns:
            mc_ret = np.log(df['btc_mc'] / df['btc_mc'].shift(1))
            X['btc_mc_log_ret'] = mc_ret.fillna(0)
        else:
            X['btc_mc_log_ret'] = 0.0

        if 'global_regime' in df.columns:
            X['global_regime'] = df['global_regime'].fillna(0)
        else:
            X['global_regime'] = 0

        # 3. Technical Indicators (Stationary versions)
        X['rsi'] = talib.RSI(close, timeperiod=config.get('rsi_period', 14))

        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=config.get('bb_period', 20),
            nbdevup=config.get('bb_std', 2.0),
            nbdevdn=config.get('bb_std', 2.0)
        )
        X['bb_pos'] = (close - lower) / (upper - lower + 1e-8)

        atr = talib.ATR(df['high'], df['low'], close, timeperiod=config.get('atr_period', 14))
        X['atr_rel'] = atr / close

        # 4. Volume features
        vol = df['volume']
        vol_lookback = config.get('vol_lookback', 20)
        X['vol_z'] = (vol - vol.rolling(vol_lookback).mean()) / (vol.rolling(vol_lookback).std() + 1e-8)

        # 5. Anchor (MTF) Features — merged from P08FeatureEngine
        if enable_mtf:
            P07FeatureEngine._add_anchor_features(X, df, config)

        # Debug NaNs
        nan_counts = X.isna().sum()
        if nan_counts.any():
            _logger.debug("NaNs before dropna: %s", nan_counts[nan_counts > 0].to_dict())

        X.dropna(inplace=True)
        return X

    @staticmethod
    def _add_anchor_features(X: pd.DataFrame, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Adds anchor timeframe features to X in-place.
        Requires anchor_close / anchor_volume columns in df (produced by DataLoader.merge_mtf).
        Falls back to neutral placeholder values if anchor columns are absent.
        """
        if 'anchor_close' in df.columns:
            a_close = df['anchor_close']
            a_high = df.get('anchor_high', df['high'])
            a_low = df.get('anchor_low', df['low'])

            a_ema_period = config.get('anchor_ema_period', 20)
            a_ema = talib.EMA(a_close, timeperiod=a_ema_period)
            X['anchor_trend'] = np.log(a_ema / a_ema.shift(1))

            regime_thresh = config.get('regime_threshold', 0.0001)
            X['anchor_regime'] = 0
            X.loc[X['anchor_trend'] > regime_thresh, 'anchor_regime'] = 1
            X.loc[X['anchor_trend'] < -regime_thresh, 'anchor_regime'] = -1

            close = df['close']
            X['mtf_divergence'] = (close - a_ema) / (a_ema + 1e-8)

            X['anchor_rsi'] = talib.RSI(a_close, timeperiod=config.get('anchor_rsi_period', 14))

            a_atr = talib.ATR(a_high, a_low, a_close, timeperiod=config.get('anchor_atr_period', 14))
            X['anchor_atr_rel'] = a_atr / (a_close + 1e-8)

            a_upper, a_middle, a_lower = talib.BBANDS(
                a_close,
                timeperiod=config.get('anchor_bb_period', 20)
            )
            X['anchor_bb_pos'] = (a_close - a_lower) / (a_upper - a_lower + 1e-8)

            a_vol = df['anchor_volume']
            X['anchor_vol_z'] = (a_vol - a_vol.rolling(20).mean()) / (a_vol.rolling(20).std() + 1e-8)
        else:
            _logger.debug("Anchor columns not found; using neutral MTF placeholders.")
            X['anchor_trend'] = 0.0
            X['anchor_regime'] = 0
            X['mtf_divergence'] = 0.0
            X['anchor_rsi'] = 50.0
            X['anchor_atr_rel'] = 0.01
            X['anchor_bb_pos'] = 0.5
            X['anchor_vol_z'] = 0.0


# Backward compatibility for existing imports
def build_features(df: pd.DataFrame, config: Dict[str, Any], enable_mtf: bool = False) -> pd.DataFrame:
    return P07FeatureEngine.build_features(df, config, enable_mtf=enable_mtf)
