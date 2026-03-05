import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P08FeatureEngine:
    """
    MTF Feature Engineering for p08_mtf.
    Calculates Local (Execution) and Anchor (Trend) indicators.
    """

    @staticmethod
    def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Builds a multi-timeframe feature set.
        Expected columns: open, high, low, close, volume, btc_mc, vix
        Optional columns: anchor_open, anchor_high, anchor_low, anchor_close, anchor_volume
        """
        X = pd.DataFrame(index=df.index)

        # --- 1. Execution (Local) Features ---
        close = df['close']
        X['log_ret_1'] = np.log(close / close.shift(1))
        X['log_ret_5'] = np.log(close / close.shift(5))

        # Technical Indicators
        X['rsi'] = talib.RSI(close, timeperiod=config.get('rsi_period', 14))

        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=config.get('bb_period', 20),
            nbdevup=config.get('bb_std', 2.0),
            nbdevdn=config.get('bb_std', 2.0)
        )
        X['bb_pos'] = (close - lower) / (upper - lower + 1e-8)

        # ATR-based volatility
        # Note: ATR handles NaNs internally if talib is installed
        atr_period = config.get('atr_period', 14)
        atr = talib.ATR(df['high'], df['low'], close, timeperiod=atr_period)
        X['atr_rel'] = atr / (close + 1e-8)

        # Volume Z-score
        vol = df['volume']
        X['vol_z'] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-8)

        # --- 2. Anchor (Trend) Features ---
        if 'anchor_close' in df.columns:
            a_close = df['anchor_close']

            # Anchor Trend: EMA Slope proxy
            a_ema = talib.EMA(a_close, timeperiod=config.get('anchor_ema_period', 20))
            X['anchor_trend'] = np.log(a_ema / a_ema.shift(1))

            # Anchor RSI (Long-term relative strength)
            X['anchor_rsi'] = talib.RSI(a_close, timeperiod=config.get('anchor_rsi_period', 14))

            # Anchor BB Position (Is the big trend overextended?)
            a_upper, a_middle, a_lower = talib.BBANDS(
                a_close,
                timeperiod=config.get('anchor_bb_period', 20)
            )
            X['anchor_bb_pos'] = (a_close - a_lower) / (a_upper - a_lower + 1e-8)

            # Anchor Volume Momentum
            a_vol = df['anchor_volume']
            X['anchor_vol_z'] = (a_vol - a_vol.rolling(20).mean()) / (a_vol.rolling(20).std() + 1e-8)
        else:
            # Fallback if MTF join failed or anchor not provided
            X['anchor_trend'] = 0.0
            X['anchor_rsi'] = 50.0
            X['anchor_bb_pos'] = 0.5
            X['anchor_vol_z'] = 0.0

        # --- 3. Macro Features (VIX, BTC_MC) ---
        if 'vix' in df.columns:
            X['vix'] = df['vix'].fillna(method='ffill').fillna(0)
        else:
            X['vix'] = 0.0

        if 'btc_mc' in df.columns:
            mc_ret = np.log(df['btc_mc'] / df['btc_mc'].shift(1))
            X['btc_mc_log_ret'] = mc_ret.fillna(0)
        else:
            X['btc_mc_log_ret'] = 0.0

        # Clean NaNs (caused by indicator warmup)
        nan_counts = X.isna().sum()
        if nan_counts.any():
            _logger.debug("P08 NaNs before cleanup: %s", nan_counts[nan_counts > 0].to_dict())

        X.dropna(inplace=True)
        return X
