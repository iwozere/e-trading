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
    """

    @staticmethod
    def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Builds a consistent feature set from OHLCV + Macro data.
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
        X['vol_z'] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-8)

        # Debug NaNs
        nan_counts = X.isna().sum()
        if nan_counts.any():
            _logger.debug("NaNs before dropna: %s", nan_counts[nan_counts > 0].to_dict())

        X.dropna(inplace=True)
        return X

# Backward compatibility for existing imports
def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    return P07FeatureEngine.build_features(df, config)
