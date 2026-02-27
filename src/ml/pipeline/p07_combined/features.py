import pandas as pd
import numpy as np
import talib
from typing import List, Dict

def build_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Builds a consistent feature set using TA-Lib.
    Priority: Stationarity and Parity with ONNX/C++ layer.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Price-based stationary features (Log Returns)
    close = df['close']
    X['log_ret_1'] = np.log(close / close.shift(1))
    X['log_ret_5'] = np.log(close / close.shift(5))

    # 2. Macro Features (already injected)
    # Fill NaNs with 0 to prevent dropping entire dataset if macro is missing
    if 'vix' in df.columns:
        X['vix'] = df['vix'].fillna(0)
    else:
        X['vix'] = 0.0

    if 'btc_mc' in df.columns:
        # Avoid log(0) if filling NaNs, though mc is usually large or NaN
        mc_ret = np.log(df['btc_mc'] / df['btc_mc'].shift(1))
        X['btc_mc_log_ret'] = mc_ret.fillna(0)
    else:
        X['btc_mc_log_ret'] = 0.0

    if 'global_regime' in df.columns:
        X['global_regime'] = df['global_regime'].fillna(0)
    else:
        X['global_regime'] = 0

    # 3. Technical Indicators (Stationary versions)
    # RSI (0-100 is stationary)
    X['rsi'] = talib.RSI(close, timeperiod=config.get('rsi_period', 14))

    # Bollinger Bands (distance from bands)
    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=config.get('bb_period', 20),
        nbdevup=config.get('bb_std', 2.0),
        nbdevdn=config.get('bb_std', 2.0)
    )
    X['bb_pos'] = (close - lower) / (upper - lower + 1e-8)

    # ATR (relative to price)
    atr = talib.ATR(df['high'], df['low'], close, timeperiod=config.get('atr_period', 14))
    X['atr_rel'] = atr / close

    # 4. Volume features
    X['vol_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-8)

    nan_counts = X.isna().sum()
    if nan_counts.any():
        from src.notification.logger import setup_logger
        _logger = setup_logger(__name__)
        _logger.debug("NaNs before dropna: %s", nan_counts[nan_counts > 0].to_dict())

    X.dropna(inplace=True)
    if len(X) == 0:
        _logger.error("All rows dropped in build_features! Features: %s", X.columns.tolist())
    return X
