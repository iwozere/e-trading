"""
Financial Feature Engineering Module.

This script provides a collection of functions to calculate common financial
technical indicators and add them as features to a pandas DataFrame. It is
designed to be modular and easily configurable.

The primary entry point is the `generate_features` function, which orchestrates
the calculation of a specified list of features using a parameter dictionary.

Core Libraries:
- pandas: For data manipulation.
- numpy: For numerical operations, specifically log returns.
- TA-Lib: For calculating standard technical indicators like RSI, MACD, etc.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # Go up 3 levels from 'src/ml/hmm'
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import talib

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the logarithmic return of the 'close' price.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'close' price column.

    Returns:
        pd.DataFrame: The DataFrame with a 'log_return' column added.
    """
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df


def add_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculates the rolling standard deviation of log returns (volatility).

    This function requires `add_log_return` to be called first.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'log_return' column.
        window (int): The rolling window period for the standard deviation calculation.

    Returns:
        pd.DataFrame: The DataFrame with a 'volatility' column added.
    """
    df['volatility'] = df['log_return'].rolling(window=window).std()
    return df


def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculates the Relative Strength Index (RSI) from the 'close' price.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'close' price column.
        period (int): The time period for the RSI calculation (e.g., 14).

    Returns:
        pd.DataFrame: The DataFrame with an 'rsi' column added.
    """
    df['rsi'] = talib.RSI(df['close'], timeperiod=period)
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Calculates the Moving Average Convergence Divergence (MACD) line.

    Note: This function only adds the main MACD line, not the signal line or histogram.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'close' price column.
        fast (int): The time period for the fast-moving average (e.g., 12).
        slow (int): The time period for the slow-moving average (e.g., 26).
        signal (int): The time period for the signal line EMA (e.g., 9).

    Returns:
        pd.DataFrame: The DataFrame with a 'macd' column added.
    """
    macd, macdsignal, macdhist = talib.MACD(
        df['close'],
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )
    df['macd'] = macd
    return df


def add_bollinger_band_width(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculates the width of the Bollinger Bands (Upper Band - Lower Band).

    A smaller width indicates lower volatility, while a larger width indicates
    higher volatility.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'close' price column.
        period (int): The time period for the Bollinger Bands calculation (e.g., 20).

    Returns:
        pd.DataFrame: The DataFrame with a 'boll_width' column added.
    """
    upper, middle, lower = talib.BBANDS(
        df['close'],
        timeperiod=period,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )
    df['boll_width'] = upper - lower
    return df


def generate_features(df: pd.DataFrame, feature_list: list[str], params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a set of technical indicator features for a given DataFrame.

    This function acts as a high-level wrapper that calls individual feature
    generation functions based on the provided `feature_list`. It uses
    default parameters for indicators unless they are overridden in the `params` dict.
    After generating all features, it removes any rows with NaN values that
    resulted from the calculations (e.g., from rolling windows).

    Args:
        df (pd.DataFrame): The input DataFrame, must contain 'close' prices.
        feature_list (list[str]): A list of strings specifying which features
            to generate. Supported values: 'volatility', 'rsi', 'macd', 'boll'.
        params (dict): A dictionary of parameters for the feature calculations.
            If a parameter is not provided, a default value is used.
            Example keys: 'vol_window', 'rsi_period', 'macd_fast', etc.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - features_df: A DataFrame with only the generated feature columns.
            - full_df: The original DataFrame with all generated features added
              and NaN rows dropped.
    """
    # Log return is always calculated as it's a base for other features.
    df = add_log_return(df)

    if 'volatility' in feature_list:
        df = add_volatility(df, window=params.get('vol_window', 20))
    if 'rsi' in feature_list:
        df = add_rsi(df, period=params.get('rsi_period', 14))
    if 'macd' in feature_list:
        df = add_macd(df,
                      fast=params.get('macd_fast', 12),
                      slow=params.get('macd_slow', 26),
                      signal=params.get('macd_signal', 9))
    if 'boll' in feature_list:
        df = add_bollinger_band_width(df, period=params.get('boll_window', 20))

    df.dropna(inplace=True)

    # Create a list of columns that were actually generated before returning
    generated_cols = [col for col in ['log_return', 'volatility', 'rsi', 'macd', 'boll_width'] if col in df.columns]

    return df[generated_cols], df
