# File: scripts/feature_engineering.py

import numpy as np
import talib


def add_log_return(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df


def add_volatility(df, window):
    df['volatility'] = df['log_return'].rolling(window=window).std()
    return df


def add_rsi(df, period):
    df['rsi'] = talib.RSI(df['close'], timeperiod=period)
    return df


def add_macd(df, fast, slow, signal):
    macd, macdsignal, macdhist = talib.MACD(
        df['close'],
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )
    df['macd'] = macd
    return df


def add_bollinger_band_width(df, period):
    upper, middle, lower = talib.BBANDS(
        df['close'],
        timeperiod=period,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )
    df['boll_width'] = upper - lower
    return df


def generate_features(df, feature_list, params):
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
    return df[[col for col in ['log_return', 'volatility', 'rsi', 'macd', 'boll_width'] if col in df]], df
