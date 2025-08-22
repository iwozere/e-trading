import numpy as np
import talib
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from src.notification.logger import setup_logger
from src.model.telegram_bot import Technicals
from src.common import get_ohlcv

logger = setup_logger(__name__)


def get_rsi_recommendation(rsi: float) -> Tuple[str, str]:
    if rsi < 30:
        return "BUY", "Oversold - Strong buy signal"
    elif rsi < 40:
        return "BUY", "Approaching oversold - Buy opportunity"
    elif rsi > 70:
        return "SELL", "Overbought - Strong sell signal"
    elif rsi > 60:
        return "SELL", "Approaching overbought - Sell opportunity"
    else:
        return "HOLD", "Neutral zone - No clear signal"

def get_bollinger_recommendation(close: float, bb_upper: float, bb_middle: float, bb_lower: float) -> Tuple[str, str]:
    if close <= bb_lower:
        return "BUY", "Price at or below lower band - Oversold"
    elif close >= bb_upper:
        return "SELL", "Price at or above upper band - Overbought"
    elif close < bb_middle:
        return "BUY", "Price below middle band - Potential buy"
    else:
        return "HOLD", "Price above middle band - Neutral"

def get_macd_recommendation(macd: float, signal: float, macd_hist: float) -> Tuple[str, str]:
    if macd > signal and macd_hist > 0:
        if macd_hist > 0.5:
            return "BUY", "Strong bullish MACD crossover"
        else:
            return "BUY", "Bullish MACD crossover"
    elif macd < signal and macd_hist < 0:
        if macd_hist < -0.5:
            return "SELL", "Strong bearish MACD crossover"
        else:
            return "SELL", "Bearish MACD crossover"
    else:
        return "HOLD", "MACD signals neutral"

def get_stochastic_recommendation(k: float, d: float) -> Tuple[str, str]:
    if k < 20 and d < 20:
        return "BUY", "Both K and D in oversold territory"
    elif k > 80 and d > 80:
        return "SELL", "Both K and D in overbought territory"
    elif k < 30 and k > d:
        return "BUY", "K crossing above D in oversold area"
    elif k > 70 and k < d:
        return "SELL", "K crossing below D in overbought area"
    else:
        return "HOLD", "Stochastic in neutral zone"

def get_adx_recommendation(adx: float, plus_di: float, minus_di: float) -> Tuple[str, str]:
    if adx > 25:
        if plus_di > minus_di:
            return "BUY", f"Strong uptrend (ADX: {adx:.1f})"
        else:
            return "SELL", f"Strong downtrend (ADX: {adx:.1f})"
    else:
        return "HOLD", "Weak trend - Sideways market"

def get_obv_recommendation(obv_current: float, obv_prev: float, price_change: float) -> Tuple[str, str]:
    if obv_current > obv_prev and price_change > 0:
        return "BUY", "OBV confirming price increase - Bullish"
    elif obv_current < obv_prev and price_change < 0:
        return "SELL", "OBV confirming price decrease - Bearish"
    elif obv_current > obv_prev and price_change < 0:
        return "BUY", "OBV divergence - Price may reverse up"
    elif obv_current < obv_prev and price_change > 0:
        return "SELL", "OBV divergence - Price may reverse down"
    else:
        return "HOLD", "OBV neutral"

def get_adr_recommendation(adr: float, avg_adr: float) -> Tuple[str, str]:
    if adr > avg_adr * 1.5:
        return "HOLD", "High volatility - Wait for stabilization"
    elif adr < avg_adr * 0.5:
        return "HOLD", "Low volatility - Wait for breakout"
    else:
        return "HOLD", "Normal volatility"

def get_overall_recommendation(recommendations: List[Tuple[str, str]]) -> Tuple[str, str]:
    buy_count = sum(1 for rec, _ in recommendations if rec == "BUY")
    sell_count = sum(1 for rec, _ in recommendations if rec == "SELL")
    hold_count = sum(1 for rec, _ in recommendations if rec == "HOLD")
    total = len(recommendations)
    if buy_count > sell_count and buy_count >= total * 0.4:
        return "BUY", f"Strong buy signal ({buy_count}/{total} indicators)"
    elif sell_count > buy_count and sell_count >= total * 0.4:
        return "SELL", f"Strong sell signal ({sell_count}/{total} indicators)"
    else:
        return "HOLD", f"Mixed signals - Hold position ({hold_count}/{total} neutral)"

def calculate_technicals_from_df(df, indicators: List[str] = None, indicator_params: Dict[str, Dict[str, Any]] = None):
    """
    Calculate technical indicators from a DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and return a tuple (updated DataFrame, Technicals object).
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        indicators: Optional list of indicator names to calculate (e.g. ['rsi', 'macd'])
        indicator_params: Optional dict of indicator names to TA-Lib parameter dicts
    Returns:
        (DataFrame, Technicals): DataFrame with indicator columns, Technicals dataclass with latest values
    """
    if df is None or df.empty:
        logger.error("No data provided for technicals calculation.")
        return None, None
    df = df.dropna()
    if len(df) < 50:
        logger.error("Insufficient data for technicals: %d rows", len(df))
        return None, None
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    volume = df['volume'].values.astype(float)
    open_price = df['open'].values.astype(float)
    min_length = min(len(high), len(low), len(close), len(volume), len(open_price))
    if min_length < 50:
        logger.error("Insufficient data after cleaning: %d rows", min_length)
        return None, None
    high = high[-min_length:]
    low = low[-min_length:]
    close = close[-min_length:]
    volume = volume[-min_length:]
    open_price = open_price[-min_length:]

    df['log_return'] = np.log(df['close']).diff()

    all_indicators = [
        'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d', 'adx', 'plus_di', 'minus_di', 'obv', 'adr', 'sma_50', 'sma_200'
    ]
    if indicators is None:
        indicators = all_indicators
    indicators = set(indicators)

    # Default TA-Lib params
    default_params = {
        'rsi': {'timeperiod': 14},
        'bb_upper': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0},
        'bb_middle': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0},
        'bb_lower': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0},
        'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        'macd_signal': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        'macd_hist': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        'stoch_k': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
        'stoch_d': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
        'adx': {'timeperiod': 14},
        'plus_di': {'timeperiod': 14},
        'minus_di': {'timeperiod': 14},
        'obv': {},
        'adr': {'timeperiod': 14},
        'sma_50': {'timeperiod': 50},
        'sma_200': {'timeperiod': 200},
    }
    # Merge user params
    params = {k: v.copy() for k, v in default_params.items()}
    if indicator_params:
        for k, v in indicator_params.items():
            if k in params:
                params[k].update(v)

    results = {}
    try:
        if 'rsi' in indicators:
            results['rsi'] = talib.RSI(close, **params['rsi'])
        if {'bb_upper', 'bb_middle', 'bb_lower'} & indicators:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, **params['bb_upper'])
            if 'bb_upper' in indicators:
                results['bb_upper'] = bb_upper
            if 'bb_middle' in indicators:
                results['bb_middle'] = bb_middle
            if 'bb_lower' in indicators:
                results['bb_lower'] = bb_lower
        if {'macd', 'macd_signal', 'macd_hist'} & indicators:
            macd, macd_signal, macd_hist = talib.MACD(close, **params['macd'])
            if 'macd' in indicators:
                results['macd'] = macd
            if 'macd_signal' in indicators:
                results['macd_signal'] = macd_signal
            if 'macd_hist' in indicators:
                results['macd_hist'] = macd_hist
        if {'stoch_k', 'stoch_d'} & indicators:
            stoch_k, stoch_d = talib.STOCH(high, low, close, **params['stoch_k'])
            if 'stoch_k' in indicators:
                results['stoch_k'] = stoch_k
            if 'stoch_d' in indicators:
                results['stoch_d'] = stoch_d
        if 'adx' in indicators:
            results['adx'] = talib.ADX(high, low, close, **params['adx'])
        if 'plus_di' in indicators:
            results['plus_di'] = talib.PLUS_DI(high, low, close, **params['plus_di'])
        if 'minus_di' in indicators:
            results['minus_di'] = talib.MINUS_DI(high, low, close, **params['minus_di'])
        if 'obv' in indicators:
            results['obv'] = talib.OBV(close, volume)
        if 'adr' in indicators:
            daily_range = high - low
            results['adr'] = talib.SMA(daily_range, **params['adr'])
        if 'sma_50' in indicators:
            results['sma_50'] = talib.SMA(close, **params['sma_50'])
        if 'sma_200' in indicators:
            results['sma_200'] = talib.SMA(close, **params['sma_200'])
    except Exception as e:
        logger.exception("TA-Lib calculation failed for data: ")
        return None, None

    df = df.copy()
    for key, arr in results.items():
        df[key] = arr

    current_idx = -1
    def get_val(name, default):
        arr = results.get(name)
        if arr is not None:
            v = arr[current_idx]
            return v if not np.isnan(v) else default
        return default

    recommendations = {}
    if 'rsi' in indicators:
        rsi_signal, rsi_reason = get_rsi_recommendation(get_val('rsi', 50.0))
        recommendations['rsi'] = {'signal': rsi_signal, 'reason': rsi_reason}
    if {'macd', 'macd_signal', 'macd_hist'} & indicators:
        macd_signal, macd_reason = get_macd_recommendation(
            get_val('macd', 0.0), get_val('macd_signal', 0.0), get_val('macd_hist', 0.0))
        recommendations['macd'] = {'signal': macd_signal, 'reason': macd_reason}
    if {'bb_upper', 'bb_middle', 'bb_lower'} & indicators:
        bb_signal, bb_reason = get_bollinger_recommendation(
            close[current_idx],
            get_val('bb_upper', close[current_idx]),
            get_val('bb_middle', close[current_idx]),
            get_val('bb_lower', close[current_idx]))
        recommendations['bollinger'] = {'signal': bb_signal, 'reason': bb_reason}
    if {'stoch_k', 'stoch_d'} & indicators:
        stoch_signal, stoch_reason = get_stochastic_recommendation(
            get_val('stoch_k', 50.0), get_val('stoch_d', 50.0))
        recommendations['stochastic'] = {'signal': stoch_signal, 'reason': stoch_reason}
    if 'adx' in indicators:
        adx_signal, adx_reason = get_adx_recommendation(
            get_val('adx', 25.0), get_val('plus_di', 25.0), get_val('minus_di', 25.0))
        recommendations['adx'] = {'signal': adx_signal, 'reason': adx_reason}
    if 'obv' in indicators:
        prev_idx = -2 if len(close) > 1 else -1
        prev_obv = results['obv'][prev_idx] if 'obv' in results and not np.isnan(results['obv'][prev_idx]) else get_val('obv', 0.0)
        price_change = close[current_idx] - close[prev_idx]
        obv_signal, obv_reason = get_obv_recommendation(get_val('obv', 0.0), prev_obv, price_change)
        recommendations['obv'] = {'signal': obv_signal, 'reason': obv_reason}
    if 'adr' in indicators:
        adr_signal, adr_reason = get_adr_recommendation(get_val('adr', 0.0), get_val('adr', 0.0))
        recommendations['adr'] = {'signal': adr_signal, 'reason': adr_reason}

    technicals_kwargs = {}
    for name in all_indicators:
        if name in indicators:
            # Map field names to match Technicals dataclass
            if name == 'macd_hist':
                technicals_kwargs['macd_histogram'] = get_val(name, 0.0)
            else:
                technicals_kwargs[name] = get_val(name, 0.0 if 'sma' not in name else close[current_idx])
        else:
            # Provide default values for required fields that weren't calculated
            if name == 'rsi':
                technicals_kwargs[name] = 50.0
            elif name in ['sma_50', 'sma_200']:
                technicals_kwargs[name] = close[current_idx]
            elif name in ['macd', 'macd_signal']:
                technicals_kwargs[name] = 0.0
            elif name == 'macd_hist':
                technicals_kwargs['macd_histogram'] = 0.0
            elif name in ['stoch_k', 'stoch_d']:
                technicals_kwargs[name] = 50.0
            elif name in ['adx', 'plus_di', 'minus_di']:
                technicals_kwargs[name] = 25.0
            elif name in ['obv', 'adr']:
                technicals_kwargs[name] = 0.0
            elif name in ['bb_upper', 'bb_middle', 'bb_lower']:
                technicals_kwargs[name] = close[current_idx]
            else:
                technicals_kwargs[name] = 0.0

    # Add required fields that are not in all_indicators
    technicals_kwargs['avg_adr'] = technicals_kwargs.get('adr', 0.0)
    technicals_kwargs['trend'] = 'NEUTRAL'
    technicals_kwargs['bb_width'] = 0.0
    technicals_kwargs['recommendations'] = recommendations

    technicals = Technicals(**technicals_kwargs)
    return df, technicals

def format_technical_analysis(ticker: str, technicals: Technicals, current_price: float = None) -> str:
    if not technicals:
        return f"❌ Unable to analyze {ticker}"

    # Use the actual current price if provided, otherwise fall back to bb_middle
    if current_price is not None:
        close = current_price
    else:
        close = technicals.bb_middle  # Fallback to Bollinger Bands middle line
    rsi = technicals.rsi
    trend = technicals.trend
    overall_rec = technicals.recommendations.get("overall", {}) if technicals.recommendations else {}
    message = f"📊 *Technical Analysis: {ticker}*\n\n"
    message += f"💰 Price: ${close:.2f}\n"
    message += f"📈 Trend: {trend}\n"
    message += f"🎯 Overall: *{overall_rec.get('signal', 'HOLD')}*\n"
    message += f"💡 {overall_rec.get('reason', 'No reason available')}\n\n"
    message += "*Key Indicators:*\n"
    rsi_rec = technicals.recommendations.get("rsi", {}) if technicals.recommendations else {}
    message += f"• RSI ({rsi:.1f}): {rsi_rec.get('signal', 'HOLD')} - {rsi_rec.get('reason', 'No reason provided')}\n"
    macd_rec = technicals.recommendations.get("macd", {}) if technicals.recommendations else {}
    macd_val = technicals.macd
    message += f"• MACD ({macd_val:.4f}): {macd_rec.get('signal', 'HOLD')} - {macd_rec.get('reason', 'No reason provided')}\n"
    bb_rec = technicals.recommendations.get("bollinger", {}) if technicals.recommendations else {}
    bb_lower = technicals.bb_lower
    bb_upper = technicals.bb_upper
    message += f"• BB ({bb_lower:.2f} - {bb_upper:.2f}): {bb_rec.get('signal', 'HOLD')} - {bb_rec.get('reason', 'No reason provided')}\n"
    stoch_rec = technicals.recommendations.get("stochastic", {}) if technicals.recommendations else {}
    stoch_k = technicals.stoch_k
    stoch_d = technicals.stoch_d
    message += f"• Stoch K/D ({stoch_k:.1f}/{stoch_d:.1f}): {stoch_rec.get('signal', 'HOLD')} - {stoch_rec.get('reason', 'No reason provided')}\n"
    adx_rec = technicals.recommendations.get("adx", {}) if technicals.recommendations else {}
    adx_val = technicals.adx
    message += f"• ADX ({adx_val:.1f}): {adx_rec.get('signal', 'HOLD')} - {adx_rec.get('reason', 'No reason provided')}\n"
    obv_rec = technicals.recommendations.get("obv", {}) if technicals.recommendations else {}
    obv_val = technicals.obv
    message += f"• OBV: {obv_rec.get('signal', 'HOLD')} - {obv_rec.get('reason', 'No reason provided')}\n"
    adr_rec = technicals.recommendations.get("adr", {}) if technicals.recommendations else {}
    adr_val = technicals.adr
    message += f"• ADR ({adr_val:.2f}): {adr_rec.get('signal', 'HOLD')} - {adr_rec.get('reason', 'No reason provided')}\n"
    return message

def get_technicals(
    ticker: str,
    interval: str = "1d",
    period: str = "2y",
    provider: str = None,
    indicators: Optional[List[str]] = None,
    indicator_params: Optional[Dict[str, Dict[str, Any]]] = None,
    return_df: bool = False
) -> Any:
    """
    Fetch OHLCV data for the given ticker/interval/period/provider, calculate technical indicators, and return the Technicals object (and optionally the DataFrame).

    Args:
        ticker: Ticker symbol (e.g., 'AAPL')
        interval: Data interval (e.g., '1d', '1h')
        period: Period string (e.g., '2y', '6mo')
        provider: Data provider code (e.g., 'yf', 'bnc')
        indicators: Optional list of indicator names to calculate
        indicator_params: Optional dict of indicator names to TA-Lib parameter dicts
        return_df: If True, also return the DataFrame with technical columns

    Returns:
        Technicals object, or (DataFrame, Technicals) if return_df is True
    """
    df = get_ohlcv(ticker, interval, period, provider)
    df_with_technicals, technicals = calculate_technicals_from_df(df, indicators, indicator_params)
    if return_df:
        return df_with_technicals, technicals
    return technicals

if __name__ == "__main__":
    test_ticker = "AAPL"
    print(f"Testing technical analysis for {test_ticker}...")
    df, technicals = calculate_technicals_from_df(pd.DataFrame({
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [100],
        'volume': [1000]
    }))
    if df is not None and technicals:
        print(format_technical_analysis(test_ticker, technicals))
    else:
        print(f"Failed to analyze {test_ticker}")
