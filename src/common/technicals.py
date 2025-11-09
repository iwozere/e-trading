from src.notification.logger import setup_logger
from src.model.telegram_bot import Technicals
import pandas as pd
import talib
import numpy as np

_logger = setup_logger(__name__)


def calculate_technicals_talib(df: pd.DataFrame) -> Technicals:
    """
    Calculate technical indicators using TALib directly.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

    Returns:
        Technicals object with calculated indicators and recommendations
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract price data
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)

    # Check for sufficient data
    if len(close) < 50:
        _logger.warning("Insufficient data for indicator calculations: %d points (need at least 50)", len(close))

    # Calculate all indicators using TALib
    try:
        # RSI
        rsi = talib.RSI(close, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

        # Moving Averages
        sma_fast = talib.SMA(close, timeperiod=50)
        sma_slow = talib.SMA(close, timeperiod=200)
        ema_fast = talib.EMA(close, timeperiod=12)
        ema_slow = talib.EMA(close, timeperiod=26)

        # ADX and DI
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)

        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

        # Other indicators
        atr = talib.ATR(high, low, close, timeperiod=14)
        williams_r = talib.WILLR(high, low, close, timeperiod=14)
        cci = talib.CCI(high, low, close, timeperiod=14)
        roc = talib.ROC(close, timeperiod=10)
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        obv = talib.OBV(close, volume)

        # ADR (Average Daily Range)
        daily_range = high - low
        adr = talib.SMA(daily_range, timeperiod=14)

    except Exception as e:
        _logger.error("Error calculating indicators with TALib: %s", e)
        raise

    # Get the last (current) values from the indicator arrays
    # Handle NaN values by converting to None
    def get_last_value(arr):
        """Get last non-NaN value or None"""
        if arr is None or len(arr) == 0:
            return None
        last_val = arr[-1]
        if np.isnan(last_val):
            return None
        return float(last_val)

    # Build technical data dictionary with current indicator values
    technical_data = {
        'rsi': get_last_value(rsi),
        'sma_fast': get_last_value(sma_fast),
        'sma_slow': get_last_value(sma_slow),
        'ema_fast': get_last_value(ema_fast),
        'ema_slow': get_last_value(ema_slow),
        'macd': get_last_value(macd),
        'macd_signal': get_last_value(macd_signal),
        'macd_histogram': get_last_value(macd_hist),
        'stoch_k': get_last_value(stoch_k),
        'stoch_d': get_last_value(stoch_d),
        'adx': get_last_value(adx),
        'plus_di': get_last_value(plus_di),
        'minus_di': get_last_value(minus_di),
        'obv': get_last_value(obv),
        'adr': get_last_value(adr),
        'avg_adr': None,  # Not calculated
        'trend': 'NEUTRAL',
        'bb_upper': get_last_value(bb_upper),
        'bb_middle': get_last_value(bb_middle),
        'bb_lower': get_last_value(bb_lower),
        'bb_width': None,  # Will calculate below
        'cci': get_last_value(cci),
        'roc': get_last_value(roc),
        'mfi': get_last_value(mfi),
        'williams_r': get_last_value(williams_r),
        'atr': get_last_value(atr),
        'recommendations': {}
    }

    # Generate recommendations based on indicator values
    recommendations = {}
    current_price = close[-1]

    # RSI recommendations
    if technical_data['rsi'] is not None:
        rsi_val = technical_data['rsi']
        if rsi_val > 70:
            recommendations['rsi'] = {"signal": "SELL", "reason": "Overbought - Sell opportunity", "confidence": 0.7}
        elif rsi_val < 30:
            recommendations['rsi'] = {"signal": "BUY", "reason": "Oversold - Buy opportunity", "confidence": 0.7}
        else:
            recommendations['rsi'] = {"signal": "HOLD", "reason": "Neutral zone - No clear signal", "confidence": 0.5}

    # MACD recommendations
    if technical_data['macd'] is not None and technical_data['macd_signal'] is not None:
        if technical_data['macd'] > technical_data['macd_signal']:
            recommendations['macd'] = {"signal": "BUY", "reason": "MACD above signal - Bullish", "confidence": 0.6}
        else:
            recommendations['macd'] = {"signal": "SELL", "reason": "MACD below signal - Bearish", "confidence": 0.6}

    # Bollinger Bands recommendations
    if technical_data['bb_upper'] is not None and technical_data['bb_lower'] is not None:
        if current_price >= technical_data['bb_upper']:
            recommendations['bb_middle'] = {"signal": "SELL", "reason": "Price at upper band - Overbought", "confidence": 0.6}
        elif current_price <= technical_data['bb_lower']:
            recommendations['bb_middle'] = {"signal": "BUY", "reason": "Price at lower band - Oversold", "confidence": 0.6}
        else:
            recommendations['bb_middle'] = {"signal": "HOLD", "reason": "Price within bands - Normal range", "confidence": 0.5}

    # Stochastic recommendations
    if technical_data['stoch_k'] is not None and technical_data['stoch_d'] is not None:
        stoch_k_val = technical_data['stoch_k']
        stoch_d_val = technical_data['stoch_d']
        if stoch_k_val > 80:
            recommendations['stoch_k'] = {"signal": "SELL", "reason": "Stochastic overbought (K > 80)", "confidence": 0.7}
        elif stoch_k_val < 20:
            recommendations['stoch_k'] = {"signal": "BUY", "reason": "Stochastic oversold (K < 20)", "confidence": 0.7}
        elif stoch_k_val > stoch_d_val:
            recommendations['stoch_k'] = {"signal": "BUY", "reason": "Stochastic K above D - Bullish crossover", "confidence": 0.6}
        else:
            recommendations['stoch_k'] = {"signal": "HOLD", "reason": "Stochastic in neutral zone", "confidence": 0.5}

    # ADX recommendations
    if technical_data['adx'] is not None:
        adx_val = technical_data['adx']
        if adx_val > 25:  # Strong trend
            if technical_data['plus_di'] is not None and technical_data['minus_di'] is not None:
                if technical_data['plus_di'] > technical_data['minus_di']:
                    recommendations['adx'] = {"signal": "BUY", "reason": f"Strong uptrend (ADX: {adx_val:.1f})", "confidence": 0.8}
                else:
                    recommendations['adx'] = {"signal": "SELL", "reason": f"Strong downtrend (ADX: {adx_val:.1f})", "confidence": 0.8}
            else:
                recommendations['adx'] = {"signal": "HOLD", "reason": f"Strong trend (ADX: {adx_val:.1f})", "confidence": 0.6}
        else:
            recommendations['adx'] = {"signal": "HOLD", "reason": f"Weak trend (ADX: {adx_val:.1f})", "confidence": 0.4}

    # OBV recommendations
    if technical_data['obv'] is not None:
        obv_val = technical_data['obv']
        if obv_val > 0:
            recommendations['obv'] = {"signal": "BUY", "reason": "Positive OBV - Accumulation", "confidence": 0.5}
        else:
            recommendations['obv'] = {"signal": "SELL", "reason": "Negative OBV - Distribution", "confidence": 0.5}

    # ADR recommendations
    if technical_data['adr'] is not None:
        adr_val = technical_data['adr']
        if adr_val > 3.0:
            recommendations['adr'] = {"signal": "HOLD", "reason": f"High volatility ({adr_val:.1f}%) - Caution", "confidence": 0.5}
        elif adr_val < 1.0:
            recommendations['adr'] = {"signal": "HOLD", "reason": f"Low volatility ({adr_val:.1f}%) - Stable price action", "confidence": 0.5}
        else:
            recommendations['adr'] = {"signal": "HOLD", "reason": f"Normal volatility ({adr_val:.1f}%)", "confidence": 0.5}

    # Generate overall recommendation based on individual signals
    buy_signals = sum(1 for rec in recommendations.values() if rec["signal"] in ["STRONG_BUY", "BUY"])
    sell_signals = sum(1 for rec in recommendations.values() if rec["signal"] in ["STRONG_SELL", "SELL"])
    total_signals = len(recommendations)

    if total_signals > 0:
        if buy_signals > sell_signals:
            overall_signal = "BUY"
            reason = f"More buy signals ({buy_signals} vs {sell_signals})"
            confidence = buy_signals / total_signals
        elif sell_signals > buy_signals:
            overall_signal = "SELL"
            reason = f"More sell signals ({sell_signals} vs {buy_signals})"
            confidence = sell_signals / total_signals
        else:
            overall_signal = "HOLD"
            reason = "Mixed signals"
            confidence = 0.5
    else:
        overall_signal = "HOLD"
        reason = "No clear signals available"
        confidence = 0.5

    recommendations["overall"] = {
        "signal": overall_signal,
        "reason": reason,
        "confidence": confidence
    }

    # Calculate trend based on SMA comparison
    if technical_data['sma_fast'] is not None and technical_data['sma_slow'] is not None:
        if technical_data['sma_fast'] > technical_data['sma_slow']:
            technical_data['trend'] = 'BULLISH'
        else:
            technical_data['trend'] = 'BEARISH'

    # Calculate BB width if BB values are available
    if (technical_data['bb_upper'] is not None and
        technical_data['bb_middle'] is not None and
        technical_data['bb_lower'] is not None and
        technical_data['bb_middle'] != 0):
        technical_data['bb_width'] = (technical_data['bb_upper'] - technical_data['bb_lower']) / technical_data['bb_middle']

    # Set recommendations
    technical_data['recommendations'] = recommendations

    return Technicals(**technical_data)






def format_technical_analysis(ticker: str, technicals: Technicals, current_price: float = None) -> str:
    if not technicals:
        return f"❌ Unable to analyze {ticker}"

    # Use the actual current price if provided, otherwise fall back to bb_middle
    if current_price is not None:
        close = current_price
    else:
        close = technicals.bb_middle  # Fallback to Bollinger Bands middle line

    # Handle None values for price
    if close is not None:
        price_str = f"${close:.2f}"
    else:
        price_str = "N/A"

    rsi = technicals.rsi
    trend = technicals.trend
    overall_rec = technicals.recommendations.get("overall", {}) if technicals.recommendations else {}
    message = f"📊 *Technical Analysis: {ticker}*\n\n"
    message += f"💰 Price: {price_str}\n"
    message += f"📈 Trend: {trend}\n"
    message += f"�� Overall: *{overall_rec.get('signal', 'HOLD')}*\n"
    message += f"💡 {overall_rec.get('reason', 'No reason available')}\n\n"
    message += "*Key Indicators:*\n"

    # RSI
    rsi_rec = technicals.recommendations.get("rsi", {}) if technicals.recommendations else {}
    if rsi is not None:
        message += f"• RSI ({rsi:.1f}): {rsi_rec.get('signal', 'HOLD')} - {rsi_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• RSI (N/A): {rsi_rec.get('signal', 'HOLD')} - {rsi_rec.get('reason', 'No reason provided')}\n"

    # MACD
    macd_rec = technicals.recommendations.get("macd", {}) if technicals.recommendations else {}
    macd_val = technicals.macd
    if macd_val is not None:
        message += f"• MACD ({macd_val:.4f}): {macd_rec.get('signal', 'HOLD')} - {macd_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• MACD (N/A): {macd_rec.get('signal', 'HOLD')} - {macd_rec.get('reason', 'No reason provided')}\n"

    # Bollinger Bands
    bb_rec = technicals.recommendations.get("bb_middle", {}) if technicals.recommendations else {}
    bb_lower = technicals.bb_lower
    bb_upper = technicals.bb_upper
    if bb_lower is not None and bb_upper is not None:
        message += f"• BB ({bb_lower:.2f} - {bb_upper:.2f}): {bb_rec.get('signal', 'HOLD')} - {bb_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• BB (N/A): {bb_rec.get('signal', 'HOLD')} - {bb_rec.get('reason', 'No reason provided')}\n"

    # Stochastic
    stoch_rec = technicals.recommendations.get("stoch_k", {}) if technicals.recommendations else {}
    stoch_k = technicals.stoch_k
    stoch_d = technicals.stoch_d
    if stoch_k is not None and stoch_d is not None:
        message += f"• Stoch K/D ({stoch_k:.1f}/{stoch_d:.1f}): {stoch_rec.get('signal', 'HOLD')} - {stoch_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• Stoch K/D (N/A): {stoch_rec.get('signal', 'HOLD')} - {stoch_rec.get('reason', 'No reason provided')}\n"

    # ADX
    adx_rec = technicals.recommendations.get("adx", {}) if technicals.recommendations else {}
    adx_val = technicals.adx
    if adx_val is not None:
        message += f"• ADX ({adx_val:.1f}): {adx_rec.get('signal', 'HOLD')} - {adx_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• ADX (N/A): {adx_rec.get('signal', 'HOLD')} - {adx_rec.get('reason', 'No reason provided')}\n"

    # OBV
    obv_rec = technicals.recommendations.get("obv", {}) if technicals.recommendations else {}
    obv_val = technicals.obv
    if obv_val is not None:
        message += f"• OBV ({obv_val:.0f}): {obv_rec.get('signal', 'HOLD')} - {obv_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• OBV (N/A): {obv_rec.get('signal', 'HOLD')} - {obv_rec.get('reason', 'No reason provided')}\n"

    # ADR
    adr_rec = technicals.recommendations.get("adr", {}) if technicals.recommendations else {}
    adr_val = technicals.adr
    if adr_val is not None:
        message += f"• ADR ({adr_val:.2f}): {adr_rec.get('signal', 'HOLD')} - {adr_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• ADR (N/A): {adr_rec.get('signal', 'HOLD')} - {adr_rec.get('reason', 'No reason provided')}\n"
    return message




