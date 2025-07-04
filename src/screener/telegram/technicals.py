import pandas as pd
from typing import Tuple, List
from src.notification.logger import setup_logger
from src.screener.telegram.models import Technicals

logger = setup_logger("telegram_bot")


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

def calculate_technicals_from_df(df):
    """
    Calculate technical indicators from a DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and return a tuple (updated DataFrame, Technicals object).
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    Returns:
        (DataFrame, Technicals): DataFrame with indicator columns, Technicals dataclass with latest values
    """
    import numpy as np
    import talib
    from src.screener.telegram.models import Technicals

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
    try:
        rsi = talib.RSI(close, timeperiod=14)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        obv = talib.OBV(close, volume)
        daily_range = high - low
        adr = talib.SMA(daily_range, timeperiod=14)
        sma_50 = talib.SMA(close, timeperiod=50)
        sma_200 = talib.SMA(close, timeperiod=200)
    except Exception as e:
        logger.error("TA-Lib calculation failed for data: %s", e, exc_info=True)
        return None, None

    # Add indicator arrays as columns to df
    df = df.copy()
    df['rsi'] = rsi
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['obv'] = obv
    df['adr'] = adr
    df['sma_50'] = sma_50
    df['sma_200'] = sma_200

    current_idx = -1
    current_close = close[current_idx]
    current_rsi = rsi[current_idx] if not np.isnan(rsi[current_idx]) else 50.0
    current_bb_upper = bb_upper[current_idx] if not np.isnan(bb_upper[current_idx]) else current_close
    current_bb_middle = bb_middle[current_idx] if not np.isnan(bb_middle[current_idx]) else current_close
    current_bb_lower = bb_lower[current_idx] if not np.isnan(bb_lower[current_idx]) else current_close
    current_macd = macd[current_idx] if not np.isnan(macd[current_idx]) else 0.0
    current_macd_signal = macd_signal[current_idx] if not np.isnan(macd_signal[current_idx]) else 0.0
    current_macd_hist = macd_hist[current_idx] if not np.isnan(macd_hist[current_idx]) else 0.0
    current_stoch_k = stoch_k[current_idx] if not np.isnan(stoch_k[current_idx]) else 50.0
    current_stoch_d = stoch_d[current_idx] if not np.isnan(stoch_d[current_idx]) else 50.0
    current_adx = adx[current_idx] if not np.isnan(adx[current_idx]) else 25.0
    current_plus_di = plus_di[current_idx] if not np.isnan(plus_di[current_idx]) else 25.0
    current_minus_di = minus_di[current_idx] if not np.isnan(minus_di[current_idx]) else 25.0
    current_obv = obv[current_idx] if not np.isnan(obv[current_idx]) else 0.0
    current_adr = adr[current_idx] if not np.isnan(adr[current_idx]) else 0.0
    current_sma_50 = sma_50[current_idx] if not np.isnan(sma_50[current_idx]) else current_close
    current_sma_200 = sma_200[current_idx] if not np.isnan(sma_200[current_idx]) else current_close
    prev_idx = -2 if len(close) > 1 else -1
    prev_close = close[prev_idx]
    prev_obv = obv[prev_idx] if not np.isnan(obv[prev_idx]) else current_obv
    price_change = current_close - prev_close
    # --- Recommendations ---
    recommendations = {}
    rsi_signal, rsi_reason = get_rsi_recommendation(current_rsi)
    recommendations['rsi'] = {'signal': rsi_signal, 'reason': rsi_reason}
    macd_signal, macd_reason = get_macd_recommendation(current_macd, current_macd_signal, current_macd_hist)
    recommendations['macd'] = {'signal': macd_signal, 'reason': macd_reason}
    bb_signal, bb_reason = get_bollinger_recommendation(current_close, current_bb_upper, current_bb_middle, current_bb_lower)
    recommendations['bollinger'] = {'signal': bb_signal, 'reason': bb_reason}
    stoch_signal, stoch_reason = get_stochastic_recommendation(current_stoch_k, current_stoch_d)
    recommendations['stochastic'] = {'signal': stoch_signal, 'reason': stoch_reason}
    adx_signal, adx_reason = get_adx_recommendation(current_adx, current_plus_di, current_minus_di)
    recommendations['adx'] = {'signal': adx_signal, 'reason': adx_reason}
    obv_signal, obv_reason = get_obv_recommendation(current_obv, prev_obv, price_change)
    recommendations['obv'] = {'signal': obv_signal, 'reason': obv_reason}
    avg_adr = np.mean(adr[-20:]) if len(adr) >= 20 else current_adr
    adr_signal, adr_reason = get_adr_recommendation(current_adr, avg_adr)
    recommendations['adr'] = {'signal': adr_signal, 'reason': adr_reason}
    # Overall
    overall_signal, overall_reason = get_overall_recommendation([(v['signal'], v['reason']) for v in recommendations.values()])
    recommendations['overall'] = {'signal': overall_signal, 'reason': overall_reason}
    trend = "Uptrend" if current_sma_50 > current_sma_200 and current_close > current_sma_50 else \
            "Downtrend" if current_sma_50 < current_sma_200 and current_close < current_sma_50 else "Sideways"
    technicals = Technicals(
        rsi=round(current_rsi, 2),
        sma_50=round(current_sma_50, 2),
        sma_200=round(current_sma_200, 2),
        macd=round(current_macd, 4),
        macd_signal=round(current_macd_signal, 4),
        macd_histogram=round(current_macd_hist, 4),
        stoch_k=round(current_stoch_k, 2),
        stoch_d=round(current_stoch_d, 2),
        adx=round(current_adx, 2),
        plus_di=round(current_plus_di, 2),
        minus_di=round(current_minus_di, 2),
        obv=round(current_obv, 0),
        adr=round(current_adr, 2),
        avg_adr=round(avg_adr, 2),
        trend=trend,
        bb_upper=round(current_bb_upper, 2),
        bb_middle=round(current_bb_middle, 2),
        bb_lower=round(current_bb_lower, 2),
        bb_width=round((current_bb_upper - current_bb_lower) / current_bb_middle, 4) if current_bb_middle else 0.0,
        recommendations=recommendations
    )
    return df, technicals

def format_technical_analysis(ticker: str, technicals: Technicals) -> str:
    if not technicals:
        return f"❌ Unable to analyze {ticker}"
    close = technicals.bb_middle  # Or use another field for price if preferred
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