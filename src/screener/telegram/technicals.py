import numpy as np
import pandas as pd
import yfinance as yf
import talib
from typing import Dict, Tuple, List
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

def calculate_technicals(ticker: str, period: str = "2y", interval: str = "1d") -> Technicals:
    try:
        logger.debug(f"Starting technical analysis for {ticker} (period={period}, interval={interval})")
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            logger.error(f"No data downloaded for ticker {ticker}")
            return None
        logger.debug(f"Downloaded {len(df)} days of data for {ticker}")
        if len(df) < 50:
            logger.error(f"Insufficient data for {ticker}: {len(df)} days")
            return None
        df = df.dropna()
        logger.debug(f"After dropna: {len(df)} days for {ticker}")
        if len(df) < 50:
            logger.error(f"Insufficient data after cleaning for {ticker}: {len(df)} days")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            ticker_col = df.columns[0][1]
            high = df[('High', ticker_col)].values.astype(float)
            low = df[('Low', ticker_col)].values.astype(float)
            close = df[('Close', ticker_col)].values.astype(float)
            volume = df[('Volume', ticker_col)].values.astype(float)
            open_price = df[('Open', ticker_col)].values.astype(float)
        else:
            high = df['High'].values.astype(float)
            low = df['Low'].values.astype(float)
            close = df['Close'].values.astype(float)
            volume = df['Volume'].values.astype(float)
            open_price = df['Open'].values.astype(float)
        if len(close) == 0:
            logger.error(f"Empty close price array for {ticker}")
            return None
        min_length = min(len(high), len(low), len(close), len(volume), len(open_price))
        logger.debug(f"Min length for {ticker}: {min_length}")
        if min_length < 50:
            logger.error(f"Insufficient data after length check for {ticker}: {min_length} days")
            return None
        high = high[-min_length:]
        low = low[-min_length:]
        close = close[-min_length:]
        volume = volume[-min_length:]
        open_price = open_price[-min_length:]
        logger.debug(f"After truncation for {ticker}: all arrays have length {len(close)}")
        if np.any(np.isnan(close)) or np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(volume)):
            logger.error(f"NaN values found in data for {ticker}")
            return None
        if np.any(np.isinf(close)) or np.any(np.isinf(high)) or np.any(np.isinf(low)) or np.any(np.isinf(volume)):
            logger.error(f"Infinite values found in data for {ticker}")
            return None
        logger.debug(f"Starting TA-Lib calculations for {ticker}")
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
            logger.debug(f"TA-Lib calculations completed successfully for {ticker}")
        except Exception as e:
            logger.error(f"TA-Lib calculation failed for {ticker}: {e}")
            logger.error(f"Array shapes - close: {close.shape}, high: {high.shape}, low: {low.shape}, volume: {volume.shape}")
            return None
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
        rsi_rec, rsi_reason = get_rsi_recommendation(current_rsi)
        bb_rec, bb_reason = get_bollinger_recommendation(current_close, current_bb_upper, current_bb_middle, current_bb_lower)
        macd_rec, macd_reason = get_macd_recommendation(current_macd, current_macd_signal, current_macd_hist)
        stoch_rec, stoch_reason = get_stochastic_recommendation(current_stoch_k, current_stoch_d)
        adx_rec, adx_reason = get_adx_recommendation(current_adx, current_plus_di, current_minus_di)
        obv_rec, obv_reason = get_obv_recommendation(current_obv, prev_obv, price_change)
        valid_adr = adr[~np.isnan(adr)]
        avg_adr = np.mean(valid_adr[-20:]) if len(valid_adr) >= 20 else current_adr
        adr_rec, adr_reason = get_adr_recommendation(current_adr, avg_adr)
        all_recommendations = [rsi_rec, bb_rec, macd_rec, stoch_rec, adx_rec, obv_rec, adr_rec]
        overall_rec, overall_reason = get_overall_recommendation([
            (rsi_rec, rsi_reason),
            (bb_rec, bb_reason),
            (macd_rec, macd_reason),
            (stoch_rec, stoch_reason),
            (adx_rec, adx_reason),
            (obv_rec, obv_reason),
            (adr_rec, adr_reason)
        ])
        trend = "Uptrend" if current_sma_50 > current_sma_200 and current_close > current_sma_50 else \
                "Downtrend" if current_sma_50 < current_sma_200 and current_close < current_sma_50 else "Sideways"
        logger.debug(f"Calculated technicals for {ticker}: RSI={current_rsi:.2f}, Overall={overall_rec}")
        recommendations = {
            "rsi": {"signal": rsi_rec, "reason": rsi_reason},
            "bollinger": {"signal": bb_rec, "reason": bb_reason},
            "macd": {"signal": macd_rec, "reason": macd_reason},
            "stochastic": {"signal": stoch_rec, "reason": stoch_reason},
            "adx": {"signal": adx_rec, "reason": adx_reason},
            "obv": {"signal": obv_rec, "reason": obv_reason},
            "adr": {"signal": adr_rec, "reason": adr_reason},
            "overall": {"signal": overall_rec, "reason": overall_reason}
        }
        return Technicals(
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
            bb_width=round((current_bb_upper - current_bb_lower) / current_bb_middle, 4),
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {str(e)}", exc_info=e)
        return None

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
    message += f"• RSI ({rsi:.1f}): {rsi_rec.get('signal', 'HOLD')} - {rsi_rec.get('reason', '')}\n"
    macd_rec = technicals.recommendations.get("macd", {}) if technicals.recommendations else {}
    macd_val = technicals.macd
    message += f"• MACD ({macd_val:.4f}): {macd_rec.get('signal', 'HOLD')} - {macd_rec.get('reason', '')}\n"
    bb_rec = technicals.recommendations.get("bollinger", {}) if technicals.recommendations else {}
    bb_lower = technicals.bb_lower
    bb_upper = technicals.bb_upper
    message += f"• BB ({bb_lower:.2f} - {bb_upper:.2f}): {bb_rec.get('signal', 'HOLD')} - {bb_rec.get('reason', '')}\n"
    stoch_rec = technicals.recommendations.get("stochastic", {}) if technicals.recommendations else {}
    stoch_k = technicals.stoch_k
    stoch_d = technicals.stoch_d
    message += f"• Stoch K/D ({stoch_k:.1f}/{stoch_d:.1f}): {stoch_rec.get('signal', 'HOLD')} - {stoch_rec.get('reason', '')}\n"
    adx_rec = technicals.recommendations.get("adx", {}) if technicals.recommendations else {}
    adx_val = technicals.adx
    message += f"• ADX ({adx_val:.1f}): {adx_rec.get('signal', 'HOLD')} - {adx_rec.get('reason', '')}\n"
    obv_rec = technicals.recommendations.get("obv", {}) if technicals.recommendations else {}
    obv_val = technicals.obv
    message += f"• OBV: {obv_rec.get('signal', 'HOLD')} - {obv_rec.get('reason', '')}\n"
    adr_rec = technicals.recommendations.get("adr", {}) if technicals.recommendations else {}
    adr_val = technicals.adr
    message += f"• ADR ({adr_val:.2f}): {adr_rec.get('signal', 'HOLD')} - {adr_rec.get('reason', '')}\n"
    return message

if __name__ == "__main__":
    test_ticker = "AAPL"
    print(f"Testing technical analysis for {test_ticker}...")
    technicals = calculate_technicals(test_ticker)
    if technicals:
        print(format_technical_analysis(test_ticker, technicals))
    else:
        print(f"Failed to analyze {test_ticker}") 