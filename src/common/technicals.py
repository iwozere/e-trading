import pandas as pd
from typing import Optional
from src.notification.logger import setup_logger
from src.model.telegram_bot import Technicals
from src.common.indicator_service import get_indicator_service
from src.models.indicators import IndicatorCalculationRequest, IndicatorCategory

_logger = setup_logger(__name__)


async def calculate_technicals_unified(ticker: str, period: str = "2y", interval: str = "1d", provider: str = None) -> Technicals:
    """
    Calculate technical indicators using the unified indicator service.
    This is the new recommended approach.
    """
    indicator_service = get_indicator_service()

    request = IndicatorCalculationRequest(
        ticker=ticker,
        indicators=["RSI", "MACD", "MACD_SIGNAL", "MACD_HISTOGRAM", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "ADX", "PLUS_DI", "MINUS_DI", "ATR", "STOCH_K", "STOCH_D", "WILLIAMS_R", "CCI", "ROC", "MFI", "OBV", "ADR"],
        timeframe=interval,
        period=period,
        provider=provider,
        include_recommendations=True
    )

    indicator_set = await indicator_service.get_indicators(request)

    # Convert IndicatorSet to Technicals object for backward compatibility
    from src.model.telegram_bot import Technicals

    # Extract technical indicators from the indicator set
    # Map indicator names to Technicals class fields
    technical_data = {
        'rsi': None,
        'sma_50': None,
        'sma_200': None,
        'macd': None,
        'macd_signal': None,
        'macd_histogram': None,
        'stoch_k': None,
        'stoch_d': None,
        'adx': None,
        'plus_di': None,
        'minus_di': None,
        'obv': None,
        'adr': None,
        'avg_adr': None,
        'trend': 'NEUTRAL',
        'bb_upper': None,
        'bb_middle': None,
        'bb_lower': None,
        'bb_width': None,
        'recommendations': {}
    }

    # Map indicator names to Technicals fields
    indicator_mapping = {
        'RSI': 'rsi',
        'SMA_50': 'sma_50',
        'SMA_200': 'sma_200',
        'MACD': 'macd',
        'MACD_SIGNAL': 'macd_signal',
        'MACD_HISTOGRAM': 'macd_histogram',
        'STOCH_K': 'stoch_k',
        'STOCH_D': 'stoch_d',
        'ADX': 'adx',
        'PLUS_DI': 'plus_di',
        'MINUS_DI': 'minus_di',
        'BB_UPPER': 'bb_upper',
        'BB_MIDDLE': 'bb_middle',
        'BB_LOWER': 'bb_lower',
        'OBV': 'obv',
        'ADR': 'adr',
    }

    # Extract indicator values and build recommendations
    recommendations = {}

    for name, indicator in indicator_set.technical_indicators.items():
        if name in indicator_mapping:
            field_name = indicator_mapping[name]
            technical_data[field_name] = indicator.value

            # Add recommendation if available
            if indicator.recommendation:
                rec_type = indicator.recommendation.recommendation.value
                if rec_type == "STRONG_BUY":
                    signal = "STRONG_BUY"
                elif rec_type == "BUY":
                    signal = "BUY"
                elif rec_type == "STRONG_SELL":
                    signal = "STRONG_SELL"
                elif rec_type == "SELL":
                    signal = "SELL"
                else:
                    signal = "HOLD"

                recommendations[name.lower()] = {
                    "signal": signal,
                    "reason": indicator.recommendation.reason,
                    "confidence": indicator.recommendation.confidence
                }

    # Add overall recommendation if composite recommendation is available
    if indicator_set.overall_recommendation:
        overall_rec = indicator_set.overall_recommendation
        rec_type = overall_rec.recommendation.value
        if rec_type == "STRONG_BUY":
            signal = "STRONG_BUY"
        elif rec_type == "BUY":
            signal = "BUY"
        elif rec_type == "STRONG_SELL":
            signal = "STRONG_SELL"
        elif rec_type == "SELL":
            signal = "SELL"
        else:
            signal = "HOLD"

        recommendations["overall"] = {
            "signal": signal,
            "reason": overall_rec.reasoning,
            "confidence": overall_rec.confidence
        }
    else:
        # Generate a basic overall recommendation based on available indicators
        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        for rec in recommendations.values():
            if rec["signal"] in ["STRONG_BUY", "BUY"]:
                buy_signals += 1
            elif rec["signal"] in ["STRONG_SELL", "SELL"]:
                sell_signals += 1
            total_signals += 1

        if total_signals > 0:
            if buy_signals > sell_signals:
                overall_signal = "BUY"
                reason = f"More buy signals ({buy_signals} vs {sell_signals})"
            elif sell_signals > buy_signals:
                overall_signal = "SELL"
                reason = f"More sell signals ({sell_signals} vs {buy_signals})"
            else:
                overall_signal = "HOLD"
                reason = "Mixed signals"
        else:
            overall_signal = "HOLD"
            reason = "No clear signals available"

        recommendations["overall"] = {
            "signal": overall_signal,
            "reason": reason,
            "confidence": 0.5
        }

    # Calculate trend based on SMA comparison
    if technical_data['sma_50'] is not None and technical_data['sma_200'] is not None:
        if technical_data['sma_50'] > technical_data['sma_200']:
            technical_data['trend'] = 'BULLISH'
        else:
            technical_data['trend'] = 'BEARISH'

    # Calculate BB width if BB values are available
    if (technical_data['bb_upper'] is not None and
        technical_data['bb_middle'] is not None and
        technical_data['bb_lower'] is not None):
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
    message += f"🎯 Overall: *{overall_rec.get('signal', 'HOLD')}*\n"
    message += f"💡 {overall_rec.get('reason', 'No reason available')}\n\n"
    message += "*Key Indicators:*\n"
    rsi_rec = technicals.recommendations.get("rsi", {}) if technicals.recommendations else {}
    if rsi is not None:
        message += f"• RSI ({rsi:.1f}): {rsi_rec.get('signal', 'HOLD')} - {rsi_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• RSI (N/A): {rsi_rec.get('signal', 'HOLD')} - {rsi_rec.get('reason', 'No reason provided')}\n"
    macd_rec = technicals.recommendations.get("macd", {}) if technicals.recommendations else {}
    macd_val = technicals.macd
    if macd_val is not None:
        message += f"• MACD ({macd_val:.4f}): {macd_rec.get('signal', 'HOLD')} - {macd_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• MACD (N/A): {macd_rec.get('signal', 'HOLD')} - {macd_rec.get('reason', 'No reason provided')}\n"
    bb_rec = technicals.recommendations.get("bollinger", {}) if technicals.recommendations else {}
    bb_lower = technicals.bb_lower
    bb_upper = technicals.bb_upper
    if bb_lower is not None and bb_upper is not None:
        message += f"• BB ({bb_lower:.2f} - {bb_upper:.2f}): {bb_rec.get('signal', 'HOLD')} - {bb_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• BB (N/A): {bb_rec.get('signal', 'HOLD')} - {bb_rec.get('reason', 'No reason provided')}\n"
    stoch_rec = technicals.recommendations.get("stochastic", {}) if technicals.recommendations else {}
    stoch_k = technicals.stoch_k
    stoch_d = technicals.stoch_d
    if stoch_k is not None and stoch_d is not None:
        message += f"• Stoch K/D ({stoch_k:.1f}/{stoch_d:.1f}): {stoch_rec.get('signal', 'HOLD')} - {stoch_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• Stoch K/D (N/A): {stoch_rec.get('signal', 'HOLD')} - {stoch_rec.get('reason', 'No reason provided')}\n"
    adx_rec = technicals.recommendations.get("adx", {}) if technicals.recommendations else {}
    adx_val = technicals.adx
    if adx_val is not None:
        message += f"• ADX ({adx_val:.1f}): {adx_rec.get('signal', 'HOLD')} - {adx_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• ADX (N/A): {adx_rec.get('signal', 'HOLD')} - {adx_rec.get('reason', 'No reason provided')}\n"
    obv_rec = technicals.recommendations.get("obv", {}) if technicals.recommendations else {}
    obv_val = technicals.obv
    if obv_val is not None:
        message += f"• OBV ({obv_val:.0f}): {obv_rec.get('signal', 'HOLD')} - {obv_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• OBV (N/A): {obv_rec.get('signal', 'HOLD')} - {obv_rec.get('reason', 'No reason provided')}\n"
    adr_rec = technicals.recommendations.get("adr", {}) if technicals.recommendations else {}
    adr_val = technicals.adr
    if adr_val is not None:
        message += f"• ADR ({adr_val:.2f}): {adr_rec.get('signal', 'HOLD')} - {adr_rec.get('reason', 'No reason provided')}\n"
    else:
        message += f"• ADR (N/A): {adr_rec.get('signal', 'HOLD')} - {adr_rec.get('reason', 'No reason provided')}\n"
    return message




