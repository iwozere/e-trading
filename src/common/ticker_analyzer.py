from src.common.ticker_chart import generate_chart
from src.common.fundamentals import get_fundamentals_unified, format_fundamental_analysis
from src.common.common import get_ohlcv, determine_provider
from src.model.telegram_bot import TickerAnalysis
from src.common.technicals import format_technical_analysis
from src.indicators.service import get_unified_indicator_service
from src.indicators.models import IndicatorCalculationRequest, IndicatorCategory
from src.model.telegram_bot import Technicals
import numpy as np
import talib

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


async def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d", provider: str = None) -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations, supporting multiple providers."""

    try:
        # Infer provider if not specified
        if not provider:
            provider = determine_provider(ticker)

        # Get OHLCV data using common function
        df = get_ohlcv(ticker, interval, period, provider)

        # Get fundamentals using unified function with traditional fallback (only for stock providers)
        fundamentals = None
        if provider.lower() in ["yf", "av", "fh", "td", "pg"]:
            try:
                fundamentals = await get_fundamentals_unified(ticker, provider)
            except Exception:
                _logger.exception("Error getting fundamentals using unified function, falling back to traditional function")
                fundamentals = None

        # Get unified indicator service
        indicator_service = get_unified_indicator_service()

        # Create indicator calculation request
        request = IndicatorCalculationRequest(
            ticker=ticker,
            indicators=["RSI", "MACD", "BollingerBands", "SMA", "EMA", "ADX", "ATR", "Stochastic", "WilliamsR", "CCI", "ROC", "MFI"],
            timeframe=interval,
            period=period,
            provider=provider
        )

        # Calculate indicators using unified service
        indicator_set = await indicator_service.get_indicators(request)

        # Extract technical indicators from unified result
        # Initialize technical data with default values
        technical_data = {
            'rsi': None,
            'sma_fast': None,
            'sma_slow': None,
            'ema_fast': None,
            'ema_slow': None,
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
            'cci': None,
            'roc': None,
            'mfi': None,
            'williams_r': None,
            'atr': None,
            'recommendations': None
        }

        # Map indicator names to Technicals fields
        indicator_mapping = {
            'RSI': 'rsi',
            'SMA_FAST': 'sma_fast',
            'SMA_SLOW': 'sma_slow',
            'EMA_FAST': 'ema_fast',
            'EMA_SLOW': 'ema_slow',
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
            'CCI': 'cci',
            'ROC': 'roc',
            'MFI': 'mfi',
            'WILLIAMS_R': 'williams_r',
            'ATR': 'atr',
        }

        for name, indicator in indicator_set.technical_indicators.items():
            if name in indicator_mapping:
                field_name = indicator_mapping[name]
                technical_data[field_name] = indicator.value

        # Create initial technicals object (will be updated later)
        technicals = Technicals(**technical_data)

        # Add calculated indicators to DataFrame for chart generation
        # This is necessary because the chart generation expects indicators in DataFrame columns
        df_with_indicators = df.copy()

        # Add indicators to DataFrame based on the indicator set
        try:
            if indicator_set and indicator_set.technical_indicators:
                for name, indicator in indicator_set.technical_indicators.items():
                    if name in indicator_mapping:
                        field_name = indicator_mapping[name]
                        # For now, we'll add the current value to all rows (simplified approach)
                        # In a more sophisticated implementation, we'd calculate the full time series
                        df_with_indicators[field_name] = indicator.value
        except Exception as e:
            _logger.exception("Error adding indicators to DataFrame:")
            _logger.error("Indicator set: %s", indicator_set)
            _logger.error("Indicator mapping: %s", indicator_mapping)
            raise

        # Calculate full time series for key indicators using TA-Lib
        # This ensures the chart has the complete indicator data
        try:

            # Validate data types and quality
            close = df_with_indicators['close'].values.astype(float)
            high = df_with_indicators['high'].values.astype(float)
            low = df_with_indicators['low'].values.astype(float)
            volume = df_with_indicators['volume'].values.astype(float)

            # Check for NaN or infinite values
            if np.any(np.isnan(close)) or np.any(np.isinf(close)):
                _logger.warning("Found NaN or infinite values in close prices, cleaning data...")
                close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)

            if np.any(np.isnan(high)) or np.any(np.isinf(high)):
                _logger.warning("Found NaN or infinite values in high prices, cleaning data...")
                high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)

            if np.any(np.isnan(low)) or np.any(np.isinf(low)):
                _logger.warning("Found NaN or infinite values in low prices, cleaning data...")
                low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)

            if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
                _logger.warning("Found NaN or infinite values in volume, cleaning data...")
                volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure we have enough data for calculations
            if len(close) < 50:
                _logger.warning("Insufficient data for indicator calculations: %d points (need at least 50)", len(close))
                raise ValueError(f"Insufficient data for indicator calculations: {len(close)} points")

            # Calculate indicators and add to DataFrame
            try:
                # RSI
                rsi = talib.RSI(close, timeperiod=14)
                df_with_indicators['rsi'] = rsi

                # MACD
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                df_with_indicators['macd'] = macd
                df_with_indicators['macd_signal'] = macd_signal
                df_with_indicators['macd_hist'] = macd_hist

                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                df_with_indicators['bb_upper'] = bb_upper
                df_with_indicators['bb_middle'] = bb_middle
                df_with_indicators['bb_lower'] = bb_lower

                # Stochastic
                stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                df_with_indicators['stoch_k'] = stoch_k
                df_with_indicators['stoch_d'] = stoch_d

                # ADX
                adx = talib.ADX(high, low, close, timeperiod=14)
                plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
                minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
                df_with_indicators['adx'] = adx
                df_with_indicators['plus_di'] = plus_di
                df_with_indicators['minus_di'] = minus_di

                # Moving Averages
                sma_fast = talib.SMA(close, timeperiod=50)
                sma_slow = talib.SMA(close, timeperiod=200)
                ema_fast = talib.EMA(close, timeperiod=12)
                ema_slow = talib.EMA(close, timeperiod=26)
                df_with_indicators['sma_fast'] = sma_fast
                df_with_indicators['sma_slow'] = sma_slow
                df_with_indicators['ema_fast'] = ema_fast
                df_with_indicators['ema_slow'] = ema_slow

                # Additional indicators
                cci = talib.CCI(high, low, close, timeperiod=14)
                roc = talib.ROC(close, timeperiod=10)
                mfi = talib.MFI(high, low, close, volume, timeperiod=14)
                williams_r = talib.WILLR(high, low, close, timeperiod=14)
                atr = talib.ATR(high, low, close, timeperiod=14)
                obv = talib.OBV(close, volume)

                df_with_indicators['cci'] = cci
                df_with_indicators['roc'] = roc
                df_with_indicators['mfi'] = mfi
                df_with_indicators['williams_r'] = williams_r
                df_with_indicators['atr'] = atr
                df_with_indicators['obv'] = obv

                # ADR (Average Daily Range)
                daily_range = high - low
                adr = talib.SMA(daily_range, timeperiod=14)
                df_with_indicators['adr'] = adr



            except Exception as e:
                _logger.warning("Error calculating some indicators: %s", e)
                # Continue with partial indicators

        except Exception as e:
            _logger.warning("Error in technical indicator calculation: %s", e)
            # Continue without indicators

        # Extract current indicator values from the DataFrame for technical analysis
        current_indicators = {}
        current_price = None
        if not df_with_indicators.empty:
            # Get the last row (most recent data)
            last_row = df_with_indicators.iloc[-1]
            # Get current price from the last row
            current_price = last_row.get('close')

            # Extract current values for technical analysis
            current_indicators = {
                'rsi': last_row.get('rsi'),
                'sma_fast': last_row.get('sma_fast'),
                'sma_slow': last_row.get('sma_slow'),
                'ema_fast': last_row.get('ema_fast'),
                'ema_slow': last_row.get('ema_slow'),
                'macd': last_row.get('macd'),
                'macd_signal': last_row.get('macd_signal'),
                'macd_histogram': last_row.get('macd_hist'),
                'stoch_k': last_row.get('stoch_k'),
                'stoch_d': last_row.get('stoch_d'),
                'adx': last_row.get('adx'),
                'plus_di': last_row.get('plus_di'),
                'minus_di': last_row.get('minus_di'),
                'obv': last_row.get('obv'),
                'adr': last_row.get('adr'),
                'avg_adr': None,  # Not calculated in DataFrame
                'trend': 'NEUTRAL',  # Default trend
                'bb_upper': last_row.get('bb_upper'),
                'bb_middle': last_row.get('bb_middle'),
                'bb_lower': last_row.get('bb_lower'),
                'bb_width': None,  # Not calculated in DataFrame
                'cci': last_row.get('cci'),
                'roc': last_row.get('roc'),
                'mfi': last_row.get('mfi'),
                'williams_r': last_row.get('williams_r'),
                'atr': last_row.get('atr'),
                'recommendations': technicals.recommendations if technicals else None
            }

            # Generate recommendations for the technicals using current values
            recommendations = {}
            try:
                from src.common.recommendation_engine import RecommendationEngine
                recommendation_engine = RecommendationEngine()

                # Generate recommendations for each indicator
                if current_indicators.get('rsi') is not None:
                    rec = recommendation_engine.get_recommendation('RSI', current_indicators['rsi'])
                    recommendations['rsi'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('macd') is not None:
                    context = {
                        'macd_signal': current_indicators.get('macd_signal'),
                        'macd_histogram': current_indicators.get('macd_histogram')
                    }
                    rec = recommendation_engine.get_recommendation('MACD', current_indicators['macd'], context)
                    recommendations['macd'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('adx') is not None:
                    context = {
                        'plus_di': current_indicators.get('plus_di'),
                        'minus_di': current_indicators.get('minus_di')
                    }
                    rec = recommendation_engine.get_recommendation('ADX', current_indicators['adx'], context)
                    recommendations['adx'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('stoch_k') is not None:
                    context = {'stoch_d': current_indicators.get('stoch_d')}
                    rec = recommendation_engine.get_recommendation('STOCH_K', current_indicators['stoch_k'], context)
                    recommendations['stoch_k'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('bb_upper') is not None and current_indicators.get('bb_lower') is not None:
                    context = {
                        'current_price': current_price,
                        'bb_upper': current_indicators.get('bb_upper'),
                        'bb_lower': current_indicators.get('bb_lower')
                    }
                    rec = recommendation_engine.get_recommendation('BB_MIDDLE', current_indicators['bb_middle'], context)
                    recommendations['bb_middle'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('obv') is not None:
                    context = {'current_price': current_price}
                    rec = recommendation_engine.get_recommendation('OBV', current_indicators['obv'], context)
                    recommendations['obv'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('adr') is not None:
                    context = {'current_price': current_price}
                    rec = recommendation_engine.get_recommendation('ADR', current_indicators['adr'], context)
                    recommendations['adr'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                # Generate SMA/EMA recommendations with trend context
                if current_indicators.get('sma_fast') is not None:
                    # Determine MA trend based on fast vs slow MA relationship
                    ma_trend = 'unknown'
                    if current_indicators.get('sma_slow') is not None:
                        if current_indicators['sma_fast'] > current_indicators['sma_slow']:
                            ma_trend = 'up'
                        elif current_indicators['sma_fast'] < current_indicators['sma_slow']:
                            ma_trend = 'down'
                        else:
                            ma_trend = 'sideways'

                    context = {
                        'current_price': current_price,
                        'ma_trend': ma_trend,
                        'fast_ma': current_indicators.get('sma_fast'),
                        'slow_ma': current_indicators.get('sma_slow')
                    }
                    rec = recommendation_engine.get_recommendation('SMA_FAST', current_indicators['sma_fast'], context)
                    recommendations['sma_fast'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('sma_slow') is not None:
                    # Determine MA trend based on fast vs slow MA relationship
                    ma_trend = 'unknown'
                    if current_indicators.get('sma_fast') is not None:
                        if current_indicators['sma_fast'] > current_indicators['sma_slow']:
                            ma_trend = 'up'
                        elif current_indicators['sma_fast'] < current_indicators['sma_slow']:
                            ma_trend = 'down'
                        else:
                            ma_trend = 'sideways'

                    context = {
                        'current_price': current_price,
                        'ma_trend': ma_trend,
                        'fast_ma': current_indicators.get('sma_fast'),
                        'slow_ma': current_indicators.get('sma_slow')
                    }
                    rec = recommendation_engine.get_recommendation('SMA_SLOW', current_indicators['sma_slow'], context)
                    recommendations['sma_slow'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('ema_fast') is not None:
                    # Determine MA trend based on fast vs slow EMA relationship
                    ma_trend = 'unknown'
                    if current_indicators.get('ema_slow') is not None:
                        if current_indicators['ema_fast'] > current_indicators['ema_slow']:
                            ma_trend = 'up'
                        elif current_indicators['ema_fast'] < current_indicators['ema_slow']:
                            ma_trend = 'down'
                        else:
                            ma_trend = 'sideways'

                    context = {
                        'current_price': current_price,
                        'ma_trend': ma_trend,
                        'fast_ma': current_indicators.get('ema_fast'),
                        'slow_ma': current_indicators.get('ema_slow')
                    }
                    rec = recommendation_engine.get_recommendation('EMA_FAST', current_indicators['ema_fast'], context)
                    recommendations['ema_fast'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

                if current_indicators.get('ema_slow') is not None:
                    # Determine MA trend based on fast vs slow EMA relationship
                    ma_trend = 'unknown'
                    if current_indicators.get('ema_fast') is not None:
                        if current_indicators['ema_fast'] > current_indicators['ema_slow']:
                            ma_trend = 'up'
                        elif current_indicators['ema_fast'] < current_indicators['ema_slow']:
                            ma_trend = 'down'
                        else:
                            ma_trend = 'sideways'

                    context = {
                        'current_price': current_price,
                        'ma_trend': ma_trend,
                        'fast_ma': current_indicators.get('ema_fast'),
                        'slow_ma': current_indicators.get('ema_slow')
                    }
                    rec = recommendation_engine.get_recommendation('EMA_SLOW', current_indicators['ema_slow'], context)
                    recommendations['ema_slow'] = {
                        'signal': rec.recommendation.value,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    }

            except Exception as e:
                _logger.warning("Error generating recommendations: %s", e)
                recommendations = {}

            # Add recommendations to current indicators
            current_indicators['recommendations'] = recommendations

            # Create updated technicals with current values and recommendations
            current_technicals = Technicals(**current_indicators)
        else:
            current_technicals = technicals

        # Create a temporary TickerAnalysis for charting, since we need to pass the object
        temp_analysis = TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider,
            period=period,
            interval=interval,
            fundamentals=fundamentals,
            technicals=current_technicals,
            chart_image=None,
            ohlcv=df_with_indicators
        )
        # Generate chart
        try:
            chart_bytes = generate_chart(temp_analysis.ticker, temp_analysis.ohlcv)
        except Exception as e:
            _logger.exception("Error generating chart:")
            chart_bytes = None

        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider,
            period=period,
            interval=interval,
            fundamentals=fundamentals,
            technicals=current_technicals,
            chart_image=chart_bytes,
            ohlcv=df_with_indicators,
            current_price=current_price
        )
    except Exception as e:
        _logger.exception("Error in analyze_ticker:")
        raise

def format_ticker_report(analysis: TickerAnalysis) -> dict:
    """
    Formats a TickerAnalysis into a message and generates a chart image (as bytes).
    Returns a dict with 'message' and 'chart_bytes' (bytes or None).
    The caller should use 'chart_bytes' for sending to Telegram/email.
    """
    # Format fundamentals
    fundamentals_msg = format_fundamental_analysis(analysis.fundamentals)
    # Format technicals
    technicals_msg = ""
    if analysis.technicals is not None:
        # Use the current_price from the analysis if available
        current_price = getattr(analysis, 'current_price', None)
        technicals_msg = format_technical_analysis(analysis.ticker, analysis.technicals, current_price)
    # Generate chart as bytes
    chart_bytes = None
    if analysis.ohlcv is not None:
        try:
            chart_bytes = generate_chart(analysis.ticker, analysis.ohlcv)
            analysis.chart_image = chart_bytes
        except Exception as e:
            _logger.exception("Error generating chart:")
            chart_bytes = None
            analysis.chart_image = None

    # Compose full message
    company_name = "Unknown"
    if analysis.fundamentals:
        company_name = analysis.fundamentals.company_name or "Unknown"
    full_msg = f"{analysis.ticker} - {company_name}\n\n{fundamentals_msg}\n{technicals_msg}"
    return {
        "message": full_msg.strip(),
        "chart_bytes": chart_bytes
    }
