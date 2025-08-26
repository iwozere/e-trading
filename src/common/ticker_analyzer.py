from src.common.ticker_chart import generate_chart
from src.common.fundamentals import get_fundamentals, format_fundamental_analysis
from src.common import get_ohlcv, determine_provider, get_ticker_info
from src.model.telegram_bot import TickerAnalysis
from src.common.technicals import format_technical_analysis
from src.common.indicator_service import get_indicator_service
from src.models.indicators import IndicatorCalculationRequest, IndicatorCategory
from src.notification.logger import setup_logger
#from src.backtester.plotter.base_plotter import
import numpy as np


_logger = setup_logger(__name__)


async def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d", provider: str = None) -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations, supporting multiple providers."""
    _logger.info("Analyzing ticker: %s, period: %s, interval: %s, provider: %s", ticker, period, interval, provider)
    try:
        # Infer provider if not specified
        if not provider:
            provider = determine_provider(ticker)

        _logger.debug("provider: %s", provider)

        # Get OHLCV data using common function
        df = get_ohlcv(ticker, interval, period, provider)

        # Get fundamentals using common function (only for stock providers)
        fundamentals = None
        if provider.lower() in ["yf", "av", "fh", "td", "pg"]:
            fundamentals = get_fundamentals(ticker, provider)

        _logger.debug("Downloaded data for %s", ticker)

        # Get unified indicator service
        indicator_service = get_indicator_service()

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
        from src.model.telegram_bot import Technicals

        # Initialize technical data with default values
        technical_data = {
            'rsi': None,
            'sma_50': None,
            'sma_200': None,
            'ema_12': None,
            'ema_26': None,
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
            'SMA_50': 'sma_50',
            'SMA_200': 'sma_200',
            'EMA_12': 'ema_12',
            'EMA_26': 'ema_26',
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

        # Generate recommendations for the technicals
        recommendations = {}
        try:
            from src.common.recommendation_engine import RecommendationEngine
            recommendation_engine = RecommendationEngine()

            # Generate recommendations for each indicator
            if technical_data.get('rsi') is not None:
                rec = recommendation_engine.get_recommendation('RSI', technical_data['rsi'])
                recommendations['rsi'] = {
                    'signal': rec.recommendation.value,
                    'confidence': rec.confidence,
                    'reason': rec.reason
                }

            if technical_data.get('macd') is not None:
                context = {
                    'macd_signal': technical_data.get('macd_signal'),
                    'macd_histogram': technical_data.get('macd_histogram')  # Fixed key name
                }
                rec = recommendation_engine.get_recommendation('MACD', technical_data['macd'], context)
                recommendations['macd'] = {
                    'signal': rec.recommendation.value,
                    'confidence': rec.confidence,
                    'reason': rec.reason
                }

            if technical_data.get('adx') is not None:
                context = {
                    'plus_di': technical_data.get('plus_di'),
                    'minus_di': technical_data.get('minus_di')
                }
                rec = recommendation_engine.get_recommendation('ADX', technical_data['adx'], context)
                recommendations['adx'] = {
                    'signal': rec.recommendation.value,
                    'confidence': rec.confidence,
                    'reason': rec.reason
                }

            if technical_data.get('stoch_k') is not None:
                context = {'stoch_d': technical_data.get('stoch_d')}
                rec = recommendation_engine.get_recommendation('STOCH_K', technical_data['stoch_k'], context)
                recommendations['stochastic'] = {
                    'signal': rec.recommendation.value,
                    'confidence': rec.confidence,
                    'reason': rec.reason
                }

            if technical_data.get('bb_upper') is not None and technical_data.get('bb_lower') is not None:
                context = {
                    'current_price': technical_data.get('bb_middle'),
                    'bb_upper': technical_data.get('bb_upper'),
                    'bb_lower': technical_data.get('bb_lower')
                }
                rec = recommendation_engine.get_recommendation('BB_MIDDLE', technical_data['bb_middle'], context)
                recommendations['bollinger'] = {
                    'signal': rec.recommendation.value,
                    'confidence': rec.confidence,
                    'reason': rec.reason
                }

        except Exception as e:
            _logger.warning("Error generating recommendations: %s", e)
            recommendations = {}

        # Add recommendations to technical data
        technical_data['recommendations'] = recommendations
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
            _logger.error(f"Error adding indicators to DataFrame: {e}")
            _logger.error(f"Indicator set: {indicator_set}")
            _logger.error(f"Indicator mapping: {indicator_mapping}")
            raise

        # Calculate full time series for key indicators using TA-Lib
        # This ensures the chart has the complete indicator data
        try:
            import talib

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
                _logger.warning(f"Insufficient data for indicator calculations: {len(close)} points (need at least 50)")
                raise ValueError(f"Insufficient data for indicator calculations: {len(close)} points")

            # RSI
            try:
                df_with_indicators['rsi'] = talib.RSI(close, timeperiod=14)
            except Exception as e:
                _logger.error(f"Error calculating RSI: {e}")
                df_with_indicators['rsi'] = np.nan

            # MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                df_with_indicators['macd'] = macd
                df_with_indicators['macd_signal'] = macd_signal
                df_with_indicators['macd_hist'] = macd_hist
            except Exception as e:
                _logger.error(f"Error calculating MACD: {e}")
                df_with_indicators['macd'] = np.nan
                df_with_indicators['macd_signal'] = np.nan
                df_with_indicators['macd_hist'] = np.nan

            # Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                df_with_indicators['bb_upper'] = bb_upper
                df_with_indicators['bb_middle'] = bb_middle
                df_with_indicators['bb_lower'] = bb_lower
            except Exception as e:
                _logger.error(f"Error calculating Bollinger Bands: {e}")
                df_with_indicators['bb_upper'] = np.nan
                df_with_indicators['bb_middle'] = np.nan
                df_with_indicators['bb_lower'] = np.nan

            # Moving Averages
            try:
                df_with_indicators['sma_50'] = talib.SMA(close, timeperiod=50)
                df_with_indicators['sma_200'] = talib.SMA(close, timeperiod=200)
            except Exception as e:
                _logger.error(f"Error calculating Moving Averages: {e}")
                df_with_indicators['sma_50'] = np.nan
                df_with_indicators['sma_200'] = np.nan

            # Stochastic
            try:
                stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                df_with_indicators['stoch_k'] = stoch_k
                df_with_indicators['stoch_d'] = stoch_d
            except Exception as e:
                _logger.error(f"Error calculating Stochastic: {e}")
                df_with_indicators['stoch_k'] = np.nan
                df_with_indicators['stoch_d'] = np.nan

            # ADX
            try:
                adx = talib.ADX(high, low, close, timeperiod=14)
                plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
                minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
                df_with_indicators['adx'] = adx
                df_with_indicators['plus_di'] = plus_di
                df_with_indicators['minus_di'] = minus_di
            except Exception as e:
                _logger.error(f"Error calculating ADX: {e}")
                df_with_indicators['adx'] = np.nan
                df_with_indicators['plus_di'] = np.nan
                df_with_indicators['minus_di'] = np.nan

            # OBV
            try:
                df_with_indicators['obv'] = talib.OBV(close, volume)
            except Exception as e:
                _logger.error(f"Error calculating OBV: {e}")
                df_with_indicators['obv'] = np.nan

            # Exponential Moving Averages
            try:
                df_with_indicators['ema_12'] = talib.EMA(close, timeperiod=12)
                df_with_indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            except Exception as e:
                _logger.error(f"Error calculating EMA: {e}")
                df_with_indicators['ema_12'] = np.nan
                df_with_indicators['ema_26'] = np.nan

            # CCI (Commodity Channel Index)
            # CCI
            try:
                df_with_indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)
            except Exception as e:
                _logger.error(f"Error calculating CCI: {e}")
                df_with_indicators['cci'] = np.nan

            # ROC (Rate of Change)
            # ROC
            try:
                df_with_indicators['roc'] = talib.ROC(close, timeperiod=10)
            except Exception as e:
                _logger.error(f"Error calculating ROC: {e}")
                df_with_indicators['roc'] = np.nan

            # MFI (Money Flow Index)
            # MFI
            try:
                df_with_indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            except Exception as e:
                _logger.error(f"Error calculating MFI: {e}")
                df_with_indicators['mfi'] = np.nan

            # Williams %R
            try:
                df_with_indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            except Exception as e:
                _logger.error(f"Error calculating Williams %R: {e}")
                df_with_indicators['williams_r'] = np.nan

            # ATR (Average True Range)
            # ATR
            try:
                df_with_indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            except Exception as e:
                _logger.error(f"Error calculating ATR: {e}")
                df_with_indicators['atr'] = np.nan

            # ADR (Average Daily Range) - Custom calculation
            # ADR
            try:
                # Calculate daily range (high - low) and then take the average
                daily_range = high - low
                df_with_indicators['adr'] = talib.SMA(daily_range, timeperiod=14)
            except Exception as e:
                _logger.error(f"Error calculating ADR: {e}")
                df_with_indicators['adr'] = np.nan



        except Exception as e:
            _logger.exception("Error calculating full indicator time series for chart: %s", e)
            _logger.error("Indicator calculation error details - Error type: %s, Message: %s", type(e).__name__, str(e))
            _logger.error("DataFrame info at error time:")
            _logger.error(f"  Shape: {df_with_indicators.shape}")
            _logger.error(f"  Columns: {list(df_with_indicators.columns)}")
            _logger.error(f"  Data types: {df_with_indicators.dtypes.to_dict()}")
            # Continue with the simplified approach if full calculation fails

        # Extract current indicator values from the DataFrame for technical analysis
        current_indicators = {}
        if not df_with_indicators.empty:
            # Get the last row (most recent data)
            last_row = df_with_indicators.iloc[-1]

            # Extract current values for technical analysis
            current_indicators = {
                'rsi': last_row.get('rsi'),
                'sma_50': last_row.get('sma_50'),
                'sma_200': last_row.get('sma_200'),
                'ema_12': last_row.get('ema_12'),
                'ema_26': last_row.get('ema_26'),
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

            # Create updated technicals with current values
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
            _logger.error(f"Error generating chart: {e}")
            chart_bytes = None
        # Get current price from the last row
        current_price = None
        if not df_with_indicators.empty:
            current_price = df_with_indicators['close'].iloc[-1]

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
            _logger.error(f"Error generating chart: {e}")
            chart_bytes = None
            analysis.chart_image = None

    # Compose full message
    full_msg = f"**{analysis.ticker}** - {getattr(analysis.fundamentals, 'company_name', 'Unknown') if analysis.fundamentals else 'Unknown'}\n\n{fundamentals_msg}\n{technicals_msg}"
    return {
        "message": full_msg.strip(),
        "chart_bytes": chart_bytes
    }
