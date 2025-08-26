# ticker_bot/analyzer/chart.py

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Any
from src.notification.logger import setup_logger
from src.model.telegram_bot import TickerAnalysis
from datetime import datetime
from pathlib import Path

_logger = setup_logger(__name__)

# Modern color palette for professional charts
CHART_COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'danger': '#d62728',       # Red
    'warning': '#ff7f0e',      # Orange
    'info': '#17a2b8',         # Cyan
    'light': '#f8f9fa',        # Light gray
    'dark': '#343a40',         # Dark gray
    'price': '#2c3e50',        # Dark blue for price
    'volume': '#3498db',       # Blue for volume
    'rsi': '#9b59b6',          # Purple for RSI
    'macd': '#e74c3c',         # Red for MACD
    'bollinger': '#34495e',    # Dark gray for Bollinger
    'sma': '#f39c12',          # Orange for SMA
    'ema': '#e67e22',          # Dark orange for EMA
    'stoch': '#8e44ad',        # Purple for Stochastic
    'adx': '#16a085',          # Teal for ADX
    'cci': '#27ae60',          # Green for CCI
    'mfi': '#2980b9',          # Blue for MFI
    'williams': '#c0392b',     # Dark red for Williams %R
    'roc': '#8e44ad',          # Purple for ROC
    'obv': '#f1c40f',          # Yellow for OBV
    'atr': '#e67e22',          # Orange for ATR
}

# Chart configuration
CHART_CONFIG = {
    'figure_size': (18, 14),
    'dpi': 150,
    'grid_alpha': 0.3,
    'line_width': 1.2,
    'font_size': 10,
    'title_font_size': 16,
    'legend_font_size': 9,
    'tick_font_size': 8,
    'subplot_spacing': 0.4,
    'date_format': '%m/%d',
    'date_interval': 2,
}


def generate_chart(ticker: str, df: pd.DataFrame) -> bytes:
    """
    Generate a comprehensive technical analysis chart for a ticker.

    Args:
        ticker: Ticker symbol
        df: DataFrame with OHLCV data and technical indicators

    Returns:
        Chart image as bytes
    """
    try:
        if df is None or df.empty:
            return _generate_error_chart(ticker, "No data available")

        # Extract available indicators
        indicators = _extract_available_indicators(df)

        # Create chart layout
        if len(indicators) > 0:
            fig, axes = _create_dynamic_layout(indicators)
        else:
            fig, axes = _create_comprehensive_layout()

        # Plot price chart with indicators
        _plot_comprehensive_price_chart(axes[0], df, indicators, ticker)

        # Plot volume chart
        if len(axes) > 1:
            _plot_volume_obv(axes[1], df, indicators, ticker)

        # Plot additional indicators in remaining subplots
        current_axis = 2
        for indicator_name in ['rsi', 'macd', 'stoch_k', 'adx', 'cci', 'mfi', 'williams_r', 'roc']:
            if current_axis < len(axes) and indicator_name in indicators:
                _plot_indicator_subplot(axes[current_axis], df, indicator_name, indicators[indicator_name], ticker)
                current_axis += 1

        # Set title
        fig.suptitle(f'{ticker} Technical Analysis', fontsize=16, fontweight='bold')

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=CHART_CONFIG['dpi'],
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        img_buffer.seek(0)

        plt.close()

        chart_bytes = img_buffer.getvalue()
        _logger.info(f"Chart generated successfully - Size: {len(chart_bytes)} bytes")
        return chart_bytes

    except Exception as e:
        _logger.exception("Failed to generate chart for %s: %s", ticker, str(e))
        return _generate_error_chart(ticker, f"{type(e).__name__}: {str(e)}")


def _extract_available_indicators(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract all available technical indicators from DataFrame."""
    indicators = {}

    # List of all possible indicators
    possible_indicators = [
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d',
        'adx', 'plus_di', 'minus_di', 'obv', 'adr', 'sma_50', 'sma_200',
        'ema_12', 'ema_26', 'cci', 'roc', 'mfi', 'williams_r', 'atr',
        'bb_upper', 'bb_middle', 'bb_lower'
    ]

    for indicator in possible_indicators:
        if indicator in df.columns:
            indicator_data = df[indicator].values
            # Check if the indicator has valid data (not all NaN)
            if indicator_data is not None and len(indicator_data) > 0:
                # Count non-NaN values
                valid_count = np.sum(~np.isnan(indicator_data))
                total_count = len(indicator_data)

                if valid_count > 0:
                    indicators[indicator] = indicator_data
                else:
                    _logger.warning(f"✗ {indicator}: All values are NaN")
            else:
                _logger.warning(f"✗ {indicator}: No data available")
        else:
            _logger.debug(f"✗ {indicator}: Column not found in DataFrame")

    return indicators


def _create_comprehensive_layout() -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create figure with fixed 6-subplot comprehensive layout."""
    # Create subplot layout: 6 fixed subplots
    total_subplots = 6

    # Calculate height ratios: Price chart is larger, others are equal
    height_ratios = [3, 1, 1, 1, 1, 1]  # Price chart is larger

    fig = plt.figure(figsize=CHART_CONFIG['figure_size'])
    gs = GridSpec(total_subplots, 1, height_ratios=height_ratios,
                 hspace=CHART_CONFIG['subplot_spacing'])

    axes = []
    for i in range(total_subplots):
        axes.append(fig.add_subplot(gs[i]))

    return fig, axes


def _create_dynamic_layout(indicators: Dict[str, np.ndarray]) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create figure with dynamic subplot layout based on available indicators."""
    # Count indicators that need separate subplots
    separate_indicators = ['rsi', 'macd', 'stoch_k', 'adx', 'cci', 'mfi', 'williams_r', 'roc']
    num_separate = sum(1 for ind in separate_indicators if ind in indicators)

    # Create subplot layout: price chart + separate indicators + volume
    total_subplots = 2 + num_separate  # Price + Volume + Separate indicators

    # Calculate height ratios
    height_ratios = [3] + [1] * (total_subplots - 1)  # Price chart is larger

    fig = plt.figure(figsize=CHART_CONFIG['figure_size'])
    gs = GridSpec(total_subplots, 1, height_ratios=height_ratios,
                 hspace=CHART_CONFIG['subplot_spacing'])

    axes = []
    for i in range(total_subplots):
        axes.append(fig.add_subplot(gs[i]))

    return fig, axes


def _plot_comprehensive_price_chart(ax: plt.Axes, df: pd.DataFrame, indicators: Dict[str, np.ndarray], ticker: str):
    """Plot comprehensive main price chart with Bollinger Bands, SMA 50, SMA 200."""

    # Extract data from DataFrame
    dates = df['timestamp'] if 'timestamp' in df else df.index
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_price = df['open'].values

    # Plot candlesticks
    _plot_candlesticks(ax, dates, open_price, high, low, close)

    # Add Bollinger Bands
    bb_indicators = ['bb_upper', 'bb_middle', 'bb_lower']
    if all(ind in indicators for ind in bb_indicators):
        _plot_bollinger_bands(ax, dates, indicators)
    else:
        missing_bb = [ind for ind in bb_indicators if ind not in indicators]
        _logger.warning(f"✗ Missing Bollinger Bands indicators: {missing_bb}")

    # Add moving averages
    if 'sma_50' in indicators:
        ax.plot(dates, indicators['sma_50'], label='SMA 50',
               color=CHART_COLORS['sma'], linewidth=CHART_CONFIG['line_width'], alpha=0.8)
    else:
        _logger.warning("✗ SMA 50 not found in indicators")

    if 'sma_200' in indicators:
        ax.plot(dates, indicators['sma_200'], label='SMA 200',
               color=CHART_COLORS['dark'], linewidth=CHART_CONFIG['line_width'], alpha=0.8)
    else:
        _logger.warning("✗ SMA 200 not found in indicators")

    ax.set_title(f'{ticker} - Price with Bollinger Bands & Moving Averages',
                fontsize=CHART_CONFIG['title_font_size'], fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])


def _plot_comprehensive_rsi(ax: plt.Axes, dates: pd.Series, indicators: Dict[str, np.ndarray], ticker: str):
    """Plot RSI with 30/70 oversold/overbought lines."""
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        ax.plot(dates, rsi, label='RSI', color=CHART_COLORS['rsi'], linewidth=CHART_CONFIG['line_width'])
        ax.axhline(y=70, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.axhline(y=50, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
        ax.set_ylabel('RSI', fontsize=CHART_CONFIG['font_size'])
        ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
        ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'RSI data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('RSI', fontsize=CHART_CONFIG['font_size'])

    ax.set_title('RSI (Relative Strength Index)', fontsize=CHART_CONFIG['font_size'], fontweight='bold')


def _plot_comprehensive_macd(ax: plt.Axes, dates: pd.Series, indicators: Dict[str, np.ndarray], ticker: str):
    """Plot MACD (MACD, Signal, Histogram)."""
    if 'macd' in indicators and 'macd_signal' in indicators and 'macd_hist' in indicators:
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_hist']

        ax.plot(dates, macd, label='MACD', color=CHART_COLORS['macd'], linewidth=CHART_CONFIG['line_width'])
        ax.plot(dates, macd_signal, label='Signal', color=CHART_COLORS['primary'], linewidth=CHART_CONFIG['line_width'])
        ax.bar(dates, macd_hist, label='Histogram', color=CHART_COLORS['light'], alpha=0.5, width=0.8)
        ax.axhline(y=0, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
        ax.set_ylabel('MACD', fontsize=CHART_CONFIG['font_size'])
        ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
        ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    else:
        ax.text(0.5, 0.5, 'MACD data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('MACD', fontsize=CHART_CONFIG['font_size'])

    ax.set_title('MACD (Moving Average Convergence Divergence)', fontsize=CHART_CONFIG['font_size'], fontweight='bold')


def _plot_comprehensive_stochastic(ax: plt.Axes, dates: pd.Series, indicators: Dict[str, np.ndarray], ticker: str):
    """Plot Stochastic oscillator."""
    if 'stoch_k' in indicators and 'stoch_d' in indicators:
        stoch_k = indicators['stoch_k']
        stoch_d = indicators['stoch_d']

        ax.plot(dates, stoch_k, label='%K', color=CHART_COLORS['stoch'], linewidth=CHART_CONFIG['line_width'])
        ax.plot(dates, stoch_d, label='%D', color=CHART_COLORS['secondary'], linewidth=CHART_CONFIG['line_width'])
        ax.axhline(y=80, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought (80)')
        ax.axhline(y=20, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold (20)')
        ax.set_ylabel('Stoch', fontsize=CHART_CONFIG['font_size'])
        ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
        ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'Stochastic data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Stoch', fontsize=CHART_CONFIG['font_size'])

    ax.set_title('Stochastic Oscillator', fontsize=CHART_CONFIG['font_size'], fontweight='bold')


def _plot_comprehensive_adx(ax: plt.Axes, dates: pd.Series, indicators: Dict[str, np.ndarray], ticker: str):
    """Plot ADX (ADX, +DI, -DI, Trend Threshold)."""
    if 'adx' in indicators and 'plus_di' in indicators and 'minus_di' in indicators:
        adx = indicators['adx']
        plus_di = indicators['plus_di']
        minus_di = indicators['minus_di']

        ax.plot(dates, adx, label='ADX', color=CHART_COLORS['adx'], linewidth=CHART_CONFIG['line_width'])
        ax.plot(dates, plus_di, label='+DI', color=CHART_COLORS['success'], linewidth=CHART_CONFIG['line_width'])
        ax.plot(dates, minus_di, label='-DI', color=CHART_COLORS['danger'], linewidth=CHART_CONFIG['line_width'])
        ax.axhline(y=25, color=CHART_COLORS['dark'], linestyle='--', alpha=0.7, label='Trend Threshold (25)')
        ax.set_ylabel('ADX', fontsize=CHART_CONFIG['font_size'])
        ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
        ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'ADX data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('ADX', fontsize=CHART_CONFIG['font_size'])

    ax.set_title('ADX (Average Directional Index)', fontsize=CHART_CONFIG['font_size'], fontweight='bold')


def _plot_comprehensive_volume(ax: plt.Axes, dates: pd.Series, volume: np.ndarray,
                              indicators: Dict[str, np.ndarray], ticker: str):
    """Plot Volume (OBV and volume histogram)."""


    # Plot volume histogram
    ax.bar(dates, volume, label='Volume', color=CHART_COLORS['volume'], alpha=0.3, width=0.8)

    # Plot OBV if available
    if 'obv' in indicators:
        obv = indicators['obv']
        # Normalize OBV to fit on the same scale as volume
        obv_normalized = (obv - np.min(obv)) / (np.max(obv) - np.min(obv)) * np.max(volume)
        ax.plot(dates, obv_normalized, label='OBV (Normalized)', color=CHART_COLORS['obv'],
               linewidth=CHART_CONFIG['line_width'])
    else:
        _logger.warning("✗ OBV not found in indicators")

    ax.set_ylabel('Volume', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_title('Volume & OBV', fontsize=CHART_CONFIG['font_size'], fontweight='bold')


def _plot_enhanced_price_chart(ax: plt.Axes, dates: pd.Series, high: np.ndarray,
                             low: np.ndarray, close: np.ndarray, open_price: np.ndarray,
                             indicators: Dict[str, np.ndarray], ticker: str):
    """Plot enhanced main price chart with candlesticks and overlays."""
    # Plot candlesticks
    _plot_candlesticks(ax, dates, open_price, high, low, close)

    # Add Bollinger Bands
    if all(ind in indicators for ind in ['bb_upper', 'bb_middle', 'bb_lower']):
        _plot_bollinger_bands(ax, dates, indicators)

    # Add moving averages
    if 'sma_50' in indicators:
        ax.plot(dates, indicators['sma_50'], label='SMA 50',
               color=CHART_COLORS['sma'], linewidth=CHART_CONFIG['line_width'], alpha=0.8)
    if 'sma_200' in indicators:
        ax.plot(dates, indicators['sma_200'], label='SMA 200',
               color=CHART_COLORS['dark'], linewidth=CHART_CONFIG['line_width'], alpha=0.8)
    if 'ema_12' in indicators:
        ax.plot(dates, indicators['ema_12'], label='EMA 12',
               color=CHART_COLORS['ema'], linewidth=CHART_CONFIG['line_width'], alpha=0.7)
    if 'ema_26' in indicators:
        ax.plot(dates, indicators['ema_26'], label='EMA 26',
               color=CHART_COLORS['secondary'], linewidth=CHART_CONFIG['line_width'], alpha=0.7)

    # Add ATR bands if available
    if 'atr' in indicators:
        _plot_atr_bands(ax, dates, close, indicators['atr'])

    ax.set_title(f'{ticker} - Enhanced Technical Analysis',
                fontsize=CHART_CONFIG['title_font_size'], fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])


def _plot_candlesticks(ax: plt.Axes, dates: pd.Series, open_price: np.ndarray,
                      high: np.ndarray, low: np.ndarray, close: np.ndarray):
    """Plot candlestick chart."""
    # For simplicity, use line plot for now - can be enhanced with proper candlesticks later
    ax.plot(dates, close, label='Close Price', color=CHART_COLORS['price'],
           linewidth=CHART_CONFIG['line_width'])

    # Add high/low range as shaded area
    ax.fill_between(dates, low, high, alpha=0.1, color=CHART_COLORS['light'])


def _plot_bollinger_bands(ax: plt.Axes, dates: pd.Series, indicators: Dict[str, np.ndarray]):
    """Plot Bollinger Bands."""
    bb_upper = indicators['bb_upper']
    bb_middle = indicators['bb_middle']
    bb_lower = indicators['bb_lower']

    ax.plot(dates, bb_upper, label='BB Upper',
           color=CHART_COLORS['bollinger'], linewidth=CHART_CONFIG['line_width'], alpha=0.7)
    ax.plot(dates, bb_middle, label='BB Middle',
           color=CHART_COLORS['primary'], linewidth=CHART_CONFIG['line_width'], alpha=0.7)
    ax.plot(dates, bb_lower, label='BB Lower',
           color=CHART_COLORS['bollinger'], linewidth=CHART_CONFIG['line_width'], alpha=0.7)

    # Fill Bollinger Bands
    ax.fill_between(dates, bb_upper, bb_lower, alpha=0.1, color=CHART_COLORS['light'])


def _plot_atr_bands(ax: plt.Axes, dates: pd.Series, close: np.ndarray, atr: np.ndarray):
    """Plot ATR bands around the price."""
    # Calculate ATR bands (using SMA of close as center)
    sma_close = np.convolve(close, np.ones(20)/20, mode='valid')
    atr_valid = atr[-len(sma_close):]

    upper_band = sma_close + (atr_valid * 2)
    lower_band = sma_close - (atr_valid * 2)

    dates_valid = dates[-len(sma_close):]

    ax.plot(dates_valid, upper_band, label='ATR Upper',
           color=CHART_COLORS['atr'], linewidth=CHART_CONFIG['line_width'], alpha=0.5, linestyle='--')
    ax.plot(dates_valid, lower_band, label='ATR Lower',
           color=CHART_COLORS['atr'], linewidth=CHART_CONFIG['line_width'], alpha=0.5, linestyle='--')


def _plot_indicator_subplot(ax: plt.Axes, dates: pd.Series, indicator_name: str,
                           indicator_data: np.ndarray, ticker: str):
    """Plot individual technical indicator."""
    if indicator_name == 'rsi':
        _plot_rsi(ax, dates, indicator_data)
    elif indicator_name == 'macd':
        _plot_macd(ax, dates, indicator_data, ticker)
    elif indicator_name == 'stoch_k':
        _plot_stochastic(ax, dates, indicator_data, ticker)
    elif indicator_name == 'adx':
        _plot_adx(ax, dates, indicator_data, ticker)
    elif indicator_name == 'cci':
        _plot_cci(ax, dates, indicator_data)
    elif indicator_name == 'mfi':
        _plot_mfi(ax, dates, indicator_data)
    elif indicator_name == 'williams_r':
        _plot_williams_r(ax, dates, indicator_data)
    elif indicator_name == 'roc':
        _plot_roc(ax, dates, indicator_data)
    elif indicator_name == 'obv':
        _plot_volume_obv(ax, dates, indicator_data, ticker)


def _plot_rsi(ax: plt.Axes, dates: pd.Series, rsi: np.ndarray):
    """Plot RSI indicator."""
    ax.plot(dates, rsi, label='RSI', color=CHART_COLORS['rsi'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=70, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought')
    ax.axhline(y=30, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold')
    ax.axhline(y=50, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
    ax.set_ylabel('RSI', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_ylim(0, 100)


def _plot_macd(ax: plt.Axes, dates: pd.Series, macd: np.ndarray, ticker: str):
    """Plot MACD indicator."""
    ax.plot(dates, macd, label='MACD', color=CHART_COLORS['macd'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=0, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
    ax.set_ylabel('MACD', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])


def _plot_stochastic(ax: plt.Axes, dates: pd.Series, stoch_k: np.ndarray, ticker: str):
    """Plot Stochastic indicator."""
    ax.plot(dates, stoch_k, label='%K', color=CHART_COLORS['stoch'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=80, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought')
    ax.axhline(y=20, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold')
    ax.set_ylabel('Stoch', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_ylim(0, 100)


def _plot_adx(ax: plt.Axes, dates: pd.Series, adx: np.ndarray, ticker: str):
    """Plot ADX indicator."""
    ax.plot(dates, adx, label='ADX', color=CHART_COLORS['adx'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=25, color=CHART_COLORS['dark'], linestyle='--', alpha=0.7, label='Trend Threshold')
    ax.set_ylabel('ADX', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_ylim(0, 100)


def _plot_cci(ax: plt.Axes, dates: pd.Series, cci: np.ndarray):
    """Plot CCI indicator."""
    ax.plot(dates, cci, label='CCI', color=CHART_COLORS['cci'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=100, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought')
    ax.axhline(y=-100, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold')
    ax.axhline(y=0, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
    ax.set_ylabel('CCI', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])


def _plot_mfi(ax: plt.Axes, dates: pd.Series, mfi: np.ndarray):
    """Plot MFI indicator."""
    ax.plot(dates, mfi, label='MFI', color=CHART_COLORS['mfi'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=80, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought')
    ax.axhline(y=20, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold')
    ax.axhline(y=50, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
    ax.set_ylabel('MFI', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_ylim(0, 100)


def _plot_williams_r(ax: plt.Axes, dates: pd.Series, williams_r: np.ndarray):
    """Plot Williams %R indicator."""
    ax.plot(dates, williams_r, label='Williams %R', color=CHART_COLORS['williams'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=-20, color=CHART_COLORS['danger'], linestyle='--', alpha=0.7, label='Overbought')
    ax.axhline(y=-80, color=CHART_COLORS['success'], linestyle='--', alpha=0.7, label='Oversold')
    ax.axhline(y=-50, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
    ax.set_ylabel('Williams %R', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_ylim(-100, 0)


def _plot_roc(ax: plt.Axes, dates: pd.Series, roc: np.ndarray):
    """Plot ROC indicator."""
    ax.plot(dates, roc, label='ROC', color=CHART_COLORS['roc'], linewidth=CHART_CONFIG['line_width'])
    ax.axhline(y=0, color=CHART_COLORS['dark'], linestyle='-', alpha=0.5)
    ax.set_ylabel('ROC', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])


def _plot_volume_obv(ax: plt.Axes, df: pd.DataFrame, indicators: Dict[str, np.ndarray], ticker: str):
    """Plot Volume and OBV indicator."""
    # Extract data from DataFrame
    dates = df['timestamp'] if 'timestamp' in df else df.index
    volume = df['volume'].values

    # Plot volume histogram
    ax.bar(dates, volume, label='Volume', color=CHART_COLORS['volume'], alpha=0.3, width=0.8)

    # Plot OBV if available
    if 'obv' in indicators:
        obv = indicators['obv']
        # Normalize OBV to fit on the same scale as volume
        obv_normalized = (obv - np.min(obv)) / (np.max(obv) - np.min(obv)) * np.max(volume)
        ax.plot(dates, obv_normalized, label='OBV (Normalized)', color=CHART_COLORS['obv'],
               linewidth=CHART_CONFIG['line_width'])
    else:
        _logger.warning("✗ OBV not found in indicators")

    ax.set_ylabel('Volume', fontsize=CHART_CONFIG['font_size'])
    ax.legend(loc='upper left', fontsize=CHART_CONFIG['legend_font_size'])
    ax.grid(True, alpha=CHART_CONFIG['grid_alpha'])
    ax.set_title('Volume & OBV', fontsize=CHART_CONFIG['font_size'], fontweight='bold')


def _format_all_axes(axes: List[plt.Axes], dates: pd.Series):
    """Format all axes with consistent styling."""
    for ax in axes:
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter(CHART_CONFIG['date_format']))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=CHART_CONFIG['date_interval']))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=CHART_CONFIG['tick_font_size'])

        # Format y-axis
        ax.tick_params(axis='y', labelsize=CHART_CONFIG['tick_font_size'])

        # Set x-label for bottom subplot only
        if ax == axes[-1]:
            ax.set_xlabel('Date', fontsize=CHART_CONFIG['font_size'])


def _add_enhanced_current_values_box(ax: plt.Axes, close: np.ndarray, indicators: Dict[str, np.ndarray]):
    """Add enhanced text box with current indicator values."""
    current_idx = -1
    current_close = close[current_idx]

    # Build text string with current values
    text_lines = [f'Close: ${current_close:.2f}']

    # Add current values for key indicators
    key_indicators = ['rsi', 'macd', 'adx', 'cci', 'mfi']
    for indicator in key_indicators:
        if indicator in indicators:
            value = indicators[indicator][current_idx]
            if not np.isnan(value):
                if indicator == 'rsi':
                    text_lines.append(f'RSI: {value:.1f}')
                elif indicator == 'macd':
                    text_lines.append(f'MACD: {value:.4f}')
                elif indicator == 'adx':
                    text_lines.append(f'ADX: {value:.1f}')
                elif indicator == 'cci':
                    text_lines.append(f'CCI: {value:.1f}')
                elif indicator == 'mfi':
                    text_lines.append(f'MFI: {value:.1f}')

    textstr = '\n'.join(text_lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)


def _generate_error_chart(ticker: str, error_msg: str) -> bytes:
    """Generate error chart when chart generation fails."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'Error generating chart for {ticker}\n{error_msg}',
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title(f'Chart Error - {ticker}')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=CHART_CONFIG['dpi'], bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()

    return img_buffer.getvalue()
