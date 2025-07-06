# ticker_bot/analyzer/chart.py

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.notification.logger import setup_logger
from src.model.telegram_bot import TickerAnalysis

logger = setup_logger("telegram_bot")


def generate_chart(analysis : TickerAnalysis) -> bytes:
    """
    Generate comprehensive chart for any ticker using a TickerAnalysis object.
    All technical indicators must already be present as columns in the DataFrame or in the technicals field.

    Args:
        analysis: TickerAnalysis object with fields: ticker, df (DataFrame), technicals (indicators)

    Returns:
        bytes: Chart image as bytes
    """
    try:
        df = analysis.ohlcv
        ticker = analysis.ticker
        if df is None or df.empty:
            raise ValueError(f"No data available for {ticker}")
        if len(df) < 50:
            raise ValueError(f"Insufficient data for {ticker}: {len(df)} days")

        # Use precomputed columns from df
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        open_price = df['open'].values.astype(float)
        # Use indicator arrays from df columns
        rsi = df['rsi'].values if 'rsi' in df else None
        bb_upper = df['bb_upper'].values if 'bb_upper' in df else None
        bb_middle = df['bb_middle'].values if 'bb_middle' in df else None
        bb_lower = df['bb_lower'].values if 'bb_lower' in df else None
        macd = df['macd'].values if 'macd' in df else None
        macd_signal = df['macd_signal'].values if 'macd_signal' in df else None
        macd_hist = df['macd_hist'].values if 'macd_hist' in df else None
        stoch_k = df['stoch_k'].values if 'stoch_k' in df else None
        stoch_d = df['stoch_d'].values if 'stoch_d' in df else None
        adx = df['adx'].values if 'adx' in df else None
        plus_di = df['plus_di'].values if 'plus_di' in df else None
        minus_di = df['minus_di'].values if 'minus_di' in df else None
        obv = df['obv'].values if 'obv' in df else None
        adr = df['adr'].values if 'adr' in df else None
        sma_50 = df['sma_50'].values if 'sma_50' in df else None
        sma_200 = df['sma_200'].values if 'sma_200' in df else None
        # Use timestamp for x-axis if present
        dates = df['timestamp'] if 'timestamp' in df else df.index
        min_length = len(df)

        # Create the chart
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.3)

        # Main price chart with Bollinger Bands and SMAs
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(dates, close, label='Close Price', color='black', linewidth=1)
        ax1.plot(dates, bb_upper, label='BB Upper', color='red', alpha=0.7, linewidth=0.8)
        ax1.plot(dates, bb_middle, label='BB Middle', color='blue', alpha=0.7, linewidth=0.8)
        ax1.plot(dates, bb_lower, label='BB Lower', color='red', alpha=0.7, linewidth=0.8)
        ax1.plot(dates, sma_50, label='SMA 50', color='orange', alpha=0.8, linewidth=1.2)
        ax1.plot(dates, sma_200, label='SMA 200', color='purple', alpha=0.8, linewidth=1.2)

        # Fill Bollinger Bands
        ax1.fill_between(dates, bb_upper, bb_lower, alpha=0.1, color='gray')

        ax1.set_title(f'{ticker} - Technical Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # RSI
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(dates, rsi, label='RSI', color='purple', linewidth=1)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # MACD
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(dates, macd, label='MACD', color='blue', linewidth=1)
        ax3.plot(dates, macd_signal, label='Signal', color='red', linewidth=1)
        ax3.bar(dates, macd_hist, label='Histogram', color='gray', alpha=0.6, width=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('MACD', fontsize=10)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Stochastic
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(dates, stoch_k, label='%K', color='blue', linewidth=1)
        ax4.plot(dates, stoch_d, label='%D', color='red', linewidth=1)
        ax4.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax4.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax4.set_ylabel('Stoch', fontsize=10)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)

        # ADX
        ax5 = fig.add_subplot(gs[4])
        ax5.plot(dates, adx, label='ADX', color='black', linewidth=1)
        ax5.plot(dates, plus_di, label='+DI', color='green', linewidth=1)
        ax5.plot(dates, minus_di, label='-DI', color='red', linewidth=1)
        ax5.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Trend Threshold')
        ax5.set_ylabel('ADX', fontsize=10)
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 100)

        # Volume and OBV
        ax6 = fig.add_subplot(gs[5])
        ax6_twin = ax6.twinx()

        # Volume bars
        ax6.bar(dates, volume, alpha=0.6, color='lightblue', label='Volume')
        ax6.set_ylabel('Volume', fontsize=10, color='blue')
        ax6.tick_params(axis='y', labelcolor='blue')

        # OBV line
        ax6_twin.plot(dates, obv, color='orange', linewidth=1, label='OBV')
        ax6_twin.set_ylabel('OBV', fontsize=10, color='orange')
        ax6_twin.tick_params(axis='y', labelcolor='orange')

        ax6.set_xlabel('Date', fontsize=12)
        ax6.grid(True, alpha=0.3)

        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Add current values as text
        current_idx = -1
        current_close = close[current_idx]
        current_rsi = rsi[current_idx] if not np.isnan(rsi[current_idx]) else 50.0
        current_macd = macd[current_idx] if not np.isnan(macd[current_idx]) else 0.0
        current_adx = adx[current_idx] if not np.isnan(adx[current_idx]) else 25.0

        # Add text box with current values
        textstr = f'Close: ${current_close:.2f}\nRSI: {current_rsi:.1f}\nMACD: {current_macd:.4f}\nADX: {current_adx:.1f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        return img_buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to generate chart for {ticker}: {str(e)}")
        # Return a simple error chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error generating chart for {ticker}\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Chart Error - {ticker}')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        return img_buffer.getvalue()
