import numpy as np
import yfinance as yf
from src.notification.logger import setup_logger

logger = setup_logger("telegram_bot")


def calculate_technicals(ticker: str) -> dict:
    try:
        df = yf.download(ticker, period="6mo", interval="1d")

        if df.empty:
            logger.error(f"No data downloaded for ticker {ticker}")
            return {}

        logger.debug(f"Downloaded {len(df)} days of data for {ticker}")

        df.dropna(inplace=True)

        # SMA
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df["BB_Middle"] = df["Close"].rolling(window=bb_period).mean()
        df["BB_Std"] = df["Close"].rolling(window=bb_period).std()
        df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * bb_std)
        df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * bb_std)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

        # RSI (14)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Get latest values
        last_close = df["Close"].iloc[-1]
        sma50 = df["SMA_50"].iloc[-1]
        sma200 = df["SMA_200"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        macd_signal = df["Signal"].iloc[-1]
        bb_upper = df["BB_Upper"].iloc[-1]
        bb_middle = df["BB_Middle"].iloc[-1]
        bb_lower = df["BB_Lower"].iloc[-1]
        bb_width = df["BB_Width"].iloc[-1]

        # Determine trend
        trend = (
            "Uptrend"
            if sma50 > sma200 and last_close > sma50
            else "Downtrend" if sma50 < sma200 and last_close < sma50 else "Sideways"
        )

        logger.debug(
            f"Calculated technicals for {ticker}: RSI={rsi:.2f}, Trend={trend}"
        )

        return {
            "rsi": round(rsi, 2),
            "sma_50": round(sma50, 2),
            "sma_200": round(sma200, 2),
            "macd_signal": round(macd_signal, 2),
            "trend": trend,
            "bb_upper": round(bb_upper, 2),
            "bb_middle": round(bb_middle, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_width": round(bb_width, 4),
        }

    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {str(e)}", exc_info=e)
        return {}
