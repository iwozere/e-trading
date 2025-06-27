import datetime
from typing import Any, Dict, Optional

import pandas as pd
import ta
import yfinance as yf


class TickerAnalyzer:
    """
    Analyzes a ticker symbol and provides fundamental and technical analysis.

    This class uses yfinance to download stock data and ta to calculate technical indicators.
    """

    def analyze_ticker(ticker_symbol):
        try:
            # Download data
            ticker = yf.Ticker(ticker_symbol)

            # --- FUNDAMENTAL ANALYSIS ---
            info = ticker.info
            if not info or "longName" not in info:
                print(
                    f"\n❌ No fundamental data found for '{ticker_symbol}'. The ticker may be delisted or invalid."
                )
                return None, None
            fundamentals = {
                "Company": info.get("longName"),
                "Sector": info.get("sector"),
                "Market Cap": info.get("marketCap"),
                "P/E Ratio": info.get("trailingPE"),
                "EPS": info.get("trailingEps"),
                "Dividend Yield": info.get("dividendYield"),
                "52 Week High": info.get("fiftyTwoWeekHigh"),
                "52 Week Low": info.get("fiftyTwoWeekLow"),
                "Return on Equity": info.get("returnOnEquity"),
                "Return on Assets": info.get("returnOnAssets"),
                "Revenue": info.get("totalRevenue"),
                "Gross Profits": info.get("grossProfits"),
                "Net Income": info.get("netIncomeToCommon"),
            }

            # --- TECHNICAL ANALYSIS ---
            print("\n📈 Technical Indicators:")
            df = ticker.history(period="1y", interval="1d")
            if df is None or df.empty:
                print(
                    f"❌ No price data found for '{ticker_symbol}'. The ticker may be delisted or has no recent trading history."
                )
                return None, None

            # Clean and add indicators
            df = df.dropna()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["MACD"] = ta.trend.MACD(df["Close"]).macd()
            df["SMA_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
            df["EMA_20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
            bb = ta.volatility.BollingerBands(df["Close"])
            df["BB_High"] = bb.bollinger_hband()
            df["BB_Low"] = bb.bollinger_lband()

            latest = df.iloc[-1]
            print(f"Latest Close: {latest['Close']:.2f}")
            print(f"RSI: {latest['RSI']:.2f}")
            print(f"MACD: {latest['MACD']:.2f}")
            print(f"SMA 20: {latest['SMA_20']:.2f}")
            print(f"EMA 20: {latest['EMA_20']:.2f}")
            print(f"Bollinger High: {latest['BB_High']:.2f}")
            print(f"Bollinger Low: {latest['BB_Low']:.2f}")

            df.dropna(inplace=True)

            return fundamentals, df
        except Exception as e:
            print(f"\n❌ Error analyzing ticker '{ticker_symbol}': {e}")
            return None, None


# 🔍 Example usage
if __name__ == "__main__":
    ticker_input = input("Enter ticker symbol (e.g. AAPL): ").upper()
    fundamentals, df = TickerAnalyzer.analyze_ticker(ticker_input)
    if fundamentals and df is not None:
        print("\n📊 Fundamental Data:")
        for key, val in fundamentals.items():
            print(f"{key}: {val}")
        print("\n📈 Technical Indicators:")
        print(df)
    else:
        print("❌ Error analyzing ticker")
