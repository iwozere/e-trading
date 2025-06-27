# ticker_bot/analyzer/chart.py

import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yfinance as yf


def generate_price_chart(ticker: str) -> bytes:
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df.dropna(inplace=True)

        # Calculate indicators
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df["BB_Middle"] = df["Close"].rolling(window=bb_period).mean()
        df["BB_Std"] = df["Close"].rolling(window=bb_period).std()
        df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * bb_std)
        df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * bb_std)

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot price and moving averages
        ax.plot(df.index, df["Close"], label="Close Price", linewidth=2)
        ax.plot(df.index, df["SMA_50"], label="SMA 50", linestyle="--")
        ax.plot(df.index, df["SMA_200"], label="SMA 200", linestyle="--")

        # Plot Bollinger Bands
        ax.plot(df.index, df["BB_Upper"], label="BB Upper", color="red", alpha=0.5)
        ax.plot(df.index, df["BB_Middle"], label="BB Middle", color="green", alpha=0.5)
        ax.plot(df.index, df["BB_Lower"], label="BB Lower", color="red", alpha=0.5)

        # Fill Bollinger Bands
        ax.fill_between(
            df.index, df["BB_Upper"], df["BB_Lower"], color="gray", alpha=0.1
        )

        ax.set_title(f"{ticker} Price Chart with Indicators")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        # Date format on X axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()

        # Save to bytes
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        return buf.read()

    except Exception as e:
        print(f"[ERROR] Failed to generate chart for {ticker}: {e}")
        return buf.read()
