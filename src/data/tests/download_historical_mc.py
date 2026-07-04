from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_btc_mc():
    print("Downloading BTC-USD Price History as Market Cap Proxy...")
    ticker = "BTC-USD"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print("Failed to download data.")
        return

    # We use Close as 'btc_mc' for the regime model
    # Log-returns of Price are almost identical to Log-returns of Market Cap
    df_mc = pd.DataFrame(index=df.index)
    df_mc["timestamp"] = df.index
    df_mc["btc_mc"] = df["Close"]

    # Fix timestamp format to match what data_loader.py expects (no timezone)
    df_mc["timestamp"] = df_mc.index.tz_localize(None)

    output_path = Path("data/btc_mc/btc_mc.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_mc.to_csv(output_path, index=False)
    print(f"Historical data saved to {output_path}")


if __name__ == "__main__":
    download_btc_mc()
