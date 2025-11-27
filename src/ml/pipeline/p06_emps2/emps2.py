import os
import io
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================# 0. CONFIG# ============================================================
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
if FINNHUB_KEY is None:
    raise ValueError("Set FINNHUB_API_KEY environment variable")

# Filter parameters
MIN_PRICE = 1
MIN_AVG_VOLUME = 400_000
MIN_MARKET_CAP = 50_000_000  # $50M
MAX_MARKET_CAP = 5_000_000_000 # $5B
MAX_FLOAT = 60_000_000 # 60M shares
VOLATILITY_THRESHOLD = 0.02 # ATR/Price > 2%

# ============================================================# 1. GET FULL US UNIVERSE (NASDAQ + NYSE + AMEX)# ============================================================
def download_nasdaq_universe():
    print("Downloading NASDAQ Trader symbol list...")
    url = "https://ftp.nasdaqtrader.com/dynamic/SymbolLookup/nasdaqlisted.txt"
    other = "https://ftp.nasdaqtrader.com/dynamic/SymbolLookup/otherlisted.txt"
    df1 = pd.read_csv(io.StringIO(requests.get(url).text), sep="|")
    df2 = pd.read_csv(io.StringIO(requests.get(other).text), sep="|")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df["Test Issue"] == "N"]
    df = df[df["Symbol"].str.isalpha()] # remove weird tickers    df = df.rename(columns={"Symbol": "ticker"})
    print(f"Universe downloaded: {len(df)} tickers")
    return df["ticker"].unique().tolist()

# ============================================================# 2. GET FUNDAMENTALS IN BULK (FINNHUB)# ============================================================
def fetch_fundamentals_bulk(tickers):
    print("Fetching fundamentals (market cap, float, avg volume, sector)...")
    fundamentals = []
    for t in tickers:
        try:
            url = f"https://finnhub.io/api/v1/stock/profile2?symbol={t}&token={FINNHUB_KEY}"
            data = requests.get(url).json()
            if not data or "name" not in data:
                continue
            fundamentals.append({
                "ticker": t,
                "market_cap": data.get("marketCapitalization"),
                "sector": data.get("finnhubIndustry"),
                "float": data.get("shareOutstanding"),
                })
        except Exception:
            pass
    df = pd.DataFrame(fundamentals)
    print(f"Fundamentals fetched for {len(df)} tickers")
    return df

def fetch_volume_data_bulk(tickers):
    print("Fetching avg volume via Finnhub...")
    rows = []
    for t in tickers:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={t}&token={FINNHUB_KEY}"
            q = requests.get(url).json()
            rows.append({"ticker": t, "avg_volume": q.get("volume")})
        except:
            pass
    return pd.DataFrame(rows)

# ============================================================# 3. APPLY FUNDAMENTAL PREFILTER# ============================================================
def apply_fundamental_filter(df):
    print("Applying fundamental filters...")
    df = df.dropna(subset=["market_cap", "float"])
    df = df[
        (df["market_cap"] >= MIN_MARKET_CAP) &
        (df["market_cap"] <= MAX_MARKET_CAP) &
        (df["avg_volume"] >= MIN_AVG_VOLUME) &
        (df["float"] <= MAX_FLOAT)    ]
    print(f"After fundamentals: {len(df)} tickers")
    return df

# ============================================================# 4. DOWNLOAD 15-MIN PRICE DATA IN BULK AND APPLY ACTIVITY FILTER# ============================================================
def compute_ATR(df, period=14):
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift()).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift()).abs()
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def activity_filter(tickers):
    print("Downloading 2 days of 15m data... (bulk)")
    data = yf.download(
        tickers,
        interval="15m",
        period="7d",
        group_by="ticker",
        threads=True
        )
    good = []
    for t in tickers:
        try:
            df = data[t].dropna()
            if len(df) < 20:
                continue
            # Price filter
            last_price = df["Close"].iloc[-1]
            if last_price < MIN_PRICE:
                continue
            # ATR / Price filter
            atr = compute_ATR(df).iloc[-1]
            if atr / last_price < VOLATILITY_THRESHOLD:
                continue
            # Price range filter
            range_perc = (df["High"].max() - df["Low"].min()) / df["Low"].min()
            if range_perc < 0.05: # 5% range
                continue
            good.append(t)
        except Exception:            pass
    print(f"After activity/volatility filters: {len(good)} tickers")
    return good

# ============================================================# 5. MAIN PROCESS# ============================================================
def build_filtered_universe():
    # Step 1 — Universe list
    universe = download_nasdaq_universe()
    # Step 2 — Fundamentals (market cap, float, sector)
    df_f = fetch_fundamentals_bulk(universe)
    # Step 2b — Avg volume
    df_v = fetch_volume_data_bulk(df_f["ticker"].tolist())
    df = df_f.merge(df_v, on="ticker", how="left")
    # Step 3 — Apply fundamental filtering
    df = apply_fundamental_filter(df)
    # Step 4 — Apply activity filters with yfinance 15m bulk
    prefiltered = activity_filter(df["ticker"].tolist())
    # Build final dataframe
    final_df = df[df["ticker"].isin(prefiltered)]
    print("\nFINAL PREFILTERED UNIVERSE SIZE:", len(final_df))
    # Save result
    final_df.to_csv("prefiltered_universe.csv", index=False)
    print("Saved: prefiltered_universe.csv")
    return final_df

# ============================================================# RUN THE SCRIPT# ============================================================
if __name__ == "__main__":
    build_filtered_universe()