import requests
import pandas as pd
from pathlib import Path

def update_aaii(cache_path: str = "data/sentiment/aaii.parquet"):
    url = "https://www.aaii.com/files/surveys/sentiment.xls"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)

    # XLS содержит несколько листов, данные на листе "Sentiment Survey"
    df = pd.read_excel(
        r.content,
        sheet_name="Sentiment Survey",
        skiprows=3,   # шапка занимает 3 строки
        usecols="A:F"
    )
    df.columns = ["date", "bullish", "neutral", "bearish", "total", "bull_bear_spread"]
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])

    # Конвертируем проценты из формата 0.38 → 38.0
    for col in ["bullish", "neutral", "bearish"]:
        if df[col].max() < 2:   # значит дроби, не проценты
            df[col] = df[col] * 100

    df = df.sort_values("date").set_index("date")
    df.to_parquet(cache_path)
    print(f"AAII: {len(df)} недель, с {df.index.min().date()} по {df.index.max().date()}")
    return df