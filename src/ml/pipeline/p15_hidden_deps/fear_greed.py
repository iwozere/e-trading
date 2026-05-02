import requests
import pandas as pd

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

def fetch_cnn_fear_greed(start_date: str = "2021-02-01") -> pd.DataFrame:
    r = requests.get(
        f"{CNN_URL}/{start_date}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )
    data = r.json()

    df = pd.DataFrame(data["fear_and_greed_historical"]["data"])
    # x — unix timestamp в миллисекундах
    df["date"] = pd.to_datetime(df["x"] // 1000, unit="s").dt.date
    df = df.rename(columns={"y": "fear_greed_score"})
    df["label"] = df["fear_greed_score"].apply(
        lambda v: "extreme_fear" if v < 25 else
                  "fear" if v < 45 else
                  "neutral" if v < 55 else
                  "greed" if v < 75 else "extreme_greed"
    )
    return df[["date", "fear_greed_score", "label"]].drop_duplicates("date")

def build_full_history(
    archive_url: str = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv",
    cache_path: str = "data/sentiment/cnn_fear_greed.parquet"
):
    # Скачать архив 2011–2021
    archive = pd.read_csv(archive_url, parse_dates=["date"])

    # Дополнить свежими данными с CNN
    fresh = fetch_cnn_fear_greed(start_date="2021-02-01")
    fresh["date"] = pd.to_datetime(fresh["date"])

    combined = pd.concat([archive, fresh]).drop_duplicates("date").sort_values("date")
    combined = combined.set_index("date")
    combined.to_parquet(cache_path)
    return combined