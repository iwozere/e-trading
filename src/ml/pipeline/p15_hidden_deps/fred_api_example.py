Секторные ETF как аналог MOEXFN/MOEXOG:
XLF  — финансы
XLE  — энергетика
XLK  — технологии
XLV  — здравоохранение
XLI  — промышленность
XLB  — материалы
XLU  — utilities
XLP  — consumer staples

********************************************************************
Макро-факторы (через yfinance или отдельно):
CL=F   — нефть WTI
BZ=F   — Brent
GC=F   — золото
DX-Y.NYB — DXY (индекс доллара)
^TNX   — 10-летние трежерис (доходность)
^VIX   — волатильность
^GSPC  — S&P 500
EURUSD=X, USDJPY=X — курсы

********************************************************************

********************************************************************
Каждый день (утром, до открытия рынка):
  → обновить daily-серии (T10Y2Y, HY spread, breakevens...)
  → пересобрать fred_combined.parquet

Каждый четверг:
  → обновить ICSA (jobless claims выходят в чт 8:30 EST) --> 14:00 UTC
  → обновить WALCL (баланс ФРС)

Первые числа месяца (по расписанию BLS/BEA):
  → обновить CPI, PCE, PAYEMS, UNRATE, M2SL



import os
import json
import requests
import pandas as pd
from datetime import date, datetime
from pathlib import Path

FRED_API_KEY = "YOUR_KEY"
CACHE_DIR = Path("data/fred/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = Path("data/fred/fred_meta.json")

FRED_SERIES = {
    "FEDFUNDS":       {"name": "fed_funds_rate",    "freq": "monthly"},
    "DFF":            {"name": "fed_funds_daily",   "freq": "daily"},
    "T10Y2Y":         {"name": "yield_spread_10_2", "freq": "daily"},
    "T10Y3M":         {"name": "yield_spread_10_3m","freq": "daily"},
    "DGS2":           {"name": "yield_2y",          "freq": "daily"},
    "DGS10":          {"name": "yield_10y",         "freq": "daily"},
    "CPIAUCSL":       {"name": "cpi",               "freq": "monthly"},
    "CPILFESL":       {"name": "core_cpi",          "freq": "monthly"},
    "PCEPILFE":       {"name": "core_pce",          "freq": "monthly"},
    "PPIACO":         {"name": "ppi",               "freq": "monthly"},
    "T5YIE":          {"name": "breakeven_5y",      "freq": "daily"},
    "T10YIE":         {"name": "breakeven_10y",     "freq": "daily"},
    "UNRATE":         {"name": "unemployment",      "freq": "monthly"},
    "PAYEMS":         {"name": "nonfarm_payrolls",  "freq": "monthly"},
    "ICSA":           {"name": "jobless_claims",    "freq": "weekly"},
    "BAMLH0A0HYM2":   {"name": "hy_spread",         "freq": "daily"},
    "BAMLC0A0CM":     {"name": "ig_spread",         "freq": "daily"},
    "DRTSCILM":       {"name": "loan_standards",    "freq": "quarterly"},
    "WALCL":          {"name": "fed_balance_sheet", "freq": "weekly"},
    "M2SL":           {"name": "m2",               "freq": "monthly"},
    "MORTGAGE30US":   {"name": "mortgage_rate_30y", "freq": "weekly"},
    "UMCSENT":        {"name": "consumer_sentiment","freq": "monthly"},
    "INDPRO":         {"name": "industrial_prod",   "freq": "monthly"},
    "USREC":          {"name": "recession_flag",    "freq": "monthly"},
}

def load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {}

def save_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, indent=2))

def fetch_series(series_id: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    """Fetch full series from FRED API."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start_date,
        "observation_end":   date.today().isoformat(),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    observations = r.json()["observations"]
    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    # FRED uses "." for missing values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).rename(columns={"value": series_id})
    df = df.set_index("date").sort_index()
    return df

def update_series(series_id: str, meta: dict) -> pd.DataFrame:
    """
    Incremental update: fetch only new observations since last download.
    Falls back to full download if no cache exists.
    """
    cache_path = CACHE_DIR / f"{series_id}.parquet"

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        last_date = existing.index.max()
        # Fetch only from day after last known observation
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new_data = fetch_series(series_id, start_date=start)

        if new_data.empty:
            print(f"  {series_id}: up to date ({last_date.date()})")
            return existing

        combined = pd.concat([existing, new_data]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        print(f"  {series_id}: full download from 2010-01-01")
        combined = fetch_series(series_id, start_date="2010-01-01")

    combined.to_parquet(cache_path)
    meta[series_id] = {
        "last_updated": datetime.now().isoformat(),
        "last_observation": combined.index.max().isoformat(),
        "rows": len(combined),
    }
    print(f"  {series_id}: saved {len(combined)} rows → {cache_path.name}")
    return combined

def build_combined(series_dict: dict) -> pd.DataFrame:
    """
    Merge all raw parquet files into one wide DataFrame,
    forward-filled to daily frequency.
    """
    frames = []
    for series_id, info in series_dict.items():
        path = CACHE_DIR / f"{series_id}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.columns = [info["name"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Outer join on date index → sparse (monthly series have NaN on most days)
    combined = pd.concat(frames, axis=1).sort_index()

    # Reindex to full daily calendar
    daily_index = pd.date_range(
        start=combined.index.min(),
        end=combined.index.max(),
        freq="D"
    )
    combined = combined.reindex(daily_index)

    # Forward-fill: monthly CPI value persists until next release
    combined = combined.ffill()

    combined.index.name = "date"
    return combined

def update_all(force_full: bool = False):
    meta = {} if force_full else load_meta()

    for series_id in FRED_SERIES:
        try:
            update_series(series_id, meta)
        except Exception as e:
            print(f"  ✗ {series_id}: {e}")

    save_meta(meta)

    print("\nBuilding combined wide file...")
    combined = build_combined(FRED_SERIES)
    combined.to_parquet("data/fred/fred_combined.parquet")
    print(f"Combined: {combined.shape[0]} days × {combined.shape[1]} columns")
    return combined

if __name__ == "__main__":
    update_all()