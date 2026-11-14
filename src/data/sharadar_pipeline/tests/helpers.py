"""
Synthetic Sharadar raw files for the tests. Every value is invented; nothing here is downloaded data.

Three securities:

- ``ABC`` (permaticker 101): active domestic common stock, CIK 111. Six bars; a 2-for-1 split on day 4 and a cash
  dividend with ex-date day 2 (0.50 split-adjusted = 1.00 raw).
- ``XYZ1`` (102): delisted, ticker reused (traded as ``XYZ``), CIK 222. Acquired by ABC on day 5 for 10.00 cash
  plus 0.1 ABC shares.
- ``ABC-PA`` (103): preferred stock of the same issuer.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.data.sharadar_pipeline import config

DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]

TICKERS = pd.DataFrame([
    {"table": "stocks", "permaticker": "101", "ticker": "ABC", "name": "ABC CORP", "exchange": "NYSE", "isdelisted": "N",
     "category": "Domestic Common Stock", "cusips": "000000101", "siccode": "3570", "figi": "BBG000000101",
     "sector": "Technology", "industry": "Hardware", "firstpricedate": "2024-01-02", "lastpricedate": "2024-01-09",
     "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000111"},
    {"table": "stocks", "permaticker": "102", "ticker": "XYZ1", "name": "XYZ INC", "exchange": "NASDAQ", "isdelisted": "Y",
     "category": "Domestic Common Stock", "cusips": "000000102", "siccode": "2834", "figi": "",
     "sector": "Healthcare", "industry": "Biotechnology", "firstpricedate": "2024-01-02", "lastpricedate": "2024-01-08",
     "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000222"},
    {"table": "stocks", "permaticker": "103", "ticker": "ABC-PA", "name": "ABC CORP", "exchange": "NYSE", "isdelisted": "N",
     "category": "Domestic Preferred Stock", "cusips": "", "siccode": "3570", "figi": "",
     "sector": "Technology", "industry": "Hardware", "firstpricedate": "2024-01-02", "lastpricedate": "2024-01-09",
     "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000111"},
    {"table": "fundamentals", "permaticker": "101", "ticker": "ABC", "name": "ABC CORP", "exchange": "NYSE", "isdelisted": "N",
     "category": "Domestic Common Stock", "cusips": "000000101", "siccode": "3570", "figi": "BBG000000101",
     "sector": "Technology", "industry": "Hardware", "firstpricedate": "2024-01-02", "lastpricedate": "2024-01-09",
     "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000111"},
])

#: ABC raw closes 100, 102, 104 then 53, 54, 55 after the 2-for-1 split; Sharadar delivers split-adjusted OHLCV.
#: closeadj: day 1 carries the dividend factor 1 - 1.00/100 = 0.99 -> 50 * 0.99 = 49.5.
_ABC = [("2024-01-02", 50.0, 49.5, 100.0, 2000.0), ("2024-01-03", 51.0, 51.0, 102.0, 2000.0),
        ("2024-01-04", 52.0, 52.0, 104.0, 2000.0), ("2024-01-05", 53.0, 53.0, 53.0, 1000.0),
        ("2024-01-08", 54.0, 54.0, 54.0, 1000.0), ("2024-01-09", 55.0, 55.0, 55.0, 1000.0)]
_XYZ = [("2024-01-02", 12.0), ("2024-01-03", 12.5), ("2024-01-04", 13.0), ("2024-01-05", 13.5), ("2024-01-08", 14.0)]


def stocks_frame() -> pd.DataFrame:
    """Sharadar ``stocks`` rows of the three securities."""
    rows = [{"ticker": "ABC", "date": day, "open": close - 1, "high": close + 1, "low": close - 2, "close": close,
             "volume": volume, "closeadj": closeadj, "closeunadj": unadj, "lastupdated": "2024-01-09"}
            for day, close, closeadj, unadj, volume in _ABC]
    rows += [{"ticker": "XYZ1", "date": day, "open": close, "high": close, "low": close, "close": close, "volume": 500.0,
              "closeadj": close, "closeunadj": close, "lastupdated": "2024-01-09"} for day, close in _XYZ]
    rows += [{"ticker": "ABC-PA", "date": day, "open": 25.0, "high": 25.0, "low": 25.0, "close": 25.0, "volume": 10.0,
              "closeadj": 25.0, "closeunadj": 25.0, "lastupdated": "2024-01-09"} for day in DAYS]
    return pd.DataFrame(rows)


def actions_frame() -> pd.DataFrame:
    """Sharadar ``actions`` rows: split, dividend, acquisition with cash and stock, ticker change."""
    rows = [
        ("2024-01-05", "split", "ABC", "ABC CORP", 2.0, "N/A", "N/A"),
        ("2024-01-03", "dividend", "ABC", "ABC CORP", 0.5, "N/A", "N/A"),
        ("2024-01-08", "acquisitionby", "XYZ1", "XYZ INC", 140.0, "ABC", "ABC CORP"),
        ("2024-01-08", "delisted", "XYZ1", "XYZ INC", 140.0, "N/A", "N/A"),
        ("2024-01-08", "acquisitioncash", "XYZ1", "XYZ INC", 10.0, "ABC", "ABC CORP"),
        ("2024-01-08", "acquisitionstock", "XYZ1", "XYZ INC", 0.1, "ABC", "ABC CORP"),
        ("2024-01-03", "tickerchangefrom", "ABC", "ABC CORP", "", "OLDA", "ABC CORP"),
    ]
    return pd.DataFrame(rows, columns=["date", "action", "ticker", "name", "value", "contraticker", "contraname"])


def fundamentals_frame() -> pd.DataFrame:
    """Sharadar ``fundamentals`` rows of ABC: one filing in two dimensions plus a restated (MRQ) row."""
    base = {"ticker": "ABC", "calendardate": "2023-09-30", "reportperiod": "2023-09-30", "fiscalperiod": "2023-Q3",
            "lastupdated": "2024-01-04"}
    return pd.DataFrame([
        {**base, "dimension": "ARQ", "date": "2024-01-03", "revenue": "1000", "netinc": "100", "fcf": "80"},
        {**base, "dimension": "ART", "date": "2024-01-03", "revenue": "4000", "netinc": "400", "fcf": "300"},
        {**base, "dimension": "MRQ", "date": "2023-09-30", "revenue": "1100", "netinc": "90", "fcf": "70"},
    ])


def write_raw(data_root: Path, zipped: bool = False) -> Path:
    """
    Write the synthetic raw tables under ``data_root/raw``.

    Args:
        data_root: Data root to fill (the folder should be named ``sharadar-data`` for purge tests).
        zipped: Write ``<table>.csv.zip`` (bulk form) instead of ``<table>.csv`` (sample form).

    Returns:
        ``data_root``.
    """
    raw = data_root / config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    tables = {"tickers": TICKERS, "stocks": stocks_frame(), "actions": actions_frame(), "fundamentals": fundamentals_frame(),
              "sp500": pd.DataFrame([{"date": "2024-01-02", "action": "added", "ticker": "ABC", "name": "ABC CORP"}])}
    for name, frame in tables.items():
        text = frame.to_csv(index=False)
        if zipped:
            with zipfile.ZipFile(raw / f"{name}.csv.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"{name}.csv", text)
        else:
            (raw / f"{name}.csv").write_text(text, encoding="utf-8")
    return data_root


def fmp_securities() -> pd.DataFrame:
    """A tiny FMP vintage securities table: ABC (id 7), its other class ABCB (id 8), and an unrelated stock."""
    return pd.DataFrame([
        {"security_id": 7, "symbol": "ABC", "cik": "0000000111", "instrument_type": "stock",
         "first_price_date": pd.Timestamp("2020-01-02").date(), "last_price_date": pd.Timestamp("2024-01-09").date()},
        {"security_id": 8, "symbol": "ABCB", "cik": "0000000111", "instrument_type": "stock",
         "first_price_date": pd.Timestamp("2019-01-02").date(), "last_price_date": pd.Timestamp("2024-01-09").date()},
        {"security_id": 9, "symbol": "QQQQ", "cik": "0000000999", "instrument_type": "stock",
         "first_price_date": pd.Timestamp("2020-01-02").date(), "last_price_date": pd.Timestamp("2024-01-09").date()},
    ])
