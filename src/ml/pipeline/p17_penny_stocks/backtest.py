"""
P17 Best-Case Backtest

Answers: "If I had invested a fixed amount in every ticker the screener ever
reported, on the day it was first detected, and sold at the highest price reached
between detection and now — what would the result be?"

This is a deliberately optimistic (perfect-exit) backtest: it sells at the peak
``high`` over the holding window, so the numbers are an *upper bound* on what the
screen could have produced, not an achievable strategy. It is useful for ranking
tiers against each other and for spotting whether the screen surfaces names that
subsequently moved at all.

Method
------
1. Scan every ``{date}/{date}_candidates.csv`` under the results dir.
2. Deduplicate to each ticker's *first* appearance (its detection date), keeping
   the tier and price recorded that day.
3. For each ticker, download daily OHLCV from detection date to today and take
   the maximum ``high``.
4. Buy ``--invest`` USD at the detection-day price; "sell" at that max high.
5. Aggregate profit overall and broken down by detection tier (A/B/C/W).

Usage
-----
    python -m src.ml.pipeline.p17_penny_stocks.backtest
    python -m src.ml.pipeline.p17_penny_stocks.backtest --since 2026-05-01 --invest 1000
    python -m src.ml.pipeline.p17_penny_stocks.backtest --tiers A,B,C
"""

import argparse
import glob
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_RESULTS_DIR = "results/p17_penny_stocks"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P17 best-case ($ at peak) backtest")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory holding {date}/{date}_candidates.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--invest",
        type=float,
        default=1000.0,
        help="USD invested per ticker at detection (default: %(default)s)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only consider detection dates on/after this YYYY-MM-DD",
    )
    parser.add_argument(
        "--tiers",
        default=None,
        help="Comma-separated tiers to include (e.g. A,B,C). Default: all",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <results-dir>/backtest_results.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N tickers (for a quick smoke test)",
    )
    return parser.parse_args()


def collect_first_detections(
    results_dir: str,
    since: str | None = None,
    tiers: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a ticker -> first-detection record map from the daily candidate CSVs.

    Args:
        results_dir: Directory containing ``{date}/{date}_candidates.csv`` files.
        since: Optional ``YYYY-MM-DD`` lower bound on detection date.
        tiers: Optional list of tiers to keep (record's first-detection tier).

    Returns:
        Mapping ticker -> {detection_date, detection_price, tier, company_name,
        final_score}. Each ticker keeps its earliest appearance.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*", "*_candidates.csv")))
    records: Dict[str, Dict[str, Any]] = {}
    for f in files:
        day = os.path.basename(os.path.dirname(f))
        if since and day < since:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            _logger.warning("Could not read %s — skipping", f)
            continue
        if "ticker" not in df.columns or "price" not in df.columns:
            continue
        for r in df.itertuples():
            ticker = str(getattr(r, "ticker", "")).strip().upper()
            if not ticker or ticker == "NAN":
                continue
            if ticker in records:
                continue  # earlier date already recorded (files are date-sorted)
            try:
                price = float(getattr(r, "price"))
            except (TypeError, ValueError):
                continue
            if not price or price <= 0:
                continue
            records[ticker] = {
                "ticker": ticker,
                "detection_date": day,
                "detection_price": price,
                "tier": str(getattr(r, "tier", "W")),
                "company_name": str(getattr(r, "company_name", "")),
                "final_score": float(getattr(r, "final_score", 0.0) or 0.0),
            }

    if tiers:
        tierset = {t.strip().upper() for t in tiers}
        records = {k: v for k, v in records.items() if v["tier"] in tierset}
    return records


def backtest_ticker(
    downloader: Any,
    rec: Dict[str, Any],
    invest: float,
    end_date: datetime,
) -> Dict[str, Any]:
    """
    Compute the best-case result for a single ticker.

    Returns the record enriched with ``max_high``, ``peak_date``, ``shares``,
    ``peak_value``, ``profit``, ``return_pct`` and ``status`` (``ok`` /
    ``no_data``).
    """
    out = dict(rec)
    start = datetime.strptime(rec["detection_date"], "%Y-%m-%d")
    try:
        df = downloader.get_ohlcv(rec["ticker"], "1d", start, end_date)
    except Exception:
        _logger.debug("OHLCV fetch failed for %s", rec["ticker"])
        df = None

    if df is None or df.empty or "high" not in df.columns or df["high"].dropna().empty:
        out.update(
            {
                "status": "no_data",
                "max_high": None,
                "peak_date": None,
                "shares": None,
                "peak_value": None,
                "profit": None,
                "return_pct": None,
            }
        )
        return out

    high = df["high"].astype(float)
    max_high = high.max()
    peak_idx = high.idxmax()
    peak_date = None
    if "timestamp" in df.columns:
        try:
            peak_date = str(pd.to_datetime(df.loc[peak_idx, "timestamp"]).date())  # type: ignore[call-overload]
        except Exception:
            peak_date = None

    shares = invest / rec["detection_price"]
    peak_value = shares * max_high
    out.update(
        {
            "status": "ok",
            "max_high": round(max_high, 4),
            "peak_date": peak_date,
            "shares": round(shares, 4),
            "peak_value": round(peak_value, 2),
            "profit": round(peak_value - invest, 2),
            "return_pct": round((max_high / rec["detection_price"] - 1.0) * 100.0, 1),
        }
    )
    return out


def summarize(results: List[Dict[str, Any]], invest: float) -> pd.DataFrame:
    """Build an overall + per-tier summary DataFrame from per-ticker results."""
    ok = [r for r in results if r["status"] == "ok"]

    def agg(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        n = len(rows)
        invested = n * invest
        peak = sum(r["peak_value"] for r in rows)
        profit = sum(r["profit"] for r in rows)
        winners = sum(1 for r in rows if r["profit"] > 0)
        avg_ret = (sum(r["return_pct"] for r in rows) / n) if n else 0.0
        return {
            "group": label,
            "tickers": n,
            "invested": round(invested, 2),
            "peak_value": round(peak, 2),
            "profit": round(profit, 2),
            "roi_pct": round((profit / invested * 100.0) if invested else 0.0, 1),
            "avg_return_pct": round(avg_ret, 1),
            "win_rate_pct": round((winners / n * 100.0) if n else 0.0, 1),
        }

    rows = [agg(ok, "ALL")]
    for tier in ("A", "B", "C", "W"):
        sub = [r for r in ok if r["tier"] == tier]
        if sub:
            rows.append(agg(sub, f"Tier {tier}"))
    return pd.DataFrame(rows)


def main() -> int:
    args = _parse_args()
    tiers = [t for t in args.tiers.split(",")] if args.tiers else None
    out_path = args.out or os.path.join(args.results_dir, "backtest_results.csv")
    end_date = datetime.now(UTC).replace(tzinfo=None)

    records = collect_first_detections(args.results_dir, args.since, tiers)
    items = list(records.values())
    items.sort(key=lambda r: (r["detection_date"], r["ticker"]))
    if args.limit:
        items = items[: args.limit]

    if not items:
        _logger.error("No candidate records found under %s", args.results_dir)
        return 1

    _logger.info("Backtesting %d unique tickers (invest $%.0f each)…", len(items), args.invest)
    downloader = YahooDataDownloader()
    results: List[Dict[str, Any]] = []
    for i, rec in enumerate(items, 1):
        results.append(backtest_ticker(downloader, rec, args.invest, end_date))
        if i % 25 == 0:
            _logger.info("  %d/%d processed", i, len(items))

    detail = pd.DataFrame(results)
    detail.to_csv(out_path, index=False)

    summary = summarize(results, args.invest)
    no_data = sum(1 for r in results if r["status"] == "no_data")

    print("\n=== P17 best-case backtest (sell at peak high) ===")
    print(f"detection window: {items[0]['detection_date']} → {items[-1]['detection_date']}")
    print(f"tickers: {len(results)} ({no_data} had no price data and were excluded from totals)")
    print(f"invested per ticker: ${args.invest:.0f}\n")
    print(summary.to_string(index=False))
    print(f"\nper-ticker detail written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
