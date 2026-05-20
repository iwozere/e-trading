"""
Alert Timing Analysis Script

Analyzes the lag between when a ticker first enters the rolling window (first_seen / Phase 1 entry)
and when the Phase 2 alert is actually sent.

Answers:
  1. How many days pass between first detection and Phase 2 alert?
  2. How much has the price already moved by the time the alert fires?
  3. How does the stock perform after the alert (5d / 10d / 20d)?

Usage:
    python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing
    python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing --no-forward
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RESULTS_BASE = PROJECT_ROOT / "results" / "p06_emps2"
_SEARCH_ROOTS = [RESULTS_BASE, RESULTS_BASE / "p06_emps2"]


def _find_all_phase2_alert_files() -> list[Path]:
    files: list[Path] = []
    for root in _SEARCH_ROOTS:
        files.extend(root.glob("*/08_phase2_alerts.csv"))
    return sorted(set(files))


def _parse_folder_date(path: Path) -> Optional[date]:
    try:
        return datetime.strptime(path.parent.name, "%Y-%m-%d").date()
    except ValueError:
        return None


def _price_on_date(ticker: str, date_str: str) -> Optional[float]:
    """Return closing price for ticker from the volatility filter CSV on the given date."""
    for root in _SEARCH_ROOTS:
        vol_file = root / date_str / "05_volatility_filtered.csv"
        if vol_file.exists():
            try:
                df = pd.read_csv(vol_file)
                df = df[["ticker", "last_price"]]
                row = df[df["ticker"] == ticker]
                if not row.empty:
                    return float(row.iloc[0]["last_price"])
            except Exception:
                pass
    return None


def load_first_phase2_alerts() -> pd.DataFrame:
    """
    Load all Phase 2 alert CSV files and keep only the FIRST alert per ticker
    (its initial Phase 2 debut).

    Returns:
        DataFrame with one row per unique ticker (earliest alert_date).
    """
    files = _find_all_phase2_alert_files()
    _logger.info("Found %d Phase 2 alert files", len(files))

    records = []
    for path in files:
        alert_date = _parse_folder_date(path)
        if alert_date is None:
            continue
        try:
            df = pd.read_csv(path)
            df["alert_date"] = str(alert_date)
            records.append(df)
        except Exception:
            _logger.warning("Could not read %s", path)

    if not records:
        _logger.warning("No Phase 2 alert records found")
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    combined["first_seen"] = pd.to_datetime(combined["first_seen"]).dt.date
    combined["alert_date"] = pd.to_datetime(combined["alert_date"]).dt.date
    combined = combined.sort_values("alert_date")

    first_alerts = combined.groupby("ticker").first().reset_index()
    _logger.info(
        "Loaded %d unique ticker first-alerts from %d total records",
        len(first_alerts), len(combined)
    )
    return first_alerts


def compute_timing_metrics(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each first Phase 2 alert compute:
      - lag_days:            calendar days from first_seen to alert_date
      - price_at_first_seen: price on the day the ticker entered the rolling window
      - price_at_alert:      price from 05_volatility_filtered.csv on the alert_date
      - pre_alert_gain_pct:  % gain that occurred BEFORE the alert fired

    Note: ``latest_last_price`` stored in the Phase 2 CSV is the price from the
    *first_seen* date (rolling memory iterates newest→oldest so ``.last()`` lands on the
    oldest record). We therefore look up the alert-day price independently.

    Args:
        alerts_df: DataFrame from load_first_phase2_alerts()

    Returns:
        DataFrame with per-ticker timing metrics.
    """
    rows = []

    for _, row in alerts_df.iterrows():
        ticker = str(row["ticker"])
        first_seen: date = row["first_seen"]
        alert_date: date = row["alert_date"]
        # Look up price on the actual alert day from the volatility filter CSV
        price_at_alert: Optional[float] = _price_on_date(ticker, str(alert_date))

        lag_days: int = (alert_date - first_seen).days
        price_at_first_seen = _price_on_date(ticker, str(first_seen))

        pre_alert_gain_pct: Optional[float] = None
        if price_at_first_seen and price_at_alert and price_at_first_seen > 0:
            pre_alert_gain_pct = (price_at_alert / price_at_first_seen - 1.0) * 100.0

        rows.append({
            "ticker": ticker,
            "first_seen": first_seen,
            "alert_date": alert_date,
            "lag_days": lag_days,
            "price_at_first_seen": price_at_first_seen,
            "price_at_alert": price_at_alert,
            "pre_alert_gain_pct": pre_alert_gain_pct,
            "appearance_count": row.get("appearance_count"),
            "avg_vol_zscore": row.get("avg_vol_zscore"),
            "max_vol_zscore": row.get("max_vol_zscore"),
        })

        _logger.debug(
            "%s | first_seen=%s alert=%s lag=%dd | $%.2f → $%.2f | pre-alert: %s%%",
            ticker, first_seen, alert_date, lag_days,
            price_at_first_seen or 0.0,
            price_at_alert or 0.0,
            f"{pre_alert_gain_pct:+.1f}" if pre_alert_gain_pct is not None else "N/A"
        )

    return pd.DataFrame(rows)


def fetch_post_alert_returns(
    metrics_df: pd.DataFrame,
    days_forward: Optional[list[int]] = None
) -> pd.DataFrame:
    """
    Fetch post-alert forward returns via yfinance.

    Adds columns return_5d_pct, return_10d_pct, return_20d_pct.

    Args:
        metrics_df: Output of compute_timing_metrics()
        days_forward: Trading-day windows to measure (default: [5, 10, 20])

    Returns:
        metrics_df with forward-return columns appended.
    """
    if days_forward is None:
        days_forward = [5, 10, 20]

    try:
        import yfinance as yf
    except ImportError:
        _logger.warning("yfinance not installed – skipping post-alert returns")
        return metrics_df

    for d in days_forward:
        metrics_df[f"return_{d}d_pct"] = None

    for idx, row in metrics_df.iterrows():
        ticker = str(row["ticker"])
        alert_date: date = row["alert_date"]
        price_at_alert: Optional[float] = row["price_at_alert"]

        if price_at_alert is None or price_at_alert <= 0:
            continue

        try:
            buffer_days = max(days_forward) + 35
            end_date = alert_date + timedelta(days=buffer_days)
            hist = yf.download(
                ticker,
                start=str(alert_date),
                end=str(end_date),
                progress=False,
                auto_adjust=True
            )
            if hist is None or hist.empty:
                continue

            closes = hist["Close"].dropna()
            for d in days_forward:
                if len(closes) > d:
                    fwd_price = float(closes.iloc[d])
                    metrics_df.at[idx, f"return_{d}d_pct"] = (
                        fwd_price / price_at_alert - 1.0
                    ) * 100.0
        except Exception:
            _logger.warning("Could not fetch post-alert data for %s", ticker)

    return metrics_df


def print_summary(metrics_df: pd.DataFrame) -> None:
    """Log a human-readable summary of the timing analysis."""
    valid = metrics_df.dropna(subset=["pre_alert_gain_pct"])

    _logger.info("=== EMPS2 ALERT TIMING ANALYSIS ===")
    _logger.info("Total Phase 2 first-alerts: %d", len(metrics_df))
    _logger.info("With price-pair data: %d", len(valid))

    if valid.empty:
        return

    _logger.info("")
    _logger.info("--- Lag (first_seen → Phase 2 alert) ---")
    _logger.info("  Mean:   %.1f days", valid["lag_days"].mean())
    _logger.info("  Median: %.1f days", valid["lag_days"].median())
    _logger.info("  Range:  %d – %d days", int(valid["lag_days"].min()), int(valid["lag_days"].max()))

    _logger.info("")
    _logger.info("--- Pre-alert price gain (missed move) ---")
    _logger.info("  Mean:      %+.1f%%", valid["pre_alert_gain_pct"].mean())
    _logger.info("  Median:    %+.1f%%", valid["pre_alert_gain_pct"].median())
    _logger.info(
        "  Already >10%%: %d tickers (%.0f%%)",
        int((valid["pre_alert_gain_pct"] > 10).sum()),
        float((valid["pre_alert_gain_pct"] > 10).mean()) * 100
    )
    _logger.info(
        "  Already >20%%: %d tickers (%.0f%%)",
        int((valid["pre_alert_gain_pct"] > 20).sum()),
        float((valid["pre_alert_gain_pct"] > 20).mean()) * 100
    )

    _logger.info("")
    _logger.info("--- Top 10 'too-late' alerts (largest pre-alert gain) ---")
    worst = valid.nlargest(10, "pre_alert_gain_pct")[
        ["ticker", "first_seen", "alert_date", "lag_days",
         "price_at_first_seen", "price_at_alert", "pre_alert_gain_pct"]
    ]
    for _, r in worst.iterrows():
        _logger.info(
            "  %-6s  lag=%2dd  $%6.2f → $%6.2f  (%+.1f%%)  [%s → %s]",
            r["ticker"], r["lag_days"],
            float(r["price_at_first_seen"]) if r["price_at_first_seen"] else 0.0,
            float(r["price_at_alert"]) if r["price_at_alert"] else 0.0,
            float(r["pre_alert_gain_pct"]),
            r["first_seen"], r["alert_date"]
        )

    for col, label in [("return_5d_pct", "5d"), ("return_10d_pct", "10d"), ("return_20d_pct", "20d")]:
        if col in valid.columns:
            fwd = valid[col].dropna()
            if not fwd.empty:
                _logger.info("")
                _logger.info(
                    "--- Post-alert %s returns (n=%d) ---",
                    label, len(fwd)
                )
                _logger.info(
                    "  Mean: %+.1f%%  Median: %+.1f%%  >0: %.0f%%",
                    float(fwd.mean()), float(fwd.median()),
                    float((fwd > 0).mean()) * 100
                )


def main() -> None:
    """Entry point: run timing analysis and save results."""
    _logger.info("Starting EMPS2 Alert Timing Analysis")

    alerts_df = load_first_phase2_alerts()
    if alerts_df.empty:
        _logger.error("No Phase 2 alerts found – nothing to analyse")
        return

    metrics_df = compute_timing_metrics(alerts_df)

    fetch_forward = "--no-forward" not in sys.argv
    if fetch_forward:
        _logger.info("Fetching post-alert forward returns (pass --no-forward to skip)...")
        metrics_df = fetch_post_alert_returns(metrics_df)
    else:
        _logger.info("Skipping post-alert return fetch (--no-forward)")

    output_path = RESULTS_BASE / "timing_analysis.csv"
    metrics_df.to_csv(output_path, index=False)
    _logger.info("Saved timing analysis → %s", output_path)

    print_summary(metrics_df)


if __name__ == "__main__":
    main()
