import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Path to local CSV with VIX data
VIX_FILE = Path("data/vix/vix.csv")


def calculate_regime(vix_value: float) -> str:
    """
    Calculate market regime based on VIX value.

    Args:
        vix_value: VIX value

    Returns:
        Market regime: "calm", "normal", or "fear"
    """
    if pd.isna(vix_value):
        return "unknown"
    if vix_value < 15:
        return "calm"      # Low volatility
    elif vix_value < 25:
        return "normal"    # Normal market
    else:
        return "fear"      # Panic


def update_vix() -> None:
    """
    Update VIX data by downloading the latest data from Yahoo Finance.

    This function:
    1. Creates the data/vix/ directory if it doesn't exist
    2. Loads existing VIX data from CSV if available
    3. Downloads new data from Yahoo Finance starting from the last available date
    4. Merges old and new data, removing duplicates
    5. Saves the updated dataset to CSV with normalized format (date, vix, regime)

    The VIX (Volatility Index) data is downloaded from Yahoo Finance using the ^VIX ticker.
    The output CSV will have 3 columns: date, vix (close price), and regime (calm/normal/fear).
    """
    # Create directory if it doesn't exist
    vix_dir = VIX_FILE.parent
    vix_dir.mkdir(parents=True, exist_ok=True)
    if not vix_dir.exists():
        _logger.info("Created directory: %s", vix_dir)

    # Load existing CSV if available
    vix_local: pd.DataFrame
    last_date: datetime.date

    if VIX_FILE.exists():
        try:
            # Try to load as normalized format first
            vix_local = pd.read_csv(VIX_FILE)
            if "date" in vix_local.columns and "vix" in vix_local.columns and "regime" in vix_local.columns:
                # Already in normalized format
                vix_local["date"] = pd.to_datetime(vix_local["date"])
                vix_local.set_index("date", inplace=True)
                last_date = vix_local.index[-1].date()
                _logger.debug("Loaded existing normalized VIX data, last date: %s", last_date)
            else:
                # Old format, convert to normalized
                vix_local = pd.read_csv(VIX_FILE, parse_dates=["Date"], index_col="Date")
                vix_local = vix_local[["Close"]].rename(columns={"Close": "vix"})
                vix_local["regime"] = vix_local["vix"].map(calculate_regime)
                last_date = vix_local.index[-1].date()
                _logger.debug("Converted old format VIX data to normalized format, last date: %s", last_date)
        except Exception as e:
            _logger.warning("Error loading existing VIX data, starting fresh: %s", e)
            vix_local = pd.DataFrame(columns=["vix", "regime"])
            last_date = datetime(1990, 1, 1).date()
    else:
        vix_local = pd.DataFrame(columns=["vix", "regime"])
        last_date = datetime(1990, 1, 1).date()  # First date for VIX
        _logger.info("No existing VIX data found, starting from %s", last_date)

    # Download new data from Yahoo (starting from last date)
    today = datetime.today().date()
    # Extend end date by a few days to ensure we get the most recent trading data
    # This handles cases where the requested end date falls on a weekend/holiday
    end_date = today + timedelta(days=7)
    _logger.info("Downloading VIX data from %s to %s (extended to %s to ensure we get recent trading data)", last_date, today, end_date)

    try:
        vix_new = yf.download("^VIX", start=last_date, end=end_date)
        vix_new.reset_index(inplace=True)

        # Handle multi-index columns from yfinance
        if isinstance(vix_new.columns, pd.MultiIndex):
            # Flatten multi-index columns by taking the first level
            vix_new.columns = vix_new.columns.droplevel(1)
            _logger.debug("Flattened multi-index columns: %s", list(vix_new.columns))

    except Exception:
        _logger.exception("Failed to download VIX data from Yahoo Finance")
        return

    if not vix_new.empty:
        # Convert to normalized format
        vix_new = vix_new[["Date", "Close"]].rename(columns={"Close": "vix"})
        vix_new.set_index("Date", inplace=True)
        vix_new["regime"] = vix_new["vix"].map(calculate_regime)

        # Merge old and new data, handling empty DataFrames
        if vix_local.empty:
            vix_all = vix_new
        else:
            vix_all = pd.concat([vix_local, vix_new])
            vix_all = vix_all[~vix_all.index.duplicated(keep="last")]  # Remove duplicates

        # Reset index to make date a column and save to CSV
        vix_all.reset_index(inplace=True)

        # Ensure the date column is named correctly
        if "Date" in vix_all.columns:
            vix_all.rename(columns={"Date": "date"}, inplace=True)
        elif "date" not in vix_all.columns:
            # If the index name is different, rename it
            vix_all.rename(columns={vix_all.columns[0]: "date"}, inplace=True)

        # Save to CSV in normalized format with only the required columns
        vix_all[["date", "vix", "regime"]].to_csv(VIX_FILE, index=False)

        _logger.info("VIX data updated successfully. Last date: %s", vix_all["date"].iloc[-1].date())
        _logger.info("Total records: %d", len(vix_all))
        _logger.info("Current regime: %s (VIX: %.2f)", vix_all["regime"].iloc[-1], vix_all["vix"].iloc[-1])
    else:
        _logger.warning("No new VIX data available")


if __name__ == "__main__":
    import json

    update_vix()

    # Output result for scheduler
    try:
        if VIX_FILE.exists():
            vix_data = pd.read_csv(VIX_FILE)
            if not vix_data.empty:
                latest = vix_data.iloc[-1]
                result = {
                    "success": True,
                    "vix_current": float(latest["vix"]),
                    "regime": str(latest["regime"]),
                    "date": str(latest["date"]),
                    "total_records": len(vix_data)
                }
            else:
                result = {"success": False, "error": "No VIX data available"}
        else:
            result = {"success": False, "error": "VIX file not found"}
    except Exception as e:
        result = {"success": False, "error": str(e)}

    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
