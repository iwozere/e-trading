I want dark pool analyzer but without AST, only with TRF.
Make sure that this data is downloaded as a part of emps2 run into results/emps2/yyyy-mm-dd folder.
I would like to get this at 6am by APScheduler (see also vix.py / run_emps2_scan.py cron etc.)

Every time when this file is downloaded it should go into the yesterda's folder (last marked date folder).
Apart from FINRA data I also want there volume data from yfinance to know what volume was traded yesterday and what traded via dark pool yesterday in order to build the ratio and evaluate during today's run.

------------------------------------

ohlcv = load_your_ohlcv()
df = df.merge(ohlcv[["tradeDate", "ticker", "volume"]], on=["tradeDate", "ticker"], how="left")
df = df.rename(columns={"volume": "regular_volume"})

------------------------------------

import requests
import pandas as pd
import os

OUT_CSV = "finra_trf_daily.csv"
FINRA_TRF_URL = "https://api.finra.org/data/trf/OTCMarketVolume"


def download_finra_trf():
    print("[*] Downloading FINRA TRF (off-exchange) data...")

    r = requests.get(FINRA_TRF_URL, timeout=30)
    r.raise_for_status()

    data = r.json()

    df = pd.DataFrame(data)
    print(f"[*] Downloaded rows: {len(df)}")

    # Convert date
    if "tradeDate" in df.columns:
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])

    df = df.rename(columns={
        "issueSymbol": "ticker",
        "totalVolume": "offex_volume",
        "totalTrades": "offex_trades"
    })

    # Keep useful fields
    keep = ["tradeDate", "ticker", "offex_volume", "offex_trades", "lastPrice", "marketCategory"]
    df = df[keep]

    df = df.sort_values(["tradeDate", "ticker"])
    return df


def load_existing():
    if not os.path.exists(OUT_CSV):
        return pd.DataFrame(columns=["tradeDate", "ticker", "offex_volume", "offex_trades", "lastPrice", "marketCategory"])

    return pd.read_csv(OUT_CSV, parse_dates=["tradeDate"])


def merge_save(new_df):
    old_df = load_existing()
    combined = pd.concat([old_df, new_df], ignore_index=True)

    combined = combined.drop_duplicates(subset=["tradeDate", "ticker"], keep="last")
    combined = combined.sort_values(["tradeDate", "ticker"])

    combined.to_csv(OUT_CSV, index=False)
    print(f"[*] Saved {len(combined)} records to {OUT_CSV}")


def main():
    df = download_finra_trf()
    merge_save(df)


if __name__ == "__main__":
    main()


------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path


class DarkPoolAnalyzer:
    """
    EMPS Dark Pool + Off-Exchange Analyzer
    Loads daily TRF (off-exchange) + ATS (dark pools) data
    and produces EMPS features:
        - dark_ratio = ATS / TRF
        - offex_ratio = TRF / RegularVolume
        - ats_spike (z-score or MA ratio)
        - accum_score (aggregated EMPS signal)
    """

    def __init__(
        self,
        ats_path: str = "finra_darkpool_daily.csv",
        trf_path: str = "finra_trf_daily.csv",
        ohlcv_path: str = None,
    ):
        self.ats_path = Path(ats_path)
        self.trf_path = Path(trf_path)
        self.ohlcv_path = Path(ohlcv_path) if ohlcv_path else None

        self.ats = None
        self.trf = None
        self.ohlcv = None
        self.df = None

    # -------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------

    def load_ats(self):
        """Load ATS (dark pool) data from nightly downloaded file"""
        if not self.ats_path.exists():
            raise FileNotFoundError(f"ATS file not found: {self.ats_path}")

        df = pd.read_csv(self.ats_path, parse_dates=["tradeDate"])
        df = df.rename(
            columns={
                "volume": "ats_volume",
                "trades": "ats_trades"
            }
        )
        self.ats = df
        return self.ats

    def load_trf(self):
        """Load TRF (off-exchange) data from nightly downloaded file"""
        if not self.trf_path.exists():
            raise FileNotFoundError(f"TRF file not found: {self.trf_path}")

        df = pd.read_csv(self.trf_path, parse_dates=["tradeDate"])
        df = df.rename(
            columns={
                "offex_volume": "trf_volume",
                "offex_trades": "trf_trades"
            }
        )
        self.trf = df
        return self.trf

    def load_ohlcv(self):
        """Load your OHLCV data (daily aggregated)"""
        if not self.ohlcv_path or not self.ohlcv_path.exists():
            raise FileNotFoundError(
                f"OHLCV path not found: {self.ohlcv_path}"
            )

        df = pd.read_csv(self.ohlcv_path, parse_dates=["date"])
        df = df.rename(columns={"date": "tradeDate"})
        self.ohlcv = df
        return self.ohlcv

    # -------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------

    def merge_all(self):
        """Merge TRF + ATS + OHLCV into one table"""
        if self.ats is None:
            self.load_ats()

        if self.trf is None:
            self.load_trf()

        df = self.trf.merge(
            self.ats[["tradeDate", "ticker", "ats_volume"]],
            how="left",
            on=["tradeDate", "ticker"]
        )

        # Replace NaN ATS with 0 (not all tickers trade in dark pools)
        df["ats_volume"] = df["ats_volume"].fillna(0)

        if self.ohlcv is None:
            self.load_ohlcv()

        df = df.merge(
            self.ohlcv[["tradeDate", "ticker", "volume"]],
            on=["tradeDate", "ticker"],
            how="left"
        )
        df = df.rename(columns={"volume": "regular_volume"})

        # Replace missing regular volumes with 0 if needed
        df["regular_volume"] = df["regular_volume"].fillna(0)

        # Sort for rolling windows
        df = df.sort_values(["ticker", "tradeDate"]).reset_index(drop=True)
        self.df = df
        return df

    # -------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------

    def compute_features(self):
        """Generate EMPS dark-pool features"""

        if self.df is None:
            self.merge_all()

        df = self.df

        # -----------------------------
        # 1. dark_ratio = ATS / TRF
        # -----------------------------
        df["dark_ratio"] = df["ats_volume"] / df["trf_volume"]
        df["dark_ratio"] = df["dark_ratio"].fillna(0.0)

        # -----------------------------
        # 2. offex_ratio = TRF / Regular
        # -----------------------------
        df["offex_ratio"] = df["trf_volume"] / df["regular_volume"]
        df["offex_ratio"] = df["offex_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

        # -----------------------------
        # 3. ATS spike
        # Rolling mean ratio (20-day)
        # -----------------------------
        df["ats_ma20"] = (
            df.groupby("ticker")["ats_volume"]
            .transform(lambda x: x.rolling(20).mean())
        )

        df["ats_spike"] = df["ats_volume"] / df["ats_ma20"]
        df["ats_spike"] = df["ats_spike"].replace([np.inf, -np.inf], 0).fillna(0)

        # -----------------------------
        # 4. Accumulation Score (EMPS)
        # -----------------------------
        df["accum_score"] = (
            0.4 * df["dark_ratio"] +
            0.3 * df["offex_ratio"] +
            0.3 * df["ats_spike"]
        )

        self.df = df
        return df

    # -------------------------------------------------------------
    # Export
    # -------------------------------------------------------------

    def save(self, out_path="darkpool_features.csv"):
        if self.df is None:
            raise ValueError("Data not computed yet. Call compute_features().")

        out_path = Path(out_path)
        self.df.to_csv(out_path, index=False)
        print(f"[DarkPoolAnalyzer] Saved features to {out_path}")

    def save_parquet(self, out_path="darkpool_features.parquet"):
        if self.df is None:
            raise ValueError("Data not computed yet. Call compute_features().")

        out_path = Path(out_path)
        self.df.to_parquet(out_path, index=False)
        print(f"[DarkPoolAnalyzer] Saved features to {out_path}")
