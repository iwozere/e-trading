"""Stage 2 — Deterministic Signal Scoring: ~200 → top-25 candidates."""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.config import (
    STAGE2_TOP_N,
    STAGE2_CACHE_DIR,
    P18_HIGH_SCORE_THRESHOLD,
    P18_WEIGHTS,
    CRYPTO_TICKERS,
)
from src.data.data_manager import DataManager
from src.ml.pipeline.p05_ai_selector.signals.fundamental import score_fundamentals, build_sector_medians
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_CRYPTO_SET = set(CRYPTO_TICKERS)


class Stage2Scorer:
    """
    Computes deterministic composite scores and ranks the top-25 candidates.

    Score = momentum_score (Stage 1) + fundamental_score + p18_score.
    Ties broken by volume_surge_ratio.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or STAGE2_CACHE_DIR

    def run(
        self,
        stage1_df: pd.DataFrame,
        p18_data: Dict[str, Any],
        earnings_flags: Dict[str, date],
        as_of_date: date,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Score and rank the Stage 1 candidates.

        Args:
            stage1_df: Output of Stage1Prefilter.run().
            p18_data: Output of P18Reader.get_high_score_tickers().
            earnings_flags: {ticker: earnings_date} from EarningsCalendar.
            as_of_date: Reference date.
            force_refresh: Bypass cache.

        Returns:
            DataFrame with top STAGE2_TOP_N rows, sorted by total_score desc.
        """
        cache_file = self._cache_dir / f"{as_of_date}.csv.gz"
        if not force_refresh and cache_file.exists():
            _logger.info("Stage2: loading from cache %s", cache_file)
            return pd.read_csv(cache_file, compression="gzip")

        if stage1_df.empty:
            _logger.warning("Stage2: empty Stage 1 input")
            return pd.DataFrame()

        tickers = stage1_df["ticker"].tolist()
        _logger.info("Stage2: scoring %d candidates", len(tickers))

        fundamentals_map = self._fetch_all_fundamentals(tickers)
        sector_medians = build_sector_medians(fundamentals_map)

        rows = []
        for _, s1_row in stage1_df.iterrows():
            ticker = str(s1_row["ticker"])
            is_crypto = ticker in _CRYPTO_SET

            momentum_score = float(s1_row.get("momentum_score", 0))
            vol_surge = float(s1_row.get("volume_surge_ratio", 1.0))
            last_price = float(s1_row.get("last_price", 0))
            avg_vol_usd = float(s1_row.get("avg_vol_usd", 0))
            asset_type = str(s1_row.get("asset_type", "equity"))

            fund = fundamentals_map.get(ticker)
            if is_crypto:
                fundamental_score = 0.0
                fund_breakdown: Dict[str, object] = {}
                fundamentals_available = False
                market_cap_b = 0.0
            else:
                fundamental_score, fund_breakdown = score_fundamentals(fund, sector_medians)
                fundamentals_available = fund is not None
                try:
                    if isinstance(fund, dict):
                        mc = fund.get("market_cap")
                    else:
                        mc = getattr(fund, "market_cap", None) if fund else None
                    market_cap_b = round(float(mc) / 1e9, 4) if mc else 0.0
                except (TypeError, ValueError):
                    market_cap_b = 0.0

            p18_score, p18_breakdown = self._compute_p18_score(ticker, p18_data)

            earnings_date = earnings_flags.get(ticker.upper())
            earnings_flag = earnings_date is not None

            total_score = momentum_score + fundamental_score + p18_score

            rows.append({
                "ticker": ticker,
                "asset_type": asset_type,
                "last_price": last_price,
                "market_cap_b": market_cap_b,
                "total_score": round(total_score, 2),
                "momentum_score": round(momentum_score, 2),
                "fundamental_score": round(fundamental_score, 2),
                "p18_score": round(p18_score, 2),
                "volume_surge_ratio": vol_surge,
                "earnings_flag": earnings_flag,
                "earnings_date": str(earnings_date) if earnings_date else "",
                "fundamentals_available": fundamentals_available,
                "signal_breakdown": json.dumps({
                    **json.loads(s1_row.get("signal_breakdown", "{}")),
                    "fundamentals": fund_breakdown,
                    "p18": p18_breakdown,
                }),
            })

        df = pd.DataFrame(rows)
        df = (
            df.sort_values(["total_score", "volume_surge_ratio"], ascending=[False, False])
            .head(STAGE2_TOP_N)
            .reset_index(drop=True)
        )

        _logger.info("Stage2 complete: output=%d candidates", len(df))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False, compression="gzip")
        return df

    def _fetch_all_fundamentals(self, tickers: list) -> Dict[str, Any]:
        """Fetch fundamentals for all non-crypto tickers via DataManager."""
        dm = DataManager()
        result: Dict[str, Any] = {}

        for ticker in tickers:
            if ticker in _CRYPTO_SET:
                result[ticker] = None
                continue
            try:
                fund_dict = dm.get_fundamentals(ticker)
                result[ticker] = fund_dict if fund_dict else None
            except Exception:
                _logger.warning("Stage2: failed to fetch fundamentals for %s", ticker)
                result[ticker] = None

        return result

    def _compute_p18_score(
        self, ticker: str, p18_data: Dict[str, Any]
    ) -> tuple:
        """Apply P18 signal boosts per spec §6.3."""
        score = 0.0
        breakdown: Dict[str, object] = {}

        p18_tickers: Dict[str, float] = p18_data.get("tickers", {})
        if ticker in p18_tickers:
            score += P18_WEIGHTS.get("high_score", 40)
            breakdown["high_score"] = True

        consensus: set = p18_data.get("consensus_tickers", set())
        if ticker in consensus:
            score += P18_WEIGHTS.get("consensus_exit", 25)
            breakdown["consensus_exit"] = True

        form4_buys: set = p18_data.get("form4_buy_tickers", set())
        if ticker in form4_buys:
            score += P18_WEIGHTS.get("form4_insider_buy", 15)
            breakdown["form4_insider_buy"] = True

        dg_tickers: set = p18_data.get("13dg_tickers", set())
        if ticker in dg_tickers:
            score += P18_WEIGHTS.get("schedule_13dg", 15)
            breakdown["schedule_13dg"] = True

        return score, breakdown
