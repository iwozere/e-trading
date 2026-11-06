"""
P17 Point-in-Time Replay

Re-scores every saved daily run with the *current* Technical/Scoring code and
measures what an investor would actually have earned entering at the next
session's open — the question the best-case ``backtest.py`` cannot answer.

Point-in-time inputs per run date D:
  * Universe + fundamentals / short interest / dilution / catalyst score come
    from ``{D}/{D}_candidates.csv`` exactly as the prod run saw them.
  * Technical features are recomputed from daily OHLCV truncated at D (same
    100-calendar-day window the MarketAgent downloads).

Signal sets compared (each deduplicated to the first signal per ticker within a
``--cooldown-days`` window, so a name flagged five days running counts once):
  * ``prod_ab``        — Tier A/B or explosive as recorded in the prod CSV
  * ``replay_legacy``  — same, re-scored with the extension gate disabled (fidelity check)
  * ``replay_ab``      — Tier A/B or explosive with the extension gate on
  * ``setup``          — pre-breakout setup watchlist
  * ``universe``       — every candidate row (baseline; not deduplicated)

Usage
-----
    python -m src.ml.pipeline.p17_penny_stocks.replay --results-dir R:/apps/e-trading/results/p17_penny_stocks
    python -m src.ml.pipeline.p17_penny_stocks.replay --since 2026-07-01 --out-dir /tmp/p17_replay
"""

import argparse
import dataclasses
import glob
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p17_penny_stocks.agents.market_agent import MarketAgent
from src.ml.pipeline.p17_penny_stocks.agents.scoring_agent import ScoringAgent
from src.ml.pipeline.p17_penny_stocks.agents.short_squeeze_agent import ShortSqueezeAgent
from src.ml.pipeline.p17_penny_stocks.agents.technical_agent import TechnicalAgent
from src.ml.pipeline.p17_penny_stocks.config import P17PipelineConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_RESULTS_DIR = "results/p17_penny_stocks"
OHLCV_WINDOW_DAYS = 100  # MarketAgent: ohlcv_lookback_days (90) + 10
HORIZONS = (1, 5, 10, 20)

# Candidate fields restored from the saved CSV (everything not recomputed by TechnicalAgent).
_FLOAT_FIELDS = (
    "price", "market_cap", "volume", "avg_volume_30d", "float_shares", "high_52w", "low_52w",
    "dilution_penalty", "catalyst_score",
)
_OPTIONAL_FIELDS = (
    "short_interest_pct_float", "days_to_cover", "finra_short_vol_ratio", "revenue_growth_yoy",
    "gross_margin", "total_cash", "total_debt", "cash_runway_months",
)
_LIST_FIELDS = ("dilution_signals", "catalyst_signals")


def _optional(value: Any) -> Optional[float]:
    """Float or None (NaN / blanks → None)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def candidate_from_row(row: Dict[Any, Any]) -> Candidate:
    """
    Rebuild the pre-technical state of a Candidate from one saved CSV row.

    Args:
        row: A record from ``{date}_candidates.csv``.

    Returns:
        Candidate with identity, snapshot, fundamentals, short-interest,
        dilution and catalyst fields restored; technical fields left at defaults.
    """
    c = Candidate(ticker=str(row["ticker"]))
    for name in ("company_name", "exchange", "sector", "industry"):
        val = row.get(name)
        setattr(c, name, "" if val is None or (isinstance(val, float) and math.isnan(val)) else str(val))
    for name in _FLOAT_FIELDS:
        setattr(c, name, _optional(row.get(name)) or 0.0)
    for name in _OPTIONAL_FIELDS:
        setattr(c, name, _optional(row.get(name)))
    for name in _LIST_FIELDS:
        val = row.get(name)
        setattr(c, name, [s for s in str(val).split("|") if s] if isinstance(val, str) else [])
    c.run_date = str(row.get("run_date", ""))
    return c


def dedupe_episodes(signals: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    """
    Keep only the first signal per ticker; a new episode starts after ``cooldown_days`` of silence.

    Args:
        signals: Rows with at least ``ticker`` and ``run_date`` (datetime64).
        cooldown_days: Calendar-day gap that separates episodes.

    Returns:
        Subset of ``signals`` with one row per episode.
    """
    if signals.empty:
        return signals
    s = signals.sort_values(["ticker", "run_date"]).copy()
    gap = s.groupby("ticker")["run_date"].diff().dt.days
    return s.loc[gap.isna() | (gap > cooldown_days)]


def forward_outcomes(ohlcv: pd.DataFrame, run_date: pd.Timestamp) -> Optional[Dict[str, float]]:
    """
    Outcomes of entering at the first session open strictly after ``run_date``.

    Args:
        ohlcv: Daily OHLCV (columns Open/High/Low/Close, DatetimeIndex ascending).
        run_date: The run's target date (signal-day close).

    Returns:
        Dict with ``fwd{h}`` close-to-entry returns, ``max_up_10d`` / ``max_dd_10d``
        (intraday extremes vs entry), or None when no later session exists.
    """
    after = ohlcv.loc[ohlcv.index > run_date]
    if after.empty:
        return None
    entry = float(after["Open"].iloc[0])
    if entry <= 0:
        return None
    out: Dict[str, float] = {}
    for h in HORIZONS:
        out[f"fwd{h}"] = float(after["Close"].iloc[h - 1]) / entry - 1.0 if len(after) >= h else float("nan")
    w10 = after.iloc[:10]
    out["max_up_10d"] = float(w10["High"].max()) / entry - 1.0
    out["max_dd_10d"] = float(w10["Low"].min()) / entry - 1.0
    return out


def summarise(episodes: pd.DataFrame, name: str) -> Dict[str, Any]:
    """One summary row for a signal set."""
    row: Dict[str, Any] = {"set": name, "n": len(episodes)}
    if episodes.empty:
        return row
    row["runup_5d_med"] = episodes["price_5d_return"].median()
    for h in (5, 10, 20):
        col = episodes[f"fwd{h}"].dropna()
        row[f"fwd{h}_med"] = col.median() if len(col) else float("nan")
        row[f"fwd{h}_mean"] = col.mean() if len(col) else float("nan")
        row[f"fwd{h}_win"] = (col > 0).mean() if len(col) else float("nan")
    row["max_up_10d_med"] = episodes["max_up_10d"].median()
    row["max_dd_10d_med"] = episodes["max_dd_10d"].median()
    return row


class P17Replay:
    """Point-in-time re-scoring of saved P17 runs with forward-return evaluation."""

    def __init__(self, results_dir: str, config: Optional[P17PipelineConfig] = None) -> None:
        self.results_dir = results_dir
        self.config = config or P17PipelineConfig.create_default()
        legacy_scoring = dataclasses.replace(
            self.config.scoring_config, max_alert_5d_return=float("inf"), max_alert_extension_atr=float("inf")
        )
        ss_agent = ShortSqueezeAgent(self.config.short_squeeze_config, "")
        self._technical = TechnicalAgent(self.config.technical_config)
        self._scoring = ScoringAgent(self.config.scoring_config, ss_agent)
        self._scoring_legacy = ScoringAgent(legacy_scoring, ss_agent)

    def run_files(self, since: Optional[str], until: Optional[str]) -> List[str]:
        """Dated candidates CSVs within [since, until]."""
        files = sorted(glob.glob(os.path.join(self.results_dir, "????-??-??", "????-??-??_candidates.csv")))
        out = []
        for f in files:
            day = os.path.basename(os.path.dirname(f))
            if (since and day < since) or (until and day > until):
                continue
            out.append(f)
        return out

    def download_ohlcv(self, tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Batch-download daily OHLCV once for every ticker in the replay."""
        _logger.info("Downloading OHLCV for %d tickers (%s → %s)", len(tickers), start.date(), end.date())
        batch = YahooDataDownloader().get_ohlcv_batch(tickers, "1d", start, end)
        out: Dict[str, pd.DataFrame] = {}
        for ticker, df in batch.items():
            if df is None or df.empty:
                continue
            out[ticker] = MarketAgent._to_technical_format(MarketAgent._set_datetime_index(df))
        _logger.info("OHLCV available for %d/%d tickers", len(out), len(tickers))
        return out

    def score_day(self, rows: List[Dict[Any, Any]], day: pd.Timestamp, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Re-score one run date with both the gated and the legacy scoring config.

        Returns:
            One row per candidate with technical features, new/legacy tier flags and forward outcomes.
        """
        window_start = day - timedelta(days=OHLCV_WINDOW_DAYS)
        sliced = {
            t: df.loc[(df.index >= window_start) & (df.index <= day)]
            for t, df in ohlcv.items()
        }
        sliced = {t: df for t, df in sliced.items() if len(df) >= 20}

        new = [candidate_from_row(r) for r in rows]
        legacy = [candidate_from_row(r) for r in rows]
        self._technical.run(new, sliced)
        self._technical.run(legacy, sliced)
        self._scoring.run(new)
        self._scoring_legacy.run(legacy)
        legacy_by_ticker = {c.ticker: c for c in legacy}
        prod_by_ticker = {str(r["ticker"]): r for r in rows}

        records = []
        for c in new:
            if c.ticker not in sliced:
                continue  # no point-in-time OHLCV → cannot replay this name
            fwd = forward_outcomes(ohlcv[c.ticker], day)
            if fwd is None:
                continue
            prod = prod_by_ticker[c.ticker]
            lc = legacy_by_ticker[c.ticker]
            records.append({
                "ticker": c.ticker,
                "run_date": day,
                "price": c.price,
                "price_5d_return": c.price_5d_return,
                "price_20d_return": c.price_20d_return,
                "relative_volume": c.relative_volume,
                "dist_to_50d_high": c.dist_to_50d_high,
                "extension_atr": c.extension_atr,
                "range_10d_pct": c.range_10d_pct,
                "final_score": c.final_score,
                "setup_score": c.setup_score,
                "tier": c.tier,
                "extended": c.extended,
                "prod_ab": prod.get("tier") in ("A", "B") or bool(prod.get("explosive_candidate")),
                "replay_legacy": lc.tier in ("A", "B") or lc.explosive_candidate,
                "replay_ab": c.tier in ("A", "B") or c.explosive_candidate,
                "setup": c.setup_candidate,
                **fwd,
            })
        return pd.DataFrame(records)

    def run(self, since: Optional[str], until: Optional[str], cooldown_days: int) -> Dict[str, pd.DataFrame]:
        """
        Replay every run in range.

        Returns:
            ``rows`` (all scored candidate-days), ``episodes`` (deduplicated signals,
            tagged by set) and ``summary`` (one row per set).
        """
        files = self.run_files(since, until)
        if not files:
            raise FileNotFoundError(f"No candidates CSVs under {self.results_dir}")
        day_rows = {pd.Timestamp(os.path.basename(os.path.dirname(f))): pd.read_csv(f).to_dict("records") for f in files}
        tickers = sorted({str(r["ticker"]) for rows in day_rows.values() for r in rows})
        first_day = min(day_rows)
        ohlcv = self.download_ohlcv(
            tickers, (first_day - timedelta(days=OHLCV_WINDOW_DAYS + 10)).to_pydatetime(), datetime.now(UTC)
        )

        frames = []
        for day, rows in sorted(day_rows.items()):
            frames.append(self.score_day(rows, day, ohlcv))
            _logger.debug("Replayed %s (%d rows)", day.date(), len(rows))
        all_rows = pd.concat([f for f in frames if not f.empty], ignore_index=True)

        episodes, summary = [], [summarise(all_rows, "universe")]
        for name in ("prod_ab", "replay_legacy", "replay_ab", "setup"):
            ep = dedupe_episodes(all_rows.loc[all_rows[name]], cooldown_days).assign(set=name)
            episodes.append(ep)
            summary.append(summarise(ep, name))
        return {
            "rows": all_rows,
            "episodes": pd.concat(episodes, ignore_index=True),
            "summary": pd.DataFrame(summary),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P17 point-in-time replay with forward returns")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Dir with {date}/{date}_candidates.csv")
    parser.add_argument("--since", default=None, help="First run date YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="Last run date YYYY-MM-DD")
    parser.add_argument("--cooldown-days", type=int, default=14, help="Episode separation (default: %(default)s)")
    parser.add_argument("--out-dir", default=None, help="Where to write replay CSVs (default: <results-dir>/replay)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir or os.path.join(args.results_dir, "replay"))
    out_dir.mkdir(parents=True, exist_ok=True)

    result = P17Replay(args.results_dir).run(args.since, args.until, args.cooldown_days)
    result["rows"].to_csv(out_dir / "replay_rows.csv", index=False)
    result["episodes"].to_csv(out_dir / "replay_episodes.csv", index=False)
    result["summary"].to_csv(out_dir / "replay_summary.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        _logger.info("P17 replay summary (entry = next open):\n%s", result["summary"].to_string(index=False))
    _logger.info("Replay CSVs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
