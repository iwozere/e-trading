"""
Signal Ledger — per-stream statistics: is a signal stream profitable after costs?

Per stream and horizon:
- excess return per signal = side * (ticker return - benchmark return) - round-trip cost;
- signals with a ``data_flag`` (suspected unadjusted split) are left out;
- signals of the same date are averaged into one cohort (same-day picks are not
  independent), and the cohort series is tested with a Newey-West t-statistic
  using ``h - 1`` lags (holding windows of consecutive cohorts overlap).

Verdict: ``edge`` when t >= 2 and mean > 0 over at least ``MIN_COHORTS`` dates,
``negative`` when t <= -2, ``insufficient`` below ``MIN_COHORTS`` dates, else ``no_edge``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.ml.pipeline.signal_ledger.config import HORIZONS, cost_bps
from src.ml.pipeline.signal_ledger.evaluator import return_column

MIN_COHORTS = 20
T_THRESHOLD = 2.0

# Extra streams carved out of a source by a predicate on the row (stream name -> (source, predicate)).
Predicate = Callable[[Mapping[str, Any]], bool]
SUB_STREAMS: Dict[str, tuple[str, Predicate]] = {
    "p17_tier_ab": ("p17_candidates", lambda row: (row.get("meta") or {}).get("tier") in ("A", "B")),
}


def newey_west_t(values: Sequence[float], lags: int) -> float:
    """
    Newey-West (Bartlett kernel) t-statistic of the mean.

    Args:
        values: Time-ordered observations.
        lags: Number of autocovariance lags.

    Returns:
        t-statistic, or NaN with fewer than 3 observations or zero variance.
    """
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    d = x - x.mean()
    variance = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        variance += 2.0 * (1.0 - k / (lags + 1)) * float(d[k:] @ d[:-k]) / n
    # Constant series leave float residue (~1e-34), not an exact zero.
    if variance <= 1e-12 * max(float(x @ x) / n, 1e-300):
        return float("nan")
    return float(x.mean() / math.sqrt(variance / n))


def verdict(t_stat: float, mean: float, cohorts: int) -> str:
    """Classify a stream/horizon from its t-stat, mean excess and cohort count."""
    if cohorts < MIN_COHORTS:
        return "insufficient"
    if math.isfinite(t_stat) and t_stat >= T_THRESHOLD and mean > 0:
        return "edge"
    if math.isfinite(t_stat) and t_stat <= -T_THRESHOLD:
        return "negative"
    return "no_edge"


def _streams(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Group rows by source, plus the predicate-defined sub-streams."""
    streams: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        streams.setdefault(row["source"], []).append(row)
    for name, (source, predicate) in SUB_STREAMS.items():
        subset = [row for row in streams.get(source, []) if predicate(row)]
        if subset:
            streams[name] = subset
    return streams


def _horizon_stats(rows: Sequence[Mapping[str, Any]], source: str, horizon: int, benchmark: str) -> Optional[Dict[str, Any]]:
    """Statistics for one stream at one horizon, or None when nothing has matured."""
    ret_col, bench_col = return_column("ret", horizon), return_column(benchmark, horizon)
    cost = cost_bps(source) / 10_000.0
    records = []
    for row in rows:
        ret, bench = row.get(ret_col), row.get(bench_col)
        if ret is None or bench is None or row.get("data_flag"):
            continue
        sign = -1.0 if row.get("side") == "short" else 1.0
        records.append((row["signal_date"], sign * ret, sign * (ret - bench) - cost))
    if not records:
        return None
    frame = pd.DataFrame(records, columns=["date", "raw", "excess"])
    cohorts = frame.groupby("date")["excess"].mean().sort_index()
    t_stat = newey_west_t(cohorts.tolist(), max(horizon - 1, 0))
    cohort_mean = float(cohorts.mean())
    return {
        "horizon": horizon,
        "signals": len(frame),
        "cohorts": int(len(cohorts)),
        "mean_raw": float(frame["raw"].mean()),
        "mean_excess": float(frame["excess"].mean()),
        "median_excess": float(frame["excess"].median()),
        "hit_rate": float((frame["excess"] > 0).mean()),
        "cohort_mean_excess": cohort_mean,
        "nw_t": None if math.isnan(t_stat) else t_stat,
        "verdict": verdict(t_stat, cohort_mean, len(cohorts)),
    }


def build_scorecard(rows: Sequence[Mapping[str, Any]], benchmark: str = "vt") -> List[Dict[str, Any]]:
    """
    Build the scorecard for every stream.

    Args:
        rows: Ledger rows (``source``, ``signal_date``, ``side``, ``meta`` and the ``ret_*``/benchmark columns).
        benchmark: Lower-case benchmark prefix (``"vt"``, ``"spy"``, ``"iwm"``).

    Returns:
        One dict per stream: ``stream``, ``source``, ``signals``, ``first_date``, ``last_date``,
        ``cost_bps`` and ``horizons`` (list of per-horizon stats).
    """
    scorecard = []
    for stream, stream_rows in sorted(_streams(rows).items()):
        source = stream_rows[0]["source"]
        dates = [row["signal_date"] for row in stream_rows]
        horizons = [s for h in HORIZONS if (s := _horizon_stats(stream_rows, source, h, benchmark)) is not None]
        scorecard.append({
            "stream": stream,
            "source": source,
            "signals": len(stream_rows),
            "first_date": min(dates),
            "last_date": max(dates),
            "cost_bps": cost_bps(source),
            "benchmark": benchmark.upper(),
            "horizons": horizons,
        })
    return scorecard
