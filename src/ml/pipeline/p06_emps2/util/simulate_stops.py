"""
Stop-Loss Simulation for EMPS2 Phase 2 Alerts

Simulates several stop-loss / trailing-stop strategies on historical Phase 2 alerts
to find the best approach for capital protection.

Entry model:
  - Entry price = next trading day's OPEN after the alert (realistic: alert received EOD,
    order placed pre-market next morning)
  - Stop check uses daily LOW; fills at stop price (or open if gapped below stop)
  - Exit at daily CLOSE when the max hold-day limit is reached

Strategies tested:
  Baseline  — no stop, hold N days
  Fixed     — fixed stop below entry, hold up to N days
  Trailing  — trailing stop from high watermark, hold up to N days
  ATR-based — stop = entry × (1 - K × atr_ratio), adapts to each stock's volatility
  Breakeven — wide initial stop; once price is up X%, move stop to entry, then trail Y%
  Improved  — Breakeven + dead-money early exit + tight trail for big winners

Usage:
    python -m src.ml.pipeline.p06_emps2.util.simulate_stops
    python -m src.ml.pipeline.p06_emps2.util.simulate_stops --extended   # includes baselines
"""

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RESULTS_BASE = PROJECT_ROOT / "results" / "p06_emps2"
TIMING_CSV = RESULTS_BASE / "timing_analysis.csv"
OUTPUT_CSV = RESULTS_BASE / "stop_simulation.csv"
SUMMARY_CSV = RESULTS_BASE / "stop_simulation_summary.csv"
CACHE_DIR = RESULTS_BASE / "_ohlcv_cache"

_SEARCH_ROOTS = [RESULTS_BASE, RESULTS_BASE / "p06_emps2"]


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class SimConfig:
    """
    One stop-loss / exit configuration.

    ATR-based stop:
        If atr_multiplier > 0, per-trade stop_pct = atr_multiplier × atr_ratio.
        stop_pct is used as a fallback when ATR data is unavailable.

    Breakeven trailing:
        When the trade reaches +breakeven_trigger_pct, the stop floor is raised to entry.
        From that point the stop trails at breakeven_trail_pct below the high watermark
        (if breakeven_trail_pct > 0) or stays fixed at entry (if 0).

    Improved exits:
        dead_money_days > 0: at EOD of day N, if position is still within ±dead_money_pct
            of entry AND breakeven was never triggered, exit at close ("dead money").
            Use with max_hold_days > dead_money_days to let winners keep running.
        tight_trail_trigger_pct > 0: once price has risen past this threshold, switch to a
            tighter trailing stop (tight_trail_pct) to lock in more of the gain.
    """

    label: str
    stop_pct: float  # fallback / fixed stop (e.g. 0.08 = 8%)
    trailing: bool = False  # plain trailing stop from entry
    max_hold_days: int = 20
    atr_multiplier: float = 0.0  # > 0 → ATR-based stop
    breakeven_trigger_pct: float = 0.0  # > 0 → activate breakeven once price +X%
    breakeven_trail_pct: float = 0.0  # trail from high after breakeven (0 = fixed at entry)

    # Improved exit: cut dead-money positions early
    dead_money_days: int = 0  # > 0 → check for dead money at this day
    dead_money_pct: float = 0.03  # "near entry" = within ±3%

    # Improved exit: tighten trail once a big winner emerges
    tight_trail_trigger_pct: float = 0.0  # > 0 → once up this %, switch to tight trail
    tight_trail_pct: float = 0.0  # the tighter trailing % (e.g. 0.05)

    @property
    def key(self) -> str:
        parts = []
        if self.atr_multiplier > 0:
            parts.append(f"atr{self.atr_multiplier:.1f}x")
        elif self.trailing:
            parts.append(f"trail{int(self.stop_pct * 100)}pct")
        else:
            parts.append(f"fixed{int(self.stop_pct * 100)}pct")
        if self.breakeven_trigger_pct > 0:
            parts.append(f"be{int(self.breakeven_trigger_pct * 100)}")
            if self.breakeven_trail_pct > 0:
                parts.append(f"trail{int(self.breakeven_trail_pct * 100)}")
        if self.tight_trail_trigger_pct > 0:
            parts.append(f"tt{int(self.tight_trail_trigger_pct * 100)}")
            if self.tight_trail_pct > 0:
                parts.append(f"tttrail{int(self.tight_trail_pct * 100)}")
        if self.dead_money_days > 0:
            parts.append(f"dm{self.dead_money_days}")
        parts.append(f"{self.max_hold_days}d")
        return "_".join(parts)


# ── OHLCV fetching with local cache ───────────────────────────────────────────


def _fetch_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily OHLCV; cached in RESULTS_BASE/_ohlcv_cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(start=str(start), end=str(end), auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.to_csv(cache_file)
        return df
    except Exception:
        _logger.warning("Could not fetch OHLCV for %s (%s - %s)", ticker, start, end)
        return pd.DataFrame()


def _atr_ratio_for_alert(ticker: str, alert_date: date) -> float | None:
    """Look up ATR/price ratio from the volatility filter CSV on alert_date."""
    for root in _SEARCH_ROOTS:
        vol_file = root / str(alert_date) / "05_volatility_filtered.csv"
        if vol_file.exists():
            try:
                df = pd.read_csv(vol_file)
                row = df[df["ticker"] == ticker]
                if not row.empty and "atr_ratio" in df.columns:
                    return float(row.iloc[0]["atr_ratio"])
            except Exception:
                pass
    return None


# ── Core simulation ────────────────────────────────────────────────────────────


def _as_date(ts) -> date | None:
    if ts is None:
        return None
    return ts.date() if hasattr(ts, "date") else ts


def _simulate_trade(
    ohlcv: pd.DataFrame,
    entry_price: float,
    cfg: SimConfig,
    effective_stop_pct: float,
) -> tuple[date | None, float | None, str, int]:
    """
    Simulate a single trade from day-after-entry through the OHLCV window.

    Stop logic (evaluated each day in this order):
      1. Recompute today's effective stop price (including tight-trail override)
      2. Gap-down check: if open <= stop → fill at open
      3. Intraday check: if low <= stop → fill at stop price
      4. EOD: update high watermark; check breakeven / tight-trail triggers
      5. Dead-money check: exit at close if flat on dead_money_days
      6. Time limit: close out at close price

    Returns (exit_date, exit_price, exit_reason, hold_days).
    """
    high_watermark = entry_price
    stop_price = entry_price * (1.0 - effective_stop_pct)
    breakeven_active = False
    tight_trail_active = False

    for day_idx, (ts, row) in enumerate(ohlcv.iterrows()):
        day_open = float(row["Open"])
        day_high = float(row["High"])
        day_low = float(row["Low"])
        day_close = float(row["Close"])
        exit_date = _as_date(ts)

        # ── 1. Recompute stop for today ───────────────────────────────────────
        if breakeven_active:
            if cfg.breakeven_trail_pct > 0:
                stop_price = max(entry_price, high_watermark * (1.0 - cfg.breakeven_trail_pct))
            else:
                stop_price = entry_price
        elif cfg.trailing:
            stop_price = high_watermark * (1.0 - effective_stop_pct)
        # else: fixed stop — stop_price stays at its initial value

        # Tight trail overrides to a tighter level once the big-winner trigger fires
        if tight_trail_active and cfg.tight_trail_pct > 0:
            tight_stop = high_watermark * (1.0 - cfg.tight_trail_pct)
            stop_price = max(stop_price, tight_stop)

        # ── 2. Gap-down protection ────────────────────────────────────────────
        if day_open <= stop_price:
            return (exit_date, day_open, "gap_stop", day_idx)

        # ── 3. Intraday stop ──────────────────────────────────────────────────
        if day_low <= stop_price:
            return (exit_date, stop_price, "stop_hit", day_idx)

        # ── 4. EOD: update high watermark and check triggers ──────────────────
        if day_high > high_watermark:
            high_watermark = day_high

        if not breakeven_active and cfg.breakeven_trigger_pct > 0:
            if high_watermark >= entry_price * (1.0 + cfg.breakeven_trigger_pct):
                breakeven_active = True
                _logger.debug("Breakeven triggered for %s on day %d", exit_date, day_idx)

        if not tight_trail_active and cfg.tight_trail_trigger_pct > 0:
            if high_watermark >= entry_price * (1.0 + cfg.tight_trail_trigger_pct):
                tight_trail_active = True
                _logger.debug("Tight trail triggered for %s on day %d", exit_date, day_idx)

        # ── 5. Dead-money exit ────────────────────────────────────────────────
        # If price has gone nowhere by dead_money_days, redeploy capital early.
        # Only fires when breakeven was never triggered (i.e. no meaningful winner).
        if cfg.dead_money_days > 0 and day_idx + 1 == cfg.dead_money_days and not breakeven_active:
            if abs(day_close / entry_price - 1.0) <= cfg.dead_money_pct:
                return (exit_date, day_close, "dead_money", day_idx + 1)

        # ── 6. Time-based exit ────────────────────────────────────────────────
        if day_idx + 1 >= cfg.max_hold_days:
            return (exit_date, day_close, "time_exit", day_idx + 1)

    # Exhausted available data before time limit
    if len(ohlcv) > 0:
        last = ohlcv.iloc[-1]
        return (_as_date(last.name), float(last["Close"]), "data_end", len(ohlcv))
    return (None, None, "no_data", 0)


# ── Per-alert simulation ───────────────────────────────────────────────────────


@dataclass
class TradeResult:
    ticker: str
    alert_date: date
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    exit_reason: str
    hold_days: int
    return_pct: float | None
    config_key: str
    effective_stop_pct: float | None


def simulate_alert(
    ticker: str,
    alert_date: date,
    alert_price: float,
    cfg: SimConfig,
    atr_ratio: float | None = None,
) -> TradeResult:
    """Simulate one trade: enter next-day open, apply cfg, exit at stop or time limit."""
    effective_stop_pct = (
        cfg.atr_multiplier * atr_ratio if cfg.atr_multiplier > 0 and atr_ratio is not None else cfg.stop_pct
    )

    fetch_start = alert_date + timedelta(days=1)
    fetch_end = alert_date + timedelta(days=cfg.max_hold_days + 35)
    ohlcv = _fetch_ohlcv(ticker, fetch_start, fetch_end)

    if ohlcv.empty:
        return TradeResult(
            ticker=ticker,
            alert_date=alert_date,
            entry_date=None,
            entry_price=None,
            exit_date=None,
            exit_price=None,
            exit_reason="no_data",
            hold_days=0,
            return_pct=None,
            config_key=cfg.key,
            effective_stop_pct=effective_stop_pct,
        )

    entry_date = _as_date(ohlcv.iloc[0].name)
    entry_price = float(ohlcv.iloc[0]["Open"])
    trading_window = ohlcv.iloc[1:]

    if trading_window.empty:
        return TradeResult(
            ticker=ticker,
            alert_date=alert_date,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=entry_date,
            exit_price=entry_price,
            exit_reason="no_data",
            hold_days=0,
            return_pct=0.0,
            config_key=cfg.key,
            effective_stop_pct=effective_stop_pct,
        )

    exit_date, exit_price, exit_reason, hold_days = _simulate_trade(
        trading_window, entry_price, cfg, effective_stop_pct
    )

    return_pct: float | None = None
    if exit_price is not None and entry_price > 0:
        return_pct = (exit_price / entry_price - 1.0) * 100.0

    return TradeResult(
        ticker=ticker,
        alert_date=alert_date,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        hold_days=hold_days,
        return_pct=return_pct,
        config_key=cfg.key,
        effective_stop_pct=effective_stop_pct,
    )


def simulate_baseline(
    ticker: str,
    alert_date: date,
    alert_price: float,
    max_hold_days: int,
) -> TradeResult:
    """Buy next-day open, hold exactly max_hold_days, no stop."""
    fetch_start = alert_date + timedelta(days=1)
    fetch_end = alert_date + timedelta(days=max_hold_days + 35)
    ohlcv = _fetch_ohlcv(ticker, fetch_start, fetch_end)

    key = f"no_stop_{max_hold_days}d"
    if ohlcv.empty:
        return TradeResult(
            ticker=ticker,
            alert_date=alert_date,
            entry_date=None,
            entry_price=None,
            exit_date=None,
            exit_price=None,
            exit_reason="no_data",
            hold_days=0,
            return_pct=None,
            config_key=key,
            effective_stop_pct=None,
        )

    entry_price = float(ohlcv.iloc[0]["Open"])
    entry_date = _as_date(ohlcv.iloc[0].name)
    exit_row = ohlcv.iloc[max_hold_days] if len(ohlcv) > max_hold_days else ohlcv.iloc[-1]
    exit_price = float(exit_row["Close"])
    exit_date = _as_date(exit_row.name)

    return TradeResult(
        ticker=ticker,
        alert_date=alert_date,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason="time_exit",
        hold_days=max_hold_days,
        return_pct=(exit_price / entry_price - 1.0) * 100.0,
        config_key=key,
        effective_stop_pct=None,
    )


# ── Summary statistics ─────────────────────────────────────────────────────────


def _summarise(results: list[TradeResult], label: str) -> dict:
    valid = [r for r in results if r.return_pct is not None]
    if not valid:
        return {"label": label, "n": 0}

    returns = pd.Series([r.return_pct for r in valid])
    hold_days = pd.Series([r.hold_days for r in valid])
    stopped = [r for r in valid if r.exit_reason in ("stop_hit", "gap_stop")]
    dead_mon = [r for r in valid if r.exit_reason == "dead_money"]
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    stop_ret = pd.Series([r.return_pct for r in stopped]) if stopped else pd.Series(dtype=float)

    expectancy = (
        ((len(wins) / len(returns)) * float(wins.mean()) + (len(losses) / len(returns)) * float(losses.mean()))
        if len(returns)
        else 0.0
    )

    atr_used = [r.effective_stop_pct for r in valid if r.effective_stop_pct]

    return {
        "label": label,
        "n": len(valid),
        "avg_hold_days": round(float(hold_days.mean()), 1),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 1),
        "mean_return_pct": round(float(returns.mean()), 2),
        "median_return_pct": round(float(returns.median()), 2),
        "avg_win_pct": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_pct": round(float(losses.mean()), 2) if len(losses) else None,
        "max_loss_pct": round(float(returns.min()), 2),
        "max_gain_pct": round(float(returns.max()), 2),
        "expectancy_pct": round(expectancy, 2),
        "stop_hit_pct": round(len(stopped) / len(valid) * 100, 1),
        "dead_money_exit_pct": round(len(dead_mon) / len(valid) * 100, 1),
        "avg_stop_return_pct": round(float(stop_ret.mean()), 2) if len(stop_ret) else None,
        "crashes_gt_15pct": int((returns < -15).sum()),
        "crashes_gt_25pct": int((returns < -25).sum()),
        "avg_effective_stop_pct": round(float(pd.Series(atr_used).mean() * 100), 1) if atr_used else None,
    }


def _print_table(summaries: list[dict], title: str) -> None:
    _logger.info("")
    _logger.info("=== %s ===", title)
    _logger.info(
        "%-50s %4s %6s %5s %7s %8s %6s %8s %9s %9s",
        "Strategy",
        "n",
        "HoldD",
        "Win%",
        "Mean%",
        "Median%",
        "Stop%",
        "MaxLoss",
        "Crash>15",
        "Expect%",
    )
    _logger.info("-" * 120)
    for s in summaries:
        if s.get("n", 0) == 0:
            continue
        stop_pct = f"{s['stop_hit_pct']:>5.1f}%" if s.get("stop_hit_pct") is not None else "   N/A"
        crash15 = str(s.get("crashes_gt_15pct", "-"))
        _logger.info(
            "%-50s %4d %6.1f %4.0f%% %+6.1f%% %+7.1f%%  %s  %+8.1f%%  %6s  %+8.1f%%",
            s["label"],
            s["n"],
            s.get("avg_hold_days", 0),
            s["win_rate_pct"],
            s["mean_return_pct"],
            s["median_return_pct"],
            stop_pct,
            s["max_loss_pct"],
            crash15,
            s["expectancy_pct"],
        )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    extended = "--extended" in sys.argv

    _logger.info("Loading %s", TIMING_CSV)
    df = pd.read_csv(TIMING_CSV).dropna(subset=["price_at_alert"])
    df["alert_date"] = pd.to_datetime(df["alert_date"]).dt.date
    _logger.info("%d alerts to simulate", len(df))

    # ── Configuration sets ─────────────────────────────────────────────────────

    baseline_cfgs: list[SimConfig] = []
    if extended:
        baseline_cfgs = [
            SimConfig("Fixed stop  5% / 20d", stop_pct=0.05, max_hold_days=20),
            SimConfig("Fixed stop  8% / 20d", stop_pct=0.08, max_hold_days=20),
            SimConfig("Fixed stop 10% / 20d", stop_pct=0.10, max_hold_days=20),
            SimConfig("Fixed stop 15% / 20d", stop_pct=0.15, max_hold_days=20),
            SimConfig("Trail stop  5% / 20d", stop_pct=0.05, trailing=True, max_hold_days=20),
            SimConfig("Trail stop  8% / 20d", stop_pct=0.08, trailing=True, max_hold_days=20),
            SimConfig("Trail stop 15% / 20d", stop_pct=0.15, trailing=True, max_hold_days=20),
        ]

    atr_be_cfgs: list[SimConfig] = [
        SimConfig("ATR 2.0x stop / 20d", stop_pct=0.08, atr_multiplier=2.0, max_hold_days=20),
        SimConfig("ATR 2.5x stop / 20d", stop_pct=0.08, atr_multiplier=2.5, max_hold_days=20),
        SimConfig("ATR 3.0x stop / 20d", stop_pct=0.08, atr_multiplier=3.0, max_hold_days=20),
        SimConfig(
            "Breakeven: 8% init, +10% lock, trail 8%, 20d",
            stop_pct=0.08,
            max_hold_days=20,
            breakeven_trigger_pct=0.10,
            breakeven_trail_pct=0.08,
        ),
        SimConfig(
            "Breakeven: 8% init, +8%  lock, trail 8%, 20d",
            stop_pct=0.08,
            max_hold_days=20,
            breakeven_trigger_pct=0.08,
            breakeven_trail_pct=0.08,
        ),
        SimConfig(
            "Breakeven: 10% init, +15% lock, trail 8%, 20d",
            stop_pct=0.10,
            max_hold_days=20,
            breakeven_trigger_pct=0.15,
            breakeven_trail_pct=0.08,
        ),
        SimConfig(
            "Breakeven: 10% init, +10% lock, fixed, 20d",
            stop_pct=0.10,
            max_hold_days=20,
            breakeven_trigger_pct=0.10,
            breakeven_trail_pct=0.0,
        ),
    ]

    # Improved exit strategies — each builds on the previous
    improved_cfgs: list[SimConfig] = [
        # Improvement 1: reduce ceiling to 10 days
        SimConfig(
            "Impr-1: BE 8%/+10%/trail8%, 10d ceiling",
            stop_pct=0.08,
            max_hold_days=10,
            breakeven_trigger_pct=0.10,
            breakeven_trail_pct=0.08,
        ),
        # Improvement 2: dead-money exit at day 10, but let winners run to 20d
        SimConfig(
            "Impr-2: BE 8%/+10%/trail8%, 20d + dead-money d10",
            stop_pct=0.08,
            max_hold_days=20,
            breakeven_trigger_pct=0.10,
            breakeven_trail_pct=0.08,
            dead_money_days=10,
            dead_money_pct=0.03,
        ),
        # Improvement 3: add tight trail at +15% (5% trail instead of 8%)
        SimConfig(
            "Impr-3: BE 8%/+10%/trail8%, 20d + dm10 + tt+15%->5%",
            stop_pct=0.08,
            max_hold_days=20,
            breakeven_trigger_pct=0.10,
            breakeven_trail_pct=0.08,
            dead_money_days=10,
            dead_money_pct=0.03,
            tight_trail_trigger_pct=0.15,
            tight_trail_pct=0.05,
        ),
    ]

    all_cfgs = baseline_cfgs + atr_be_cfgs + improved_cfgs
    all_results: list[TradeResult] = []
    baseline20_results: list[TradeResult] = []
    baseline10_results: list[TradeResult] = []

    total = len(df)
    for i, row in df.iterrows():
        ticker = str(row["ticker"])
        alert_date = row["alert_date"]
        alert_price = float(row["price_at_alert"])

        if (int(i) + 1) % 25 == 0:  # type: ignore[arg-type]
            _logger.info("  %d / %d ...", int(i) + 1, total)  # type: ignore[arg-type]

        baseline20_results.append(simulate_baseline(ticker, alert_date, alert_price, 20))
        baseline10_results.append(simulate_baseline(ticker, alert_date, alert_price, 10))

        atr_ratio = _atr_ratio_for_alert(ticker, alert_date)
        for cfg in all_cfgs:
            all_results.append(simulate_alert(ticker, alert_date, alert_price, cfg, atr_ratio))

    _logger.info("Simulation complete - building summaries")

    # ── Save per-trade detail ──────────────────────────────────────────────────
    rows = [
        {
            "ticker": r.ticker,
            "alert_date": r.alert_date,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "exit_reason": r.exit_reason,
            "hold_days": r.hold_days,
            "return_pct": r.return_pct,
            "effective_stop_pct": r.effective_stop_pct,
            "config": r.config_key,
        }
        for r in all_results
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    _logger.info("Per-trade detail -> %s", OUTPUT_CSV)

    # ── Summaries ──────────────────────────────────────────────────────────────
    summaries: list[dict] = [
        _summarise(baseline20_results, "No stop, hold 20d"),
        _summarise(baseline10_results, "No stop, hold 10d"),
    ]
    for cfg in all_cfgs:
        cfg_results = [r for r in all_results if r.config_key == cfg.key]
        summaries.append(_summarise(cfg_results, cfg.label))

    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)
    _logger.info("Summary -> %s", SUMMARY_CSV)

    # ── Print: baselines ───────────────────────────────────────────────────────
    if extended:
        _print_table(summaries[: len(baseline_cfgs) + 2], "BASELINES (fixed / trailing stops)")

    # ── Print: ATR + breakeven (previous session's results) ───────────────────
    atr_be_summaries = [summaries[0]] + [s for s in summaries if any(s["label"] == cfg.label for cfg in atr_be_cfgs)]
    _print_table(atr_be_summaries, "ATR-based + Breakeven (from previous run)")

    # ── Print: improved exit strategies ───────────────────────────────────────
    # Include the old best from atr_be_cfgs as a reference row (re-label for clarity)
    old_best_raw = next((s for s in summaries if s.get("label") == atr_be_cfgs[3].label), {})
    old_best_row = dict(old_best_raw)
    old_best_row["label"] = "Old best: BE 8%/+10%/trail8%, 20d"

    impr_summaries = [
        _summarise(baseline20_results, "Baseline: no stop, 20d"),
        _summarise(baseline10_results, "Baseline: no stop, 10d"),
        old_best_row,
    ] + [s for s in summaries if any(s["label"] == cfg.label for cfg in improved_cfgs)]
    _print_table(impr_summaries, "IMPROVED EXIT STRATEGIES (incremental)")

    # ── PREMIUM vs HIGH breakdown for improved configs ─────────────────────────
    if "pre_alert_gain_pct" in df.columns:
        premium_idx = set(df[df["pre_alert_gain_pct"] < 0].index)

        def _prem(results: list[TradeResult]) -> list[TradeResult]:
            # match by position in df; use ticker+date for robustness
            p_set = set(
                zip(
                    df.loc[list(premium_idx), "ticker"].tolist(),
                    df.loc[list(premium_idx), "alert_date"].tolist(),
                )
            )
            return [r for r in results if (r.ticker, r.alert_date) in p_set]

        def _high(results: list[TradeResult]) -> list[TradeResult]:
            h_set = set(
                zip(
                    df.loc[~df.index.isin(premium_idx), "ticker"].tolist(),
                    df.loc[~df.index.isin(premium_idx), "alert_date"].tolist(),
                )
            )
            return [r for r in results if (r.ticker, r.alert_date) in h_set]

        prem20 = _prem(baseline20_results)
        high20 = _high(baseline20_results)

        prem_high_summaries: list[dict] = [
            _summarise(prem20, "PREMIUM  no stop 20d"),
            _summarise(high20, "HIGH     no stop 20d"),
        ]

        # Show old best and Impr-3 for PREMIUM vs HIGH
        for cfg in [atr_be_cfgs[3], improved_cfgs[2]]:  # old best + Impr-3
            cfg_res = [r for r in all_results if r.config_key == cfg.key]
            prem_high_summaries.append(_summarise(_prem(cfg_res), f"PREMIUM  {cfg.label}"))
            prem_high_summaries.append(_summarise(_high(cfg_res), f"HIGH     {cfg.label}"))

        _print_table(prem_high_summaries, "PREMIUM vs HIGH — old best vs Impr-3")

    # ── Crash prevention summary ───────────────────────────────────────────────
    _logger.info("")
    _logger.info("Crash prevention (trades losing >15%% / >25%%):")
    for s in [summaries[0], summaries[1]] + [
        s for s in summaries if any(s["label"] == cfg.label for cfg in improved_cfgs)
    ]:
        if s.get("n", 0) == 0:
            continue
        _logger.info(
            "  %-50s  >15%%: %2d  >25%%: %2d  (max: %+.1f%%)",
            s["label"],
            s.get("crashes_gt_15pct", 0),
            s.get("crashes_gt_25pct", 0),
            s.get("max_loss_pct", 0),
        )


if __name__ == "__main__":
    main()
