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

Usage:
    python -m src.ml.pipeline.p06_emps2.util.simulate_stops
    python -m src.ml.pipeline.p06_emps2.util.simulate_stops --extended   # includes baselines
"""

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RESULTS_BASE    = PROJECT_ROOT / "results" / "p06_emps2"
TIMING_CSV      = RESULTS_BASE / "timing_analysis.csv"
OUTPUT_CSV      = RESULTS_BASE / "stop_simulation.csv"
SUMMARY_CSV     = RESULTS_BASE / "stop_simulation_summary.csv"
CACHE_DIR       = RESULTS_BASE / "_ohlcv_cache"

_SEARCH_ROOTS   = [RESULTS_BASE, RESULTS_BASE / "p06_emps2"]


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """
    One stop-loss configuration.

    ATR-based stop:
        If atr_multiplier > 0, per-trade stop_pct = atr_multiplier × atr_ratio.
        stop_pct is used as a fallback when ATR data is unavailable.

    Breakeven trailing:
        When the trade reaches +breakeven_trigger_pct, the stop floor is raised to entry.
        From that point the stop trails at breakeven_trail_pct below the high watermark
        (if breakeven_trail_pct > 0) or stays fixed at entry (if 0).
    """
    label: str
    stop_pct: float                      # fallback / fixed stop (e.g. 0.08 = 8%)
    trailing: bool = False               # plain trailing stop from entry
    max_hold_days: int = 20
    atr_multiplier: float = 0.0          # > 0 → ATR-based stop
    breakeven_trigger_pct: float = 0.0   # > 0 → activate breakeven once price +X%
    breakeven_trail_pct: float = 0.0     # trail from high after breakeven (0 = fixed at entry)

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
        df = yf.Ticker(ticker).history(
            start=str(start), end=str(end), auto_adjust=True
        )
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.to_csv(cache_file)
        return df
    except Exception:
        _logger.warning("Could not fetch OHLCV for %s (%s – %s)", ticker, start, end)
        return pd.DataFrame()


def _atr_ratio_for_alert(ticker: str, alert_date: date) -> Optional[float]:
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

def _as_date(ts) -> Optional[date]:
    if ts is None:
        return None
    return ts.date() if hasattr(ts, "date") else ts


def _simulate_trade(
    ohlcv: pd.DataFrame,
    entry_price: float,
    cfg: SimConfig,
    effective_stop_pct: float,        # pre-computed (may differ from cfg.stop_pct for ATR-based)
) -> tuple[Optional[date], Optional[float], str, int]:
    """
    Simulate a single trade from day-after-entry through the OHLCV window.

    Stop logic (evaluated each day in this order):
      1. Compute today's effective stop price
      2. Gap-down check: if open <= stop → fill at open
      3. Intraday check: if low <= stop → fill at stop price
      4. EOD: update high watermark; check whether breakeven trigger was hit
      5. Time limit: close out at closing price

    Returns (exit_date, exit_price, exit_reason, hold_days).
    """
    high_watermark = entry_price
    stop_price = entry_price * (1.0 - effective_stop_pct)
    breakeven_active = False

    for day_idx, (ts, row) in enumerate(ohlcv.iterrows()):
        day_open  = float(row["Open"])
        day_high  = float(row["High"])
        day_low   = float(row["Low"])
        day_close = float(row["Close"])
        exit_date = _as_date(ts)

        # ── Recompute stop for today ──────────────────────────────────────────
        if breakeven_active:
            if cfg.breakeven_trail_pct > 0:
                # Trail from high watermark, floor at entry (no loss possible)
                stop_price = max(entry_price, high_watermark * (1.0 - cfg.breakeven_trail_pct))
            else:
                # Fixed at entry (absolute breakeven)
                stop_price = entry_price
        elif cfg.trailing:
            stop_price = high_watermark * (1.0 - effective_stop_pct)
        # else: fixed stop stays at initial stop_price

        # ── Gap-down protection ───────────────────────────────────────────────
        if day_open <= stop_price:
            return (exit_date, day_open, "gap_stop", day_idx)

        # ── Intraday stop ─────────────────────────────────────────────────────
        if day_low <= stop_price:
            return (exit_date, stop_price, "stop_hit", day_idx)

        # ── EOD: update high watermark ────────────────────────────────────────
        if day_high > high_watermark:
            high_watermark = day_high

        # ── Check breakeven trigger (based on today's high) ───────────────────
        if not breakeven_active and cfg.breakeven_trigger_pct > 0:
            if high_watermark >= entry_price * (1.0 + cfg.breakeven_trigger_pct):
                breakeven_active = True
                _logger.debug("Breakeven triggered for %s on day %d", exit_date, day_idx)

        # ── Time-based exit ───────────────────────────────────────────────────
        if day_idx + 1 >= cfg.max_hold_days:
            return (exit_date, day_close, "time_exit", day_idx + 1)

    # Exhausted data before time limit
    if len(ohlcv) > 0:
        last = ohlcv.iloc[-1]
        return (_as_date(last.name), float(last["Close"]), "data_end", len(ohlcv))
    return (None, None, "no_data", 0)


# ── Per-alert simulation ───────────────────────────────────────────────────────

@dataclass
class TradeResult:
    ticker: str
    alert_date: date
    entry_date: Optional[date]
    entry_price: Optional[float]
    exit_date: Optional[date]
    exit_price: Optional[float]
    exit_reason: str
    hold_days: int
    return_pct: Optional[float]
    config_key: str
    effective_stop_pct: Optional[float]   # actual stop used (useful for ATR-based)


def simulate_alert(
    ticker: str,
    alert_date: date,
    alert_price: float,
    cfg: SimConfig,
    atr_ratio: Optional[float] = None,
) -> TradeResult:
    """
    Simulate one trade: enter next-day open, apply cfg, exit at stop or time limit.
    """
    # Determine effective stop percentage
    if cfg.atr_multiplier > 0 and atr_ratio is not None:
        effective_stop_pct = cfg.atr_multiplier * atr_ratio
    else:
        effective_stop_pct = cfg.stop_pct

    fetch_start = alert_date + timedelta(days=1)
    fetch_end   = alert_date + timedelta(days=cfg.max_hold_days + 35)
    ohlcv = _fetch_ohlcv(ticker, fetch_start, fetch_end)

    if ohlcv.empty:
        return TradeResult(ticker=ticker, alert_date=alert_date,
                           entry_date=None, entry_price=None,
                           exit_date=None, exit_price=None,
                           exit_reason="no_data", hold_days=0,
                           return_pct=None, config_key=cfg.key,
                           effective_stop_pct=effective_stop_pct)

    entry_date  = _as_date(ohlcv.iloc[0].name)
    entry_price = float(ohlcv.iloc[0]["Open"])

    trading_window = ohlcv.iloc[1:]    # skip the entry day itself
    if trading_window.empty:
        return TradeResult(ticker=ticker, alert_date=alert_date,
                           entry_date=entry_date, entry_price=entry_price,
                           exit_date=entry_date, exit_price=entry_price,
                           exit_reason="no_data", hold_days=0,
                           return_pct=0.0, config_key=cfg.key,
                           effective_stop_pct=effective_stop_pct)

    exit_date, exit_price, exit_reason, hold_days = _simulate_trade(
        trading_window, entry_price, cfg, effective_stop_pct
    )

    return_pct: Optional[float] = None
    if exit_price is not None and entry_price > 0:
        return_pct = (exit_price / entry_price - 1.0) * 100.0

    return TradeResult(ticker=ticker, alert_date=alert_date,
                       entry_date=entry_date, entry_price=entry_price,
                       exit_date=exit_date, exit_price=exit_price,
                       exit_reason=exit_reason, hold_days=hold_days,
                       return_pct=return_pct, config_key=cfg.key,
                       effective_stop_pct=effective_stop_pct)


def simulate_baseline(
    ticker: str,
    alert_date: date,
    alert_price: float,
    max_hold_days: int,
) -> TradeResult:
    """Buy next-day open, hold exactly max_hold_days, no stop."""
    fetch_start = alert_date + timedelta(days=1)
    fetch_end   = alert_date + timedelta(days=max_hold_days + 35)
    ohlcv = _fetch_ohlcv(ticker, fetch_start, fetch_end)

    key = f"no_stop_{max_hold_days}d"
    if ohlcv.empty:
        return TradeResult(ticker=ticker, alert_date=alert_date,
                           entry_date=None, entry_price=None,
                           exit_date=None, exit_price=None,
                           exit_reason="no_data", hold_days=0,
                           return_pct=None, config_key=key,
                           effective_stop_pct=None)

    entry_price = float(ohlcv.iloc[0]["Open"])
    entry_date  = _as_date(ohlcv.iloc[0].name)
    exit_row    = ohlcv.iloc[max_hold_days] if len(ohlcv) > max_hold_days else ohlcv.iloc[-1]
    exit_price  = float(exit_row["Close"])
    exit_date   = _as_date(exit_row.name)

    return TradeResult(ticker=ticker, alert_date=alert_date,
                       entry_date=entry_date, entry_price=entry_price,
                       exit_date=exit_date, exit_price=exit_price,
                       exit_reason="time_exit", hold_days=max_hold_days,
                       return_pct=(exit_price / entry_price - 1.0) * 100.0,
                       config_key=key, effective_stop_pct=None)


# ── Summary statistics ─────────────────────────────────────────────────────────

def _summarise(results: list[TradeResult], label: str) -> dict:
    valid = [r for r in results if r.return_pct is not None]
    if not valid:
        return {"label": label, "n": 0}

    returns  = pd.Series([r.return_pct for r in valid])
    stopped  = [r for r in valid if r.exit_reason in ("stop_hit", "gap_stop")]
    wins     = returns[returns > 0]
    losses   = returns[returns <= 0]
    stop_ret = pd.Series([r.return_pct for r in stopped]) if stopped else pd.Series(dtype=float)

    expectancy = (
        (len(wins) / len(returns)) * float(wins.mean())
        + (len(losses) / len(returns)) * float(losses.mean())
    ) if len(returns) else 0.0

    crash_15  = int((returns < -15).sum())
    crash_25  = int((returns < -25).sum())

    atr_used  = [r.effective_stop_pct for r in valid if r.effective_stop_pct]
    avg_stop_used = float(pd.Series(atr_used).mean() * 100) if atr_used else None

    return {
        "label":                        label,
        "n":                            len(valid),
        "win_rate_pct":                 round(float((returns > 0).mean() * 100), 1),
        "mean_return_pct":              round(float(returns.mean()), 2),
        "median_return_pct":            round(float(returns.median()), 2),
        "avg_win_pct":                  round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_pct":                 round(float(losses.mean()), 2) if len(losses) else None,
        "max_loss_pct":                 round(float(returns.min()), 2),
        "max_gain_pct":                 round(float(returns.max()), 2),
        "expectancy_pct":               round(expectancy, 2),
        "stop_hit_pct":                 round(len(stopped) / len(valid) * 100, 1),
        "avg_stop_return_pct":          round(float(stop_ret.mean()), 2) if len(stop_ret) else None,
        "crashes_gt_15pct":             crash_15,
        "crashes_gt_25pct":             crash_25,
        "avg_effective_stop_pct":       round(avg_stop_used, 1) if avg_stop_used else None,
    }


def _print_table(summaries: list[dict], title: str) -> None:
    _logger.info("")
    _logger.info("=== %s ===", title)
    _logger.info(
        "%-38s %4s %5s %7s %8s %6s %8s %9s %9s %9s",
        "Strategy", "n", "Win%", "Mean%", "Median%",
        "Stop%", "StopRet", "MaxLoss", "Crash>15", "Expect%"
    )
    _logger.info("-" * 110)
    for s in summaries:
        if s.get("n", 0) == 0:
            continue
        stop_pct   = f"{s['stop_hit_pct']:>5.1f}%" if s.get("stop_hit_pct") is not None else "   N/A"
        stop_ret   = f"{s['avg_stop_return_pct']:>+7.1f}%" if s.get("avg_stop_return_pct") is not None else "    N/A"
        crash15    = str(s.get("crashes_gt_15pct", "—"))
        _logger.info(
            "%-38s %4d %4.0f%% %+6.1f%% %+7.1f%%  %s %s  %+8.1f%%  %6s  %+8.1f%%",
            s["label"], s["n"],
            s["win_rate_pct"], s["mean_return_pct"], s["median_return_pct"],
            stop_pct, stop_ret, s["max_loss_pct"],
            crash15, s["expectancy_pct"]
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    extended = "--extended" in sys.argv
    max_hold = 20

    _logger.info("Loading %s", TIMING_CSV)
    df = pd.read_csv(TIMING_CSV).dropna(subset=["price_at_alert"])
    df["alert_date"] = pd.to_datetime(df["alert_date"]).dt.date
    _logger.info("%d alerts to simulate", len(df))

    # ── Configuration sets ─────────────────────────────────────────────────────
    baseline_cfgs: list[SimConfig] = []

    if extended:
        baseline_cfgs = [
            SimConfig("Fixed stop  5% / 20d",  stop_pct=0.05, max_hold_days=20),
            SimConfig("Fixed stop  8% / 20d",  stop_pct=0.08, max_hold_days=20),
            SimConfig("Fixed stop 10% / 20d",  stop_pct=0.10, max_hold_days=20),
            SimConfig("Fixed stop 15% / 20d",  stop_pct=0.15, max_hold_days=20),
            SimConfig("Trail stop  5% / 20d",  stop_pct=0.05, trailing=True, max_hold_days=20),
            SimConfig("Trail stop  8% / 20d",  stop_pct=0.08, trailing=True, max_hold_days=20),
            SimConfig("Trail stop 15% / 20d",  stop_pct=0.15, trailing=True, max_hold_days=20),
        ]

    # The two new strategies the user asked for:
    new_cfgs: list[SimConfig] = [
        # ATR-based stops (stop = K × ATR/price, adaptive per stock)
        SimConfig("ATR 2.0x stop / 20d",
                  stop_pct=0.08, atr_multiplier=2.0, max_hold_days=20),
        SimConfig("ATR 2.5x stop / 20d",
                  stop_pct=0.08, atr_multiplier=2.5, max_hold_days=20),
        SimConfig("ATR 3.0x stop / 20d",
                  stop_pct=0.08, atr_multiplier=3.0, max_hold_days=20),
        # Breakeven trailing (wide initial stop; lock profit at +10%; trail from high)
        SimConfig("Breakeven: 8% init, +10% lock, trail 8%",
                  stop_pct=0.08, max_hold_days=20,
                  breakeven_trigger_pct=0.10, breakeven_trail_pct=0.08),
        SimConfig("Breakeven: 8% init, +8%  lock, trail 8%",
                  stop_pct=0.08, max_hold_days=20,
                  breakeven_trigger_pct=0.08, breakeven_trail_pct=0.08),
        SimConfig("Breakeven: 10% init, +15% lock, trail 8%",
                  stop_pct=0.10, max_hold_days=20,
                  breakeven_trigger_pct=0.15, breakeven_trail_pct=0.08),
        SimConfig("Breakeven: 10% init, +10% lock, fixed",
                  stop_pct=0.10, max_hold_days=20,
                  breakeven_trigger_pct=0.10, breakeven_trail_pct=0.0),
    ]

    all_cfgs = baseline_cfgs + new_cfgs
    all_results: list[TradeResult] = []
    baseline_results: list[TradeResult] = []

    total = len(df)
    for i, row in df.iterrows():
        ticker      = str(row["ticker"])
        alert_date  = row["alert_date"]
        alert_price = float(row["price_at_alert"])

        if (int(i) + 1) % 25 == 0:  # type: ignore[arg-type]
            _logger.info("  %d / %d ...", int(i) + 1, total)  # type: ignore[arg-type]

        baseline_results.append(simulate_baseline(ticker, alert_date, alert_price, max_hold))

        # Look up ATR ratio once per alert (shared across ATR-based configs)
        atr_ratio = _atr_ratio_for_alert(ticker, alert_date)

        for cfg in all_cfgs:
            all_results.append(simulate_alert(ticker, alert_date, alert_price, cfg, atr_ratio))

    _logger.info("Simulation complete — building summaries")

    # Save per-trade detail
    rows = [
        {
            "ticker":               r.ticker,
            "alert_date":           r.alert_date,
            "entry_price":          r.entry_price,
            "exit_price":           r.exit_price,
            "exit_reason":          r.exit_reason,
            "hold_days":            r.hold_days,
            "return_pct":           r.return_pct,
            "effective_stop_pct":   r.effective_stop_pct,
            "config":               r.config_key,
        }
        for r in all_results
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    _logger.info("Per-trade detail → %s", OUTPUT_CSV)

    # Summaries
    summaries: list[dict] = [_summarise(baseline_results, f"No stop (hold {max_hold}d at open)")]
    for cfg in all_cfgs:
        cfg_results = [r for r in all_results if r.config_key == cfg.key]
        summaries.append(_summarise(cfg_results, cfg.label))

    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)
    _logger.info("Summary → %s", SUMMARY_CSV)

    # ── Print tables ───────────────────────────────────────────────────────────
    if extended:
        _print_table(
            summaries[:len(baseline_cfgs) + 1],
            "BASELINES (fixed / trailing stops)"
        )

    _print_table(
        [summaries[0]] + summaries[len(baseline_cfgs) + 1:],
        "NEW STRATEGIES: ATR-based + Breakeven"
    )

    # ── PREMIUM vs HIGH breakdown for best new configs ─────────────────────────
    if "pre_alert_gain_pct" in df.columns:
        meta = df[["ticker", "alert_date", "pre_alert_gain_pct"]].copy()
        meta["alert_date"] = meta["alert_date"].astype(str)

        premium_tickers = set(df[df["pre_alert_gain_pct"] < 0]["ticker"])
        _logger.info("")
        _logger.info("=== PREMIUM vs HIGH — selected configs ===")

        selected_keys = {
            "no_stop_20d",
            "atr3.0x_20d",
            "fixed8pct_be10_trail8_20d",   # Breakeven: 8% init, +10% lock, trail 8%
        }
        chosen_cfgs = [cfg for cfg in all_cfgs if cfg.key in selected_keys]
        chosen_summaries: list[dict] = []

        chosen_summaries.append(_summarise(
            [r for r in baseline_results if r.ticker in premium_tickers],
            "PREMIUM  no stop 20d"
        ))
        chosen_summaries.append(_summarise(
            [r for r in baseline_results if r.ticker not in premium_tickers],
            "HIGH     no stop 20d"
        ))

        for cfg in chosen_cfgs:
            prem = [r for r in all_results
                    if r.config_key == cfg.key and r.ticker in premium_tickers]
            high = [r for r in all_results
                    if r.config_key == cfg.key and r.ticker not in premium_tickers]
            chosen_summaries.append(_summarise(prem, f"PREMIUM  {cfg.label}"))
            chosen_summaries.append(_summarise(high, f"HIGH     {cfg.label}"))

        _print_table(chosen_summaries, "PREMIUM vs HIGH breakdown")

    _logger.info("")
    _logger.info("Crash prevention summary (trades losing > 15%% / > 25%%):")
    for s in summaries:
        if s.get("n", 0) == 0:
            continue
        _logger.info(
            "  %-38s  >15%%: %2d trades  >25%%: %2d trades  (max loss: %+.1f%%)",
            s["label"],
            s.get("crashes_gt_15pct", 0),
            s.get("crashes_gt_25pct", 0),
            s.get("max_loss_pct", 0),
        )


if __name__ == "__main__":
    main()
