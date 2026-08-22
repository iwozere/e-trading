"""
P21 Momentum — scheduler entry point: daily mark (docs/pipeline-specification.md §11).

1. Fetch closes for holdings + MTUM + SPY + ^GSPC + ^VIX
2. Append today's NAV row to _state/nav_daily.csv; write daily_mark.json
3. Update high_water_price per position
4. Catastrophic stop check: adj close < avg_cost * 0.65 -> flag, execute at next open
5. Anomaly check: |close_t / close_t-1 - 1| > 0.35 -> flag MANUAL_REVIEW, do not auto-trade

Steps 6-7 of the original design (manual split adjustment, dividend
crediting) are dropped — see spec §10.1.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p21_momentum.calendar import is_trading_day
from src.ml.pipeline.p21_momentum.config import (
    ANOMALY_DAILY_CHANGE_PCT,
    CATASTROPHIC_STOP_PCT,
    NAV_TOTAL_USD,
    PROXY_TICKER,
    REGIME_INDEX_TICKER,
    REGIME_VOL_TICKER,
    RESULTS_DIR,
    STATE_DIR,
)
from src.ml.pipeline.p21_momentum.data.prices import fetch_price_panel
from src.ml.pipeline.p21_momentum.execution.ledger import read_current_positions, write_current_positions
from src.ml.pipeline.p21_momentum.jobs.run_common import send_abort_alert, setup_run_logging
from src.ml.pipeline.p21_momentum.quality.gates import GateOutcome, PipelineAbort, check_daily_price_change
from src.ml.pipeline.p21_momentum.results.run_io import already_processed, append_nav_row, write_daily_mark
from src.ml.pipeline.p21_momentum.schemas import DailyMarkSnapshot
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run(
    run_date: Optional[date] = None,
    force: bool = False,
    results_dir: Path = RESULTS_DIR,
    state_dir: Path = STATE_DIR,
    current_positions_path: Optional[Path] = None,
) -> Dict:
    """
    Execute one daily_mark cycle.

    Args:
        run_date: Override "today" (for tests / manual backfill).
        force: Bypass the idempotency check.
        results_dir, state_dir: Overridable for tests (real determinism/
            round-trip checks against a tmp dir) — production callers never
            pass these.
        current_positions_path: Overridable independently of state_dir;
            defaults to state_dir / "current_positions.json".

    Returns:
        Summary dict for __SCHEDULER_RESULT__.
    """
    today = run_date or date.today()
    if not is_trading_day(today):
        _logger.info("SKIP: %s is not an NYSE trading day", today)
        return {"skipped": True, "reason": "not_trading_day", "date": today.isoformat()}

    if already_processed(today, "daily_mark.json", results_dir=results_dir) and not force:
        _logger.info("SKIP: already processed for %s", today)
        return {"skipped": True, "reason": "already_processed", "date": today.isoformat()}

    positions_path = current_positions_path or (state_dir / "current_positions.json")
    try:
        return _run_daily_mark(today, results_dir, state_dir, positions_path)
    except PipelineAbort as exc:
        _logger.error("ABORT: %s", exc)
        send_abort_alert("daily_mark", exc)
        return {"aborted": True, "check": exc.check, "context": exc.context}


def _run_daily_mark(today: date, results_dir: Path, state_dir: Path, positions_path: Path) -> Dict:
    positions = read_current_positions(path=positions_path)
    tickers = sorted({p.ticker for p in positions} | {PROXY_TICKER, "SPY", REGIME_INDEX_TICKER, REGIME_VOL_TICKER})

    start_date = datetime.combine(today, datetime.min.time()) - timedelta(days=10)
    end_date = datetime.combine(today, datetime.min.time()) + timedelta(days=1)
    panel = fetch_price_panel(tickers, start_date, end_date)

    anomalies: List[Dict] = []
    catastrophic_stops: List[Dict] = []
    close_today: Dict[str, float] = {}

    for ticker, df in panel.items():
        if df is None or df.empty or len(df) < 2:
            continue
        df_sorted = df.sort_values("timestamp")
        closes = df_sorted["close"].to_numpy()
        c_today, c_prior = float(closes[-1]), float(closes[-2])
        close_today[ticker] = c_today

        if c_prior > 0:
            change_pct = c_today / c_prior - 1.0
            gate = check_daily_price_change(ticker, change_pct, ANOMALY_DAILY_CHANGE_PCT)
            if gate.outcome == GateOutcome.WARN:
                anomalies.append({"ticker": ticker, "change_pct": change_pct})

    updated_positions = []
    for p in positions:
        price = close_today.get(p.ticker)
        if price is None:
            updated_positions.append(p)
            continue
        p.high_water_price = max(p.high_water_price, price)
        if p.avg_cost > 0 and price < p.avg_cost * (1 + CATASTROPHIC_STOP_PCT):
            catastrophic_stops.append({"ticker": p.ticker, "price": price, "avg_cost": p.avg_cost})
            _logger.warning(
                "EXIT_CATASTROPHIC_STOP flagged for %s at %.2f (avg_cost %.2f)", p.ticker, price, p.avg_cost
            )
        updated_positions.append(p)

    sleeve_market_value = sum(p.shares * close_today.get(p.ticker, p.avg_cost) for p in updated_positions)
    prior_cash = _read_prior_cash(positions_path)
    cash = prior_cash if prior_cash is not None else (NAV_TOTAL_USD - sleeve_market_value)
    nav_a = cash + sleeve_market_value

    write_current_positions(
        updated_positions,
        as_of=today,
        track="A",
        nav_total=NAV_TOTAL_USD,
        cash=cash,
        sleeve_market_value=sleeve_market_value,
        regime_scalar=1.0,
        path=positions_path,
    )

    nav_row = {
        "date": today.isoformat(),
        "nav_a": nav_a,
        "nav_b": NAV_TOTAL_USD,
        "nav_c": NAV_TOTAL_USD,
        "nav_d": NAV_TOTAL_USD,
        "nav_e": NAV_TOTAL_USD,
    }
    append_nav_row(nav_row, state_dir / "nav_daily.csv")

    snapshot = DailyMarkSnapshot(
        as_of=today.isoformat(),
        nav={"A": nav_a, "B": NAV_TOTAL_USD, "C": NAV_TOTAL_USD, "D": NAV_TOTAL_USD, "E": NAV_TOTAL_USD},
        anomalies=anomalies,
        catastrophic_stops=catastrophic_stops,
    )
    write_daily_mark(today, snapshot, results_dir=results_dir)

    _logger.info(
        "daily_mark complete for %s: nav_a=%.2f, %d anomalies, %d catastrophic stops",
        today, nav_a, len(anomalies), len(catastrophic_stops),
    )
    return {
        "date": today.isoformat(),
        "nav_a": nav_a,
        "anomalies_count": len(anomalies),
        "catastrophic_stops_count": len(catastrophic_stops),
    }


def _read_prior_cash(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("cash")


def main() -> None:
    """Run daily_mark and print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("daily_mark result: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
