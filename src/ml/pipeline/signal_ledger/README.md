# Signal Ledger

## Overview
One table of every buy/sell recommendation the active pipelines emitted, with forward returns measured afterwards, so each
signal stream can be judged on evidence: does it beat VT (where the money sits otherwise), after costs, with statistical
significance? The nightly job also simulates a paper book per stream with the chosen sizing (USD 2,000 per position, at most
40 positions, rest in VT).

Pipelines are **not** modified — the ledger harvests the files they already write to `results/<pipeline>/YYYY-MM-DD/`.

## Features
- Harvesters for P05 (`top_picks.csv`), P06 (phase-2 alerts, final universe), P10 (PASSED universe), P17 (candidates,
  tier in meta), P19 (T1–T3 momentum alerts from `shadow.sqlite`) and P21 (`targets.json`).
- Forward returns at 1/5/10/20 sessions: entry at the next session's open, exit at the close; VT/SPY/IWM over the same window.
- Scorecard per stream: mean/median excess after a per-stream cost, hit rate, Newey-West t of daily cohorts, verdict
  (`edge`, `negative`, `no_edge`, `insufficient`).
- Paper book per stream: daily NAV vs 100 % VT, invested share, open positions, max drawdown.
- Data guard: a 5× one-session price jump inside a holding window (usually an unadjusted reverse split) flags the signal
  (`data_flag = 'price_jump'`); flagged signals stay in the table but are left out of scorecard and books.
- Web UI: Monitoring → **Signals** tab (API `GET /api/signals/scorecard`, `/books/{source}/nav`, `/recent`).

## Quick Start
```bash
# one-off backfill of the whole results archive (idempotent)
python src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py --full

# nightly (scheduled as "Signal Ledger Nightly", 06:00 UTC Tue-Sat)
python src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py
```

```python
from src.data.db.services.signal_ledger_service import SignalLedgerService
from src.ml.pipeline.signal_ledger.scorecard import build_scorecard

rows = SignalLedgerService().list_signals()
for stream in build_scorecard(rows, benchmark="vt"):
    twenty = next((h for h in stream["horizons"] if h["horizon"] == 20), None)
    if twenty:
        _logger.info("%s: %+.2f%% (t=%s) %s", stream["stream"], 100 * twenty["mean_excess"], twenty["nw_t"], twenty["verdict"])
```

## Integration
This module integrates with:
- `src.data.db` - `signal_ledger` / `signal_book_nav` tables (migration `010_signal_ledger`), `SignalLedgerService`
- `src.data.downloader` - `YahooDataDownloader.get_ohlcv_batch` for adjusted daily bars
- `src.data.pipeline` - scheduler spec `specs/signal_ledger_specs.py`
- `src.api` - `signal_ledger_routes.py` serves the scorecard to the Web UI

## Configuration
All knobs are in `config.py`: horizons, benchmarks, book sizing (`START_CAPITAL_USD`, `POSITION_USD`, `MAX_POSITIONS`),
round-trip cost per source (`COST_BPS`), `JUMP_RATIO`, harvest window. State lives in `results/signal_ledger/_state/prices.parquet`;
every run writes `results/signal_ledger/YYYY-MM-DD/scorecard.json`.

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
