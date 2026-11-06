# P24 Falling Knife Reversal

## Overview
Research pipeline for the "falling knife" reversal method (ADX mountain + PPO valley + CCI positive divergence after a
≥ 40% drop), from chapter 11 of *Explosive Stock Trading Strategies*. The chapter's
visual rules are turned into exact, testable rules (`docs/pipeline-specification.md` §4). Phase 1 is a historical
backtest on the FMP vintage that tests the book's ">80% winners" claim against control baselines. The 2024+ holdout
**failed** the go/no-go gate (no edge distinguishable from buying every 40% drop), so Phase 2 runs as a **paper +
notify** live scan only (owner decision 2026-11-06): daily and weekly scans, paper positions, Telegram/email alerts
with charts, and signal-ledger tracking. Both scheduler jobs are registered disabled.

## Features
- Causal detector: ZigZag reference peak, ADX mountain, PPO valley (with lower-third depth), normalized ADX/PPO
  channel width `W`, point-3 support and CCI divergence, one signal per knife with re-arm for a second leg.
- Exits: CCI ≥ +200, bearish CCI divergence, post-signal shakeout test against point 3, time stop, optional hard stop.
- Daily and weekly timeframes; costs tiered by as-traded price; delisting handled.
- Backtest with ablation variants, drop-only and matched-random controls, month-block bootstrap CIs, survivorship
  report, holdout guard; calibration grid with plateau pick.
- Live scan (`jobs/run_p24_scan.py`): signals on the last closed session/week, forming watchlist, replayed paper
  positions, one Telegram + email message per run with charts, outputs harvested by the signal ledger.

## Quick Start
```bash
# Smoke run on 300 securities, one process
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --limit 300 --workers 1 --draws 20

# Full default run (train + survivor-only upper-bound years; no holdout)
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --workers 7

# Specific symbols (any instrument type, e.g. the book's examples)
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --symbols MPWR VMW DIA QQQ --workers 1

# Calibration grid on train 2022–2023 (spec §7.3), then the frozen run with the holdout
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_calibrate.py --workers 7 --run-id <run_id>
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --workers 7 --holdout     --calibration results/p24_falling_knife/calibration/<run_id>/calibration_frozen.json
```
Calibration results land in `results/p24_falling_knife/calibration/<run_id>/` (`calibration_grid.csv`,
`calibration_summary.md`, `calibration_frozen.json`, `charts/heatmap_*.png`). `--repick <dir> --frozen-shakeout off`
re-picks from a saved grid without rerunning it.
Results land in `results/p24_falling_knife/backtest/<run_id>/` (`trades_daily.csv`, `trades_weekly.csv`,
`random_control_draws.csv`, `survivorship.csv`, `manifest.json`, `report.md`, `charts/samples/*.png`).

```python
from src.ml.pipeline.p24_falling_knife.config import DAILY_PARAMS
from src.ml.pipeline.p24_falling_knife.detector import build_context, detect

ctx = build_context(bars, DAILY_PARAMS)        # bars: OHLCV DataFrame on a DatetimeIndex
signals, _ = detect(ctx, DAILY_PARAMS)
```

### Live scan
```bash
# Dry run on a few symbols (no notification, scratch paper state)
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_scan.py --timeframe daily --symbols SONO MPWR \
    --no-notify --state /tmp/p24_paper.json
# Full daily scan as the scheduler runs it (it appends --user-id)
.venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_scan.py --timeframe daily
```
Without `--date` a run before 21:15 UTC scans the previous session (today's may still be trading). Outputs land in
`results/p24_falling_knife/YYYY-MM-DD/` (folder date = signal date): `signals_<tf>.csv`, `forming_<tf>.csv`,
`paper_positions_<tf>.csv`, `paper_exits_<tf>.csv`, `charts/*.png`, `run_summary_<tf>.json`, `pipeline_<tf>.log`.
Paper state: `results/p24_falling_knife/_state/paper_positions.json`.

**Deploy (Pi):** `git pull`, then `python -m src.data.pipeline.register_jobs` registers `P24 Falling Knife Daily
Scan` (22:45 UTC Mon–Fri) and `P24 Falling Knife Weekly Scan` (12:00 UTC Sat) **disabled**; enable them in the
scheduler when wanted. The first run fetches ~4 years of daily bars for ~5,000 symbols (cold cache, up to ~1–2 h).

## Integration
This module integrates with:
- `src.data.fmp_pipeline` — FMP vintage reader (`FmpStore`), `adjust()`, bar flags (backtest)
- `src.data.data_manager.DataManager` — live daily bars (Yahoo path; the FMP adapter is off for live jobs)
- `src.ml.pipeline.shared.universe_downloader` — NASDAQ Trader symbol directory (live universe)
- `src.data.db.services.notification_service` / `users_service` — Telegram + email messages
- `src.data.pipeline.specs.p24_specs` — scheduler job specs; `src.scheduler` injects `--user-id`
- `src.ml.pipeline.signal_ledger` — `harvest_p24` reads `signals_<tf>.csv` (sources `p24_daily`, `p24_weekly`)

## Configuration
All thresholds are in `config.py` (`DAILY_PARAMS`, `WEEKLY_PARAMS`, `UniverseParams`, `CostModel`, `BacktestWindow`).
The live scan uses `live_params(tf)` = defaults + `LIVE_OVERRIDES` (the frozen 2026-11-06 calibration) and
`LiveParams` (lookback, paper notional, forming threshold, charts per message).
The FMP data root defaults to `D:/data-cache/fmp-data` (`--data-root`).

## Related Documentation
- [Specification](docs/pipeline-specification.md) — the binding spec
- [Implementation plan](docs/implementation-plan.md)
- [Requirements](docs/Requirements.md) — Technical requirements
- [Design](docs/Design.md) — Architecture and design
- [Tasks](docs/Tasks.md) — Implementation roadmap
