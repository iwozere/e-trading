# P24 Falling Knife Reversal

## Overview
Research pipeline for the "falling knife" reversal method (ADX mountain + PPO valley + CCI positive divergence after a
≥ 40% drop), from chapter 11 of *Explosive Stock Trading Strategies* (summary in `docs/11-chapter.html`). The chapter's
visual rules are turned into exact, testable rules (`docs/pipeline-specification.md` §4). Phase 1 is a historical
backtest on the FMP vintage that tests the book's ">80% winners" claim against control baselines. A live scan
(Phase 2) is built only after the go/no-go gate in spec §11.

## Features
- Causal detector: ZigZag reference peak, ADX mountain, PPO valley (with lower-third depth), normalized ADX/PPO
  channel width `W`, point-3 support and CCI divergence, one signal per knife with re-arm for a second leg.
- Exits: CCI ≥ +200, bearish CCI divergence, post-signal shakeout test against point 3, time stop, optional hard stop.
- Daily and weekly timeframes; costs tiered by as-traded price; delisting handled.
- Backtest with ablation variants, drop-only and matched-random controls, month-block bootstrap CIs, survivorship
  report, holdout guard.

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

## Integration
This module integrates with:
- `src.data.fmp_pipeline` — FMP vintage reader (`FmpStore`), `adjust()`, bar flags
- `src.notification` — logger (Phase 2: alerts)
- `src.ml.pipeline.signal_ledger` — Phase 2 harvester (not built yet)

## Configuration
All thresholds are in `config.py` (`DAILY_PARAMS`, `WEEKLY_PARAMS`, `UniverseParams`, `CostModel`, `BacktestWindow`).
The FMP data root defaults to `D:/data-cache/fmp-data` (`--data-root`).

## Related Documentation
- [Specification](docs/pipeline-specification.md) — the binding spec
- [Implementation plan](docs/implementation-plan.md)
- [Requirements](docs/Requirements.md) — Technical requirements
- [Design](docs/Design.md) — Architecture and design
- [Tasks](docs/Tasks.md) — Implementation roadmap
