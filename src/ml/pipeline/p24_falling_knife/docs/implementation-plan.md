# P24 Falling Knife — Implementation Plan

Companion to `pipeline-specification.md` (the spec wins on any conflict). Each milestone ends green:
`.venv/Scripts/python.exe -m pytest src/ml/pipeline/p24_falling_knife/tests` and `scripts/typecheck.py` at 0 errors.
One commit per milestone, imperative message.

## M0 — Skeleton and config
- Folder layout per spec §10; empty `__init__.py`; `README.md`, `docs/Requirements.md`, `docs/Design.md`, `docs/Tasks.md`
  from the CLAUDE.md templates.
- `config.py`: frozen dataclasses `TimeframeParams` (every §4/§4.9 number) with `DAILY` and `WEEKLY` instances,
  `CostTier` list (§7.2), `BacktestWindow` (§14.1 dates), `P24Config` tying them together, plus `with_overrides(**kw)` for
  the calibration grid and ablations. No literals elsewhere.

## M1 — Indicators (`indicators.py`)
- `compute_indicators(df, params) -> DataFrame` adding `adx, plus_di, minus_di, ppo, ppo_signal, ppo_hist, cci`.
- ADX/DI/CCI from TA-Lib on float64 numpy; PPO with `ewm(adjust=False)`, in percent.
- Tests: PPO vs hand-computed EMA; ADX/CCI vs TA-Lib on a fixture; warm-up NaNs.

## M2 — Geometry (`geometry.py`)
- `zigzag(close, pct) -> list[Swing]` where `Swing(kind, bar, price, confirmed_bar)`: one O(n) pass, a swing exists only
  from `confirmed_bar` on.
- Per-bar helpers computed from the swing lists: `last_confirmed_high_before(t)`, `swing_lows_between(a, b, confirmed_by=t)`.
- `channel_width(adx, ppo, range_bars) -> (W, a, p, ppo_floor)` rolling, causal.
- Tests: causality under truncation, known zigzag fixture, W at extremes.

## M3 — Detector (`detector.py`)
- `evaluate(df, params, eligible_mask) -> ConditionTable`: one row per bar with every §4 boolean and metric
  (`drop, bar_p1, bar_l2, bar_x, bar_y, bar_wmin, s3, cci_2, cci_3, w_min, score, ...`).
- `signals(table, params) -> list[Signal]`: state machine with one signal per knife and the re-arm rule (§4.8).
- Ablation switches (`require_adx`, `require_ppo`, `require_channel`, `require_cci`, `ppo_lower_third`) live in params.
- Tests: synthetic knife (fires once; 35% drop / flat CCI / no support / wide W → none), S3 = support not bounce top,
  re-arm, 50-cut truncation test on real securities (marked to skip if the vintage is absent).

## M4 — Exits and trade simulation (`exits.py`, `costs.py`)
- `simulate_trade(df, table, signal, params, costs) -> Trade`: next-open entry, §4.9 priority list, §5.3 shakeout,
  delisting at last close, MAE/MFE, shakeout flags.
- `costs.per_side(price)` tiered, with a multiplier for 0.5×/2×/3× reports.
- Tests: one fixture per exit reason; fill price = next open × (1 ± cost).

## M5 — Data (`data.py`, `universe.py`)
- Load the active vintage through `FmpStore`; common-stock filter (reuse P22 `is_common_listing` logic); `bulk_daily` in
  chunks of ~500 ids; `adjust(mode="split")` with `events_many`; drop suspect-flagged bars.
- `to_weekly(daily)` (W-FRI, completed weeks only, by the XNYS calendar).
- Point-in-time eligibility mask (history, $1M median dollar volume, min price) and the suspect-gap flag (§3.3).
- `survivorship_report()` → `survivorship.csv` (§7.4).
- Tests: weekly resample incl. holiday week and incomplete week; suspect-gap flag; eligibility uses only past bars.

## M6 — Backtest runner (`backtest.py`, `jobs/run_p24_backtest.py`)
- Per security: load → indicators → condition table → signals → trades (full system + all ablation variants from one
  condition table, so the costly part runs once).
- Controls (§6.4): drop-only, matched-random (200 seeds), SPY excess per trade.
- Portfolio layer (§6.3) on the trade list.
- `ProcessPoolExecutor` with module-level worker; chunks of securities; results to `results/p24_falling_knife/backtest/<run_id>/`.
- CLI: `--timeframe daily|weekly|both`, `--window headline|upper_bound`, `--limit N` (smoke), `--securities ...`,
  `--holdout` (refuses to run unless the calibration file is frozen).
- First milestone check: default-parameter run on the 2021+ window, plus the MPWR / VMW / DIA / QQQ example check
  (§7.6) on the survivor window. Review numbers with the owner before calibration.

## M7 — Validation and report (`validation.py`, `report.py`, `chart.py`)
- Calibration grid on train (2021–2023) over the five §7.3 parameters, plateau pick, `calibration_grid.csv` + heatmap,
  frozen into `calibration_frozen.json`.
- Ablations, cost multiples, price buckets, month-concentration, block bootstrap CIs.
- `chart.py` 4-panel trade chart (also used by Phase 2); 20 sample charts in the report.
- Holdout run once → `report.md` with the §11 gate table. **Stop here for the owner gate.**

## M8 — Phase 2 (only after the gate)
- `scan.py` + `jobs/run_p24_scan.py` on DataManager + `universe_downloader`, `paper.py` state, notifications,
  signal-ledger harvester + test, `src/data/pipeline/specs/p24_specs.py` registered disabled.

## Effort notes
- Vintage 2021+ window: about 9–10k common stocks after filters, ~1,500 daily bars each including warm-up; the per-bar
  state machine is the hot path. Profile after M6; numba only if a full run exceeds ~20 min on the dev machine.
- The truncation test is the most expensive test; keep it at 20 securities × 50 cuts and mark it `slow`.
