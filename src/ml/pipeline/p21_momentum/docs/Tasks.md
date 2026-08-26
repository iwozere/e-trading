# Tasks

Derived from `implementation-plan.md` v1. Steps 0-18 (the live paper pipeline) are implemented and tested.
Step 19 (backtest harness) is implemented and tested (see below); the frozen-panel fetch + the full
2005-2026 study run against it are still an operator step, not yet executed. Step 20 (parameter freezing,
which depends on Phase 1 having run) is not started — see "Not Yet Started" below.

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Specification — `pipeline-specification.md` v1.2
- [x] Implementation plan — `implementation-plan.md` v1

**Step 0 — Scaffolding**
- [x] `config.py`, empty `__init__.py` files (CLAUDE.md §2.2)
- [x] `README.md`, `docs/Requirements.md`, `docs/Design.md`
- [x] Add `pandas_market_calendars` to `requirements.txt`
- [x] `results/p21_momentum/` — no gitignore change needed; already covered by the existing blanket
      `results/` entry (verified via `git check-ignore`)

**Step 1 — `calendar.py`**
- [x] `is_trading_day()`, `last_trading_day_of_month()`, `first_trading_day_of_month()`,
      `is_last_trading_day_of_month()`, `is_first_trading_day_of_month()`
- [x] Holiday-pinning unit tests (Thanksgiving 2026, Good Friday 2026) — `tests/test_calendar.py`, 9 tests

**Step 2 — Universe**
- [x] `get_sp500_constituents_with_sector()` added to `src/util/tickers_list.py` — smoke-tested against
      live Wikipedia (503 constituents, 11 GICS sectors)
- [x] `data/universe.py` wrapper + ticker normalization + `universe.json` builder
- [x] Survivorship-bias caveat documented in module docstring

**Step 3 — Data fetch wrappers**
- [x] `data/prices.py` (batch fetch, non-emptiness check -> `PipelineAbort`, fundamentals/sector TTL caches)
- [x] `data/earnings.py` (isolated direct-yfinance call)
- [x] `data/exclusions.py` (F5 reader, expiry filtering) + seeded `config/pipeline/p21_exclusions.json`

**Step 4 — `strategy/signal.py`** (§4)
- [x] `compute_signal()`
- [x] Test: MIN_HISTORY boundary, `vol < 0.05` guard
- [x] Test: ranks by `signal`, not `raw_return` (regression guard) + SKIP_RECENT window exclusion test

**Step 5 — `strategy/filters.py`** (§5)
- [x] F1 history, F2 liquidity, F3 gap, F4 quality, F5 exclusions, F6 earnings
- [x] `run_all()` composition + `tally_f4_missing_pct()` for the §12.6 D5 criterion
- [x] Test: F3's `total <= 0` short-circuit; F4's pass-on-missing-data; F1 short-circuits F2+

**Step 6 — `strategy/selection.py`** (§6)
- [x] 7-step operation order, exact sequence
- [x] Deterministic tie-break (`signal_desc, ticker_asc`) at rank and at sector-cap-drop
- [x] Test: underfill -> widen to top-40 -> `WARN_UNDERFILLED` path; sector cap never relaxed even underfilled
- [x] Bug found + fixed during testing: forced-exit tickers are now also excluded from the *fill* step, not
      just the retain step (defensive — in practice they shouldn't reach `ranked` at all, but cheap to guard)

**Step 7 — `strategy/sizing.py`** (§7)
- [x] `size_positions()`, iterative capping; `shares_from_allocation()`
- [x] Test: cap verified against total NAV, not sleeve size
- [x] Test: converges, sums to `sleeve_usd ± rounding`; regime_scalar scales sleeve proportionally

**Step 8 — `strategy/regime.py`** (§8)
- [x] `compute_regime()`, asymmetric hysteresis (immediate down / 2-month-confirmed up)
- [x] Redesigned hysteresis state to derive confirmation streak from `regime_history.json`'s own stored
      `bear`/`high_vol` fields (`recent_raw_states`) instead of a separately-tracked counter that could drift
      out of sync — pure function of input history, safer for the §14.9 B10 determinism requirement
- [x] Test: 20-day VIX smoothing survives a single-day spike; bear via 12m return OR 200dma

**Step 9 — `strategy/tracks.py`** (§9)
- [x] `Track` dataclass, `cost_model` flag (commissions_slippage vs ter); `apply_ter_drag()`, `build_nav_series()`
- [x] `compute_attribution()`: B−C / A−B / D−C / A−D decomposition, tested against hand-computed values

**Step 10 — Execution**
- [x] `execution/fills.py`: `simulate_fill()`, sells-before-buys, `WARN_INSUFFICIENT_CASH` proportional scale-down
- [x] `execution/ledger.py`: append-only `_state/ledger.jsonl`, `_state/current_positions.json` r/w (overwritten)
- [x] `results_dir`/`state_dir`/path overrides threaded through every job's `run()` for testability

**Step 11 — `quality/gates.py`** (§13)
- [x] `PipelineAbort`, `GateOutcome = PASS | WARN | ABORT | HOLD`, `run_gates()`
- [x] One `check_*()` function per §13 table row; one test per row, both pass and fail sides (21 tests)

**Step 12 — `results/run_io.py`** (§3)
- [x] Dated-folder read/write helpers, one per schema type; `append_regime_history()`, `append_nav_row()`
- [x] `already_processed()` idempotency check + `--force`/`force=True` bypass

**Step 13–15 — Job scripts**
- [x] `jobs/run_monthly_rebalance.py` — full fetch->validate->signal->filter->rank->select->size->regime->TARGET
- [x] `jobs/run_monthly_execute.py` — TARGET->fills->POSITIONS+LEDGER->REPORT (Track A only, see Known Issues)
- [x] `jobs/run_daily_mark.py` — NAV mark, catastrophic stop flag, anomaly flag
- [x] Each prints `__SCHEDULER_RESULT__:{...}` on success; each self-guards on its trigger day via `calendar.py`

**Step 16 — `reporting/monthly_report.py`** (§12)
- [x] §12.1–12.6 sections; disclaimer-text regression test (`STATISTICAL_POWER_DISCLAIMER_KEY_PHRASE`)
- [x] "Insufficient history" rendering for the decision panel before T+12

**Step 17 — Scheduling**
- [x] Resolved Open Decision #2: `job_schedules.cron` is plain croniter, no native month-boundary concept ->
      all three jobs run on a daily cron and self-guard internally (confirmed via `scheduler_service.py`)
- [x] `jobs/register_jobs.py`, idempotent upsert, 3 rows

**Step 18 — Tests**
- [x] One unit-test module per `strategy/`/`execution/`/`quality/`/`results/`/`reporting/`/`data/` file
- [x] `tests/test_integration_monthly_cycle.py` — full cycle against real file I/O in a tmp dir, including the
      §14.9 B10 determinism check (two `monthly_rebalance` runs -> byte-identical `targets.json`)
- [x] 154 tests total; 0 pyright errors; 0 mypy errors; 0 flake8 issues (E402 excepted, expected repo pattern)

## ✅ COMPLETED FEATURES (continued)

**Step 19 — Backtest harness** (§14, `backtest/p21_momentum/`) — separate build track, gates deployment
- [x] **Open Decision #1 resolved**: Option A (current S&P 500 applied backward, free, banner-flagged) —
      confirmed by the user over the ~$50-70/mo Option B point-in-time alternative (`runner.py` module
      docstring records this)
- [x] `runner.py` — single daily-pass engine over a frozen panel, shared with the live pipeline's own
      `strategy/`/`execution/` modules; documented simplifications (F4/F6 always-pass, same-day catastrophic
      stop, tracks C/D/E not itemized in the trade ledger) called out in the module docstring, not silent
- [x] `missing_data.py` — 3-day forward-fill cap, never back-fill, -30% delisting haircut (spec §14.4)
- [x] `fetch_frozen_panel.py` — one-time price + sector freeze to `data/prices.parquet` +
      `data/constituents.json` (sectors frozen alongside prices so a later Wikipedia edit can't silently
      change attribution mid-study); not yet run against real Yahoo/Wikipedia data (see Testing Requirements)
- [x] `metrics.py` — spec §14.8's mechanical/risk/return tiers (turnover, position count, sector
      concentration, filter attrition, regime histogram, holding period, trade size/commission, drawdown,
      tracking error, beta/corr, CAGR/Sharpe/Sortino/information ratio)
- [x] `stress_windows.py` — all 9 spec §14.6 windows, table-driven, A-B decisive comparison + regime scalar
      range per window, `stress_windows.md` renderer
- [x] `robustness.py` — 729-combination grid (§14.5 Rule 2), deflated-Sharpe band + top-quartile-separation
      decision rule (Rule 3), mechanically enforced single out-of-sample touch via `OutOfSampleReaccessError`
      + append-only `oos_access_log.md` (Rule 4)
- [x] `cost_sensitivity.py` — 4-level slippage sweep + `edge_survives_10bps()` (§14.9 B8), cost-neutral
      turnover/net-return curve across `hold_rank` (§14.7)
- [x] `phase0_report.py` — orchestrates base case -> stress windows -> cost sensitivity -> in-sample
      robustness -> §14.9 B1-B10 acceptance table -> `PHASE0_REPORT.md` (banner + table lead, per §14.10);
      `--verify-determinism` CLI flag re-runs the base case twice and diffs `nav_daily` byte-for-byte (B10)
- [x] 79 tests total (`backtest/p21_momentum/tests/`), synthetic fixtures shared via `tests/fixtures.py`, no
      network; 0 pyright errors, 0 mypy errors (`--ignore-missing-imports`; `backtest/` is outside both
      `pyrightconfig.json`'s `include` and `scripts/typecheck.py`'s `MYPY_TARGETS` — see Technical Debt), 0
      flake8 issues (E402/E203 excepted, same pattern already tolerated in `src/ml/pipeline/p21_momentum/`)
- [x] Bug found + fixed during Step 19: `BacktestResult.nav_daily` was keyed by `datetime.date`, not a real
      `pd.DatetimeIndex` — broke `.resample()`/`.rolling()` in `metrics.py` and date comparisons in
      `stress_windows.py` the moment they ran against a real `run_backtest()` result rather than a synthetic
      `pd.bdate_range` fixture. Fixed in `runner.py` by keying `nav_rows` off `ts` (the loop's `pd.Timestamp`)
      instead of `day` (`ts.date()`), plus an explicit `pd.DatetimeIndex(...)` cast on the empty-panel branch.

## 🔄 Not Yet Started

**Step 19 — Backtest harness, remaining work**
- [ ] Run `fetch_frozen_panel.py` for real (one-time, network) to produce `data/prices.parquet` +
      `data/constituents.json`, then run `phase0_report.py` end-to-end over 2005-2026 and inspect the
      resulting `PHASE0_REPORT.md` acceptance table — nothing in Step 19 has been exercised against real
      market data yet, only synthetic fixtures
- [ ] `robustness.py`'s out-of-sample evaluation (2017-01 -> 2026-06) is a deliberate, separate manual call
      (`run_grid(..., acknowledge_oos_reaccess=...)`) — `phase0_report.py` only ever touches the in-sample
      grid automatically, by design (spec §14.5 Rule 4); an operator still has to make that one OOS call
- [ ] `marginal_surfaces.png` (per-parameter plateau plots from the 729-grid, spec §14.10) is not yet
      generated — `grid_729.csv` has everything needed to plot it, but no renderer was built

**Step 20 — Parameter freezing** (§15, post-Phase-1 only — nothing to freeze against until Phase 1 runs)
- [ ] `scripts/freeze_params.py`, `verify_frozen_params()` wired into every job's `main()`

## Technical Debt
- [ ] `jobs/run_monthly_execute.py`'s report generation renders literal `0.0` placeholders for tracks B/C/D/E
      NAV, the §9 attribution decomposition, and the §12.5/§12.6 realized-metrics rollups (turnover, costs,
      decision criteria) — these need a full walk over `_state/nav_daily.csv` + `_state/ledger.jsonl` history.
      `strategy/tracks.py` has the building blocks (`apply_ter_drag`, `build_nav_series`, `compute_attribution`)
      but they are not yet wired into the job. Track A's own state is real, not a placeholder. Flagged in the
      module's own docstring — see `docs/Design.md`.
- [ ] `jobs/register_jobs.py`'s two ET-hour cron entries (`20:30 UTC` for the 16:30 ET jobs) do not adjust for
      DST — same known limitation as every other ET-scheduled job in this repo (P20 shares it); not solved here.
- [ ] `backtest/` is outside both type-check gates' scope: `pyrightconfig.json`'s `include` is `["src",
      "tests", "config"]` and `scripts/typecheck.py`'s `MYPY_TARGETS` is `["src"]`. Verified manually clean
      (0 pyright, 0 mypy `--ignore-missing-imports`) as of this session, but CI will not catch a future
      regression here unless one of those two configs is deliberately extended to include `backtest` —
      flagged, not fixed, since widening CI scope is a process decision, not a code change this task implied.
- [x] `src/ml/pipeline/p21_momentum/calendar.py`'s `import pandas_market_calendars` — **resolved**: `pyproject.toml`
      already carries a `[[tool.mypy.overrides]]` entry with `pandas_market_calendars.*` under
      `ignore_missing_imports = true` (2026-08-26 solution-architect review confirmed `python scripts/typecheck.py
      --mypy` passes clean, 1143 files, 0 errors). This item's original text was stale.

## Known Issues
- `jobs/run_monthly_execute.py` full multi-track NAV/attribution rollup — see Technical Debt above

## Testing Requirements
- [x] All of Step 18 above
- [x] Backtest harness's own tests (Step 19) — 79 tests, synthetic fixtures, no network; `--verify-determinism`
      is implemented and unit-tested against a deliberately-corrupted second run, but not yet run for real
      against the frozen 2005-2026 panel (that panel doesn't exist on disk yet — see Step 19 remaining work)
- [ ] One manual, non-CI smoke test of `data/earnings.py` against real yfinance (only `data/universe.py` has
      been smoke-tested against live Wikipedia so far; `data/prices.py` is exercised only via mocks)

## Documentation Updates
- [x] `README.md`, `docs/Requirements.md`, `docs/Design.md` (Step 0)
- [ ] Keep `pipeline-specification.md` and `implementation-plan.md` in sync — spec changes (especially to
      the provisional §10.1 decision) must be reflected in the plan before code changes, not worked around
      ad hoc
