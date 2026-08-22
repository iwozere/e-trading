# P21 Momentum — Implementation Plan v1

**Date:** 2026-08-22
**Based on:** `pipeline-specification.md` v1.2
**Status:** Not started — no code exists under `src/ml/pipeline/p21_momentum/` yet, only the spec.

---

## 1. Scope of This Plan

Two build tracks share the same core modules but are otherwise independent and run in this order:

1. **Backtest harness (§14)** — a mechanical validation harness over 2005–2026 history. Must pass its
   acceptance table (§14.9) *before* the live paper pipeline is ever scheduled (§15 Phase 0 gate).
2. **Live paper pipeline (§1, §11, §12)** — the three scheduled jobs (`monthly_rebalance`,
   `monthly_execute`, `daily_mark`) that run for real, forward-only, starting Phase 1.

The signal/filter/selection/sizing/regime/execution modules (§4–§10) are written **once** and imported by
both tracks — the backtest is not a separate reimplementation of the strategy logic. Only the outer loop
differs: the backtest iterates historical dates against a frozen Parquet panel; the live pipeline iterates
real time against `YahooDataDownloader`. Build order below reflects this: strategy modules first (§6.1),
backtest harness second (§8) because it gates deployment, live job wiring third (§6.3) because it depends on
Phase 0 having passed.

---

## 2. Reuse Map

Everything in this table is **read, not reimplemented**. Nothing in `p21_momentum/` should duplicate logic
that already exists in one of these.

| Spec need | Existing component | Notes |
|---|---|---|
| OHLCV batch (signal, MTUM, SPY, ^GSPC, ^VIX, tracks) | `DataDownloaderFactory.create_downloader("yahoo")` → `YahooDataDownloader.get_ohlcv_batch()` (`src/data/downloader/yahoo_data_downloader.py`) | `auto_adjust=True`; batches with automatic per-symbol fallback; see §10.1 of the spec for why this is the only price source |
| Fundamentals (F4 quality filter) | `YahooDataDownloader.get_fundamentals_batch()` → `Fundamentals.free_cash_flow`, `.net_income` | TTM as exposed by yfinance `.info`; already handles the batch/fallback split |
| S&P 500 tickers (existing, ticker-only) | `src.util.tickers_list.get_sp500_tickers_wikipedia()` | Reused as the base for the new sector-aware function (§4 below) — same page, same header workaround for Wikipedia's 403 |
| Logger | `src.notification.logger.setup_logger(__name__)` | Project-wide standard; every new module uses this, never `print()` |
| Notifications (ABORT alerts, §13) | `src.notification.service.client.NotificationServiceClient` | Same client P20 uses; `send_to_admins()` for ABORT-level gate failures |
| Trading calendar | **New dependency: `pandas_market_calendars`** | Not currently in `requirements.txt` anywhere in this repo — confirmed absent. Add `pandas_market_calendars` (pin latest stable) to `requirements.txt`. No existing in-repo NYSE-calendar utility was found to reuse instead |
| Scheduler registration | `job_schedules` table, `job_type='data_processing'`, `task_params.script_path` — same mechanism P14/P17/P18/P20 all use (confirmed via P18's `docs/plan.md` SQL and P20's `jobs/register_jobs.py`) | Each job script needs `main()` + `if __name__ == "__main__":` + a final `print(f"__SCHEDULER_RESULT__: {json.dumps(...)}")`, matching `sp500_stock_screener.py`'s pattern |

**Not reused, and why (both already flagged in the spec):**

| Gap | Why not reused | Plan |
|---|---|---|
| Earnings calendar (F6) | No downloader class exposes `get_earnings_dates()` | New, small, isolated helper — `data/earnings.py` — calling `yf.Ticker(symbol).get_earnings_dates()` directly. This is the *one* place P21 talks to yfinance outside `YahooDataDownloader` (spec §2) |
| GICS sector per constituent | `get_sp500_tickers_wikipedia()` only returns `Symbol`, drops the `GICS Sector` column from the same table | New function `get_sp500_constituents_with_sector()` added to the *same* `src/util/tickers_list.py` file, next to the existing one — not a p21-local copy, so any other pipeline that later needs sectors can reuse it too |
| `src.trading.broker.paper_trading_mixin.PaperTradingMixin` | Considered and rejected: it's an async, live-market, non-deterministic (`random.random()` reject/partial-fill probabilities) engine for intraday bots, with a flat `commission_rate` model. §14.9 B10 requires **bit-identical reruns**, and §10 needs IBKR's tiered per-share commission with a per-order minimum — neither fits without fighting the mixin's design. Spec's own `simulate_fill()` (§10) stays a small deterministic function in `execution.py` | No action — documented here so it isn't "discovered" again mid-build |
| IBKR bid/ask reconciliation (§10.2) | `IBKRDownloader` only wraps `reqHistoricalData`; live bid/ask is new integration work against `ibkr_broker.py` | Deferred to Phase 3+ per spec — not in this plan's scope at all |

---

## 3. New Module Structure

```
src/ml/pipeline/p21_momentum/
├── __init__.py
├── config.py                      # §16 parameters as module constants; RESULTS_DIR, STATE_DIR
├── schemas.py                     # dataclasses for every §3 JSON shape (SignalRow, TargetPosition,
│                                   #   Position, LedgerEntry, RegimeState, DailyMarkSnapshot, ...)
├── calendar.py                    # pandas_market_calendars XNYS wrapper: is_trading_day(),
│                                   #   last_trading_day_of_month(), first_trading_day_of_month()
├── data/
│   ├── __init__.py
│   ├── universe.py                # §2: wraps get_sp500_constituents_with_sector(), ticker normalization
│   ├── prices.py                  # §2: thin wrapper over YahooDataDownloader.get_ohlcv_batch()
│   │                               #   + non-emptiness checks (§13) + fundamentals_cache/sectors_cache I/O
│   ├── earnings.py                # §2: the one direct-yfinance call, isolated and documented as such
│   └── exclusions.py              # §5 F5: reads config/pipeline/p21_exclusions.json, filters expired
├── strategy/
│   ├── __init__.py
│   ├── signal.py                  # §4: compute_signal()
│   ├── filters.py                 # §5: F1–F6, each a pure function over the survivor list
│   ├── selection.py               # §6: rank → hysteresis retain → sector cap → fill to TARGET_COUNT
│   ├── sizing.py                  # §7: size_positions(), inverse-vol with iterative capping
│   ├── regime.py                  # §8: bear/high_vol/scalar + upgrade-confirmation hysteresis
│   └── tracks.py                  # §9: A/B/C/D/E attribution bookkeeping
├── execution/
│   ├── __init__.py
│   ├── fills.py                   # §10.1: simulate_fill(), chatter threshold, sells-before-buys
│   └── ledger.py                  # §3: append-only writer/reader for _state/ledger.jsonl,
│                                   #   _state/current_positions.json, _state/nav_daily.csv
├── quality/
│   ├── __init__.py
│   └── gates.py                   # §13: all threshold checks; raises PipelineAbort on ABORT-level
├── results/
│   ├── __init__.py
│   └── run_io.py                  # §3: dated-folder read/write, idempotency check (§3 "Idempotency"),
│                                   #   _state/ cache TTL logic
├── reporting/
│   ├── __init__.py
│   └── monthly_report.py          # §12: assembles report.md from the run's artifacts + ledger slice
├── jobs/
│   ├── __init__.py
│   ├── run_monthly_rebalance.py   # §1 pipeline: fetch→validate→signal→filter→rank→select→size→regime→TARGET
│   ├── run_monthly_execute.py     # §1 pipeline: TARGET→fills→POSITIONS+LEDGER→tracks→REPORT
│   ├── run_daily_mark.py          # §11
│   └── register_jobs.py           # one-time, idempotent: INSERT job_schedules rows (§9 of this plan)
├── tests/
│   ├── __init__.py
│   ├── test_calendar.py
│   ├── test_universe.py
│   ├── test_signal.py
│   ├── test_filters.py
│   ├── test_selection.py
│   ├── test_sizing.py
│   ├── test_regime.py
│   ├── test_tracks.py
│   ├── test_fills.py
│   ├── test_ledger.py
│   ├── test_gates.py
│   ├── test_run_io.py
│   ├── test_monthly_report.py
│   └── test_integration_monthly_cycle.py   # rebalance → execute → daily_mark on synthetic fixed data
├── README.md                      # written at Step 0 alongside config.py — see §6.1
└── docs/
    ├── pipeline-specification.md  # already exists
    ├── implementation-plan.md     # this file
    ├── Requirements.md            # written at Step 0 — see §6.1
    ├── Design.md                  # written at Step 0 — see §6.1
    └── Tasks.md                   # written alongside this plan

# Backtest harness — separate top-level tree per spec §14.10, NOT under results/p21_momentum/ (§3 note)
backtest/p21_momentum/
├── data/
│   ├── prices.parquet             # frozen panel, fetch timestamp in filename or sidecar JSON
│   └── constituents.parquet       # only if Option B (point-in-time) is later funded
├── runner.py                      # §14: iterates historical months, calls the same strategy/ modules
├── robustness.py                  # §14.5: grid runner + marginal-surface plots
├── cost_sensitivity.py            # §14.7: slippage grid + turnover/net-return curve
├── stress_windows.py              # §14.6: per-window reports
├── oos_access_log.md              # §14.5 Rule 4: append-only, timestamped
└── results/
    ├── base_case/
    ├── robustness/
    └── cost_sensitivity/
```

`config/pipeline/p21_exclusions.json` and `config/frozen_params.json` (project-root `config/`, not under
`src/ml/pipeline/p21_momentum/`) — same convention as P20's `config/pipeline/activists.json`.

---

## 4. Data Contracts (`schemas.py`)

One dataclass per §3 JSON shape, `slots=True`, with `to_dict()`/`from_dict()` for JSON round-tripping (this
repo's pattern elsewhere is plain dataclasses + `dataclasses.asdict()`, not pydantic, for file-I/O-only
models — pydantic is reserved for schemas crossing an API boundary, per the mypy-clean-up era convention).

| Dataclass | Backs | Key fields |
|---|---|---|
| `SignalRow` | `signals.json` | ticker, raw_return, vol, signal, rank, sector, filters_passed (dict of F1–F6 bool/flag) |
| `TargetPosition` | `targets.json` | ticker, target_weight_pct, target_usd, rank, sector |
| `Position` | `_state/current_positions.json`, `positions.json` | ticker, shares, avg_cost, entry_date, entry_rank, current_rank, sector, target_weight_pct, high_water_price |
| `LedgerEntry` | `_state/ledger.jsonl` rows | ts, track, ticker, side, shares, ref_open, fill_price, slippage_bps, commission_usd, gross_usd, net_usd, reason (`Literal` of the §3 permitted-values list) |
| `RegimeState` | `_state/regime_history.json` entries | date, spx_12m_return, spx_vs_200dma, vix_20d_avg, bear, high_vol, scalar_raw, scalar_applied, months_at_state |
| `DailyMarkSnapshot` | `daily_mark.json` | as_of, nav per track (A–E), flagged anomalies, catastrophic-stop triggers |
| `NavRow` | `_state/nav_daily.csv` rows | date, nav_a, nav_b, nav_c, nav_d, nav_e |

`LedgerEntry.reason` is a `Literal[...]` of the nine permitted values in spec §3 (note:
`CORPORATE_ACTION_SPLIT` is **not** in the list — see spec §10.1) so a bad reason string is a type error, not
a silent data-quality issue discovered three months later in the monthly report.

---

## 5. Config & Constants (`config.py`)

Mirrors spec §16 verbatim as module-level constants (no YAML parsing — this repo's convention for pipeline
constants is Python constants, e.g. P20's `config.py`, not a parsed config file, except for the two
operator-facing JSON files below):

```python
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "p21_momentum"
STATE_DIR = RESULTS_DIR / "_state"
EXCLUSIONS_PATH = PROJECT_ROOT / "config" / "pipeline" / "p21_exclusions.json"
FROZEN_PARAMS_PATH = PROJECT_ROOT / "config" / "frozen_params.json"   # written at end of Phase 1, §15

# Signal (§4)
LOOKBACK_START = 252
SKIP_RECENT = 21
MIN_HISTORY = 260

# Filters (§5 / §16)
MIN_ADV_USD = 50_000_000
ADV_WINDOW_DAYS = 60
GAP_FILTER_TOP3_SHARE = 0.40
EARNINGS_BLACKOUT_DAYS = 5

# Selection (§6 / §16)
ENTRY_RANK = 20
HOLD_RANK = 60
MAX_PER_SECTOR = 4
TARGET_COUNT = 20
FALLBACK_POOL_RANK = 40

# Sizing (§7 / §16)
NAV_TOTAL_USD = 250_000
SLEEVE_TARGET_PCT = 0.20
MAX_POSITION_PCT = 0.01
MIN_TRADE_USD = 150

# Regime (§8 / §16)
BEAR_LOOKBACK_DAYS = 252
MA_DAYS = 200
VIX_SMOOTHING_DAYS = 20
VIX_THRESHOLD = 28
SCALAR_NORMAL, SCALAR_BEAR_LOWVOL, SCALAR_BEAR_HIGHVOL = 1.00, 0.60, 0.25
UPGRADE_CONFIRMATION_MONTHS = 2

# Execution (§10 / §16)
SLIPPAGE_BPS = 3.0
COMMISSION_MIN_USD = 0.35
COMMISSION_PER_SHARE = 0.0035
COMMISSION_MAX_PCT = 0.01

# Risk (§16)
CATASTROPHIC_STOP_PCT = -0.35

# Benchmark (§16)
PROXY_TICKER = "MTUM"
TER_ADJUSTMENT_ANNUAL = 0.0005
MARKET_REFERENCE = "SPY"
```

**Frozen-params hash check** (§15): once Phase 1 ends, `run_monthly_rebalance.py`'s first step becomes
"import `config`, hash the module's constant block, compare to the hash stored in `frozen_params.json`,
abort loudly on mismatch." Implemented as a small `verify_frozen_params()` call at the top of every job's
`main()`, added in Step 20 below — not before Phase 1 exists to freeze against.

---

## 6. Build Steps

### 6.1 Step 0 — Scaffolding

- `__init__.py` files empty, per CLAUDE.md §2.2.
- `config.py` as above.
- `README.md`, `docs/Requirements.md`, `docs/Design.md` written *now*, not deferred — CLAUDE.md §12.4 requires
  them for any new `src/` module and there is enough in the spec + this plan to write them accurately:
  - `Requirements.md`: `pandas_market_calendars` (new), `yfinance` (already a project dependency, used only
    in `data/earnings.py`), everything else via `src.data.downloader` / `src.notification` — no new API keys.
  - `Design.md`: condense spec §1 (architecture), §3 (results layout), §9 (four-track attribution), and the
    §10.1 adjusted-close-only decision (flagged provisional, same wording as the spec) — this doc should
    point *into* `pipeline-specification.md` for anything more detailed than a paragraph, not duplicate it.
  - `README.md`: quick-start showing how to run `run_monthly_rebalance.py --force` locally against a small
    universe for smoke-testing, and where to look in `results/p21_momentum/` afterward.
- `results/p21_momentum/_state/cache/` created on first run by `run_io.py` (`mkdir(parents=True,
  exist_ok=True)`), not committed to git — add `results/p21_momentum/` to the existing bare `data/`-style
  `.gitignore` pattern used elsewhere (careful: the repo's typecheck-CI gotcha memo warns a bare `data/`
  ignore entry previously swallowed real source packages — use an explicit `results/` path, not a bare
  pattern, and verify with `git check-ignore` before committing).
- Add `pandas_market_calendars` to `requirements.txt`.

### 6.2 Step 1 — `calendar.py`

```python
import pandas_market_calendars as mcal

_XNYS = mcal.get_calendar("XNYS")

def is_trading_day(d: date) -> bool: ...
def last_trading_day_of_month(year: int, month: int) -> date: ...
def first_trading_day_of_month(year: int, month: int) -> date: ...
```
Unit tests pin known holidays (e.g. 2026 Thanksgiving, Good Friday) to catch a bad calendar version pin.

### 6.3 Step 2 — `data/universe.py` + `tickers_list.get_sp500_constituents_with_sector()`

Add the new function to `src/util/tickers_list.py` (not `p21_momentum/`, per §2 reuse-map rationale):

```python
def get_sp500_constituents_with_sector() -> pd.DataFrame:
    """Return Symbol + GICS Sector for current S&P 500 constituents.

    Same Wikipedia table and header workaround as get_sp500_tickers_wikipedia();
    additionally keeps the 'GICS Sector' column that function drops.
    """
```
`data/universe.py` calls this, applies `.replace('.', '-')` (already done inside the new function, matching
existing behavior), and writes `universe.json` for the run (§3).

**Known gap to flag in the module docstring, not silently fixed:** this is Option A from spec §14.3 —
*current* constituents applied to every date, including historical backtest dates. That is fine for the live
pipeline (constituents are always "current" there by construction) but is the exact survivorship bias §14.2
requires the backtest to carry a banner about. `data/universe.py` itself has no time dimension — the banner
logic lives in the backtest harness (§8 below), not here.

### 6.4 Step 3 — `data/prices.py`, `data/earnings.py`, `data/exclusions.py`

- `prices.py`: `fetch_price_panel(tickers, start, end) -> dict[str, pd.DataFrame]` wrapping
  `YahooDataDownloader.get_ohlcv_batch()`; raises `PipelineAbort` (quality gate §13, "Tickers with complete
  price data ≥ 95%") if too many symbols come back empty. Also owns `fetch_fundamentals_cached()` (90-day
  TTL against `_state/cache/fundamentals.json`) and `fetch_sectors_cached()` (30-day TTL).
- `earnings.py`: single function `next_earnings_date(ticker) -> date | None`, the one direct-yfinance call
  in the whole pipeline, documented as such at the top of the file per spec §2.
- `exclusions.py`: reads `config/pipeline/p21_exclusions.json`, drops expired entries by `expires` date,
  returns a `set[str]` of excluded tickers. Read-only — the pipeline never writes to this file (spec §5).

### 6.5 Step 4 — `strategy/signal.py` (§4)

`compute_signal(adj_close: pd.Series) -> SignalResult | None` — near-verbatim port of the spec's pseudocode.
Unit tests: exact boundary at `MIN_HISTORY=260`, the `vol < 0.05` guard, and — most important — a
regression test asserting the function ranks by `signal` (`raw_return / vol`) and **not** `raw_return`,
since the spec calls this out as the single most consequential implementation error.

### 6.6 Step 5 — `strategy/filters.py` (§5)

Six pure functions, `F1_history`, `F2_liquidity`, `F3_gap`, `F4_quality`, `F5_exclusions`, `F6_earnings`,
each `(candidate, context) -> FilterResult(passed: bool, flag: str | None)`, run in the cheap-first order
the spec mandates (§5 intro). `filters.run_all(candidates)` composes them and tallies `f4_data_missing` for
the §12.6 D5 decision criterion. F3's `total <= 0` short-circuit and F4's "pass rather than exclude on
missing data" are both spec-mandated non-obvious branches — each gets its own test.

### 6.7 Step 6 — `strategy/selection.py` (§6)

Implements the 7-step operation order **exactly in the order given** (spec is explicit that order matters):
rank → retain (≤HOLD_RANK) → forced exits → sector cap on retained → fill from top-20 → widen to top-40 on
underfill → accept-smaller-and-`WARN_UNDERFILLED` as last resort. Deterministic tie-break
(`signal_desc, ticker_asc`, per §16) applied at the rank step, not left to Python's sort stability alone —
explicit `key=lambda x: (-x['signal'], x['ticker'])`. This determinism requirement (§14.9 B10) is the reason
`enforce_sector_cap`'s "drop the worst-ranked" tie-break must also be `ticker_asc` on ties, not dict-order.

### 6.8 Step 7 — `strategy/sizing.py` (§7)

`size_positions()` ported near-verbatim; the 10-iteration capping loop is already deterministic (no
randomness) so no extra test burden beyond checking it converges and sums to `sleeve_usd ± rounding`, and
that the cap is verified against **total NAV**, not sleeve size (spec calls this out as a deliberate,
non-obvious choice — dedicated test).

### 6.9 Step 8 — `strategy/regime.py` (§8)

`compute_regime(spx, vix, prior_state) -> RegimeState`. The asymmetric hysteresis (downward = immediate,
upward = 2 consecutive months) is the one piece of state this function needs from outside itself
(`prior_state.months_at_state`) — passed in explicitly rather than read from disk inside the function, so
the function stays pure and testable. The caller (`run_monthly_rebalance.py`) is responsible for reading
`_state/regime_history.json`'s last entry and passing it in, then appending the result.

### 6.10 Step 9 — `strategy/tracks.py` (§9)

Maintains A/B/C/D/E NAV series from a shared set of daily closes. B and C/D need their own "always 100%
invested" / "TER 0.20%/252 daily" bookkeeping distinct from A's cost-and-overlay-bearing path — implemented
as one `Track` dataclass with a `cost_model: Literal["commissions_slippage", "ter"]` flag rather than five
near-duplicate functions, so the A−D / B−C / D−C / A−B decomposition (§9) is computed from one consistent
NAV table, not from independently-drifting per-track calculations.

### 6.11 Step 10 — `execution/fills.py` + `execution/ledger.py` (§10, §3)

- `fills.py`: `simulate_fill()` ported verbatim (§10.1 code block); sells-before-buys ordering and the
  `WARN_INSUFFICIENT_CASH` proportional-scale-down live here too, since they're part of the same "how do we
  actually turn a TARGET into fills" concern.
- `ledger.py`: append-only writer for `_state/ledger.jsonl` (never truncates, never rewrites a line — only
  appends), full read for the monthly report's "this month's trades" slice (filter by `ts` date range, per
  spec §3's explicit "don't duplicate, filter" instruction), and read/write for
  `_state/current_positions.json` (this one *does* get overwritten each run, unlike the ledger).

### 6.12 Step 11 — `quality/gates.py` (§13)

One `PipelineAbort(Exception)` and one `run_gates(context) -> list[GateResult]` that checks every row of the
§13 table in order, raising `PipelineAbort` (which every job's `main()` catches, alerts via
`NotificationServiceClient.send_to_admins()`, and exits non-zero without touching `results/` for that run)
on the first ABORT-level failure, but collecting WARN-level results to include in the report instead of
stopping. `^GSPC`/`^VIX` unavailable is the one HOLD-not-ABORT case (retain prior regime scalar) — modeled
as a third `GateOutcome` value (`ABORT | WARN | HOLD`), not a boolean, so this doesn't need a special case
bolted on later.

### 6.13 Step 12 — `results/run_io.py` (§3)

- `run_dir_for(date) -> Path`, creating `results/p21_momentum/YYYY-MM-DD/` on first write.
- `already_processed(run_date, primary_output_filename) -> bool` — the idempotency check every job runs
  first (spec §3 "Idempotency"); `--force` CLI flag bypasses it.
- Typed read/write helpers per schema (`write_signals(run_date, rows: list[SignalRow])`, etc.) so a job
  script never touches `json.dump`/`Path` directly — one place to get the dated-folder-vs-`_state/` split
  right (spec §3).

### 6.14 Step 13–15 — Job scripts

`run_monthly_rebalance.py`, `run_monthly_execute.py`, `run_daily_mark.py` are thin orchestrators: import the
`strategy/`, `execution/`, `quality/`, `results/` modules built above and wire them in the exact order the
spec's §1 pipeline diagram and §11 daily-job list specify. Each ends with
`print(f"__SCHEDULER_RESULT__: {json.dumps({...})}")` (scheduler convention, §2 reuse map) summarizing
counts (`positions_count`, `trades_count`, `warn_underfilled`, etc.) for the scheduler's log.

`run_monthly_execute.py` additionally calls `reporting/monthly_report.py` at the end (§12) — report
generation is not a separate scheduled job, it's the last step of `monthly_execute`, matching spec §1's
pipeline diagram (`... → generate REPORT`).

### 6.15 Step 16 — `reporting/monthly_report.py` (§12)

Renders `report.md` from: this run's `targets.json`/`positions.json`, the `_state/ledger.jsonl` slice for
the month, `_state/nav_daily.csv`, and `_state/regime_history.json`'s latest entry. §12.2's statistical-power
disclaimer paragraph (with the computed t-statistic) is a module-level string template — verified by a test
that greps the rendered output for the required phrase, so a future edit can't silently drop it. §12.6's
decision-criteria panel only evaluates meaningfully at T+12 — before that, render the table with an
"insufficient history" note per criterion rather than a misleading `PASS`/`FAIL` on a 2-month sample.

### 6.16 Step 17 — `jobs/register_jobs.py`

Idempotent `INSERT INTO job_schedules ... ON CONFLICT (user_id, name) DO NOTHING`, same pattern as P20's
`register_jobs.py` and P18's documented SQL. Three rows:

| job name | cron | script_path |
|---|---|---|
| `p21_monthly_rebalance` | last NYSE trading day of month, 16:30 ET | `src/ml/pipeline/p21_momentum/jobs/run_monthly_rebalance.py` |
| `p21_monthly_execute` | first NYSE trading day of month, 09:45 ET | `src/ml/pipeline/p21_momentum/jobs/run_monthly_execute.py` |
| `p21_daily_mark` | every NYSE trading day, 16:30 ET | `src/ml/pipeline/p21_momentum/jobs/run_daily_mark.py` |

Cron expressions for "last/first trading day of month" can't be expressed as plain cron (holidays shift
them) — either (a) schedule daily and let the job itself no-op via `calendar.py` on non-trading-days /
non-boundary-days, or (b) if the scheduler supports a pre-execution guard callback, use that. **Needs a
decision against the actual `scheduler_service.py` capabilities before Step 17** — flagged as an open
question in §10 below, not resolved by this plan.

### 6.17 Step 18 — Tests

One test module per `strategy/`/`execution/`/`quality/`/`results/` file (unit, synthetic fixtures, no
network) plus `test_integration_monthly_cycle.py`: a full `monthly_rebalance → monthly_execute → daily_mark`
run against a small fixed synthetic universe (10–15 tickers, hand-constructed price series covering at least
one sector-cap-triggering month and one underfill month), asserting the final `_state/` files match expected
byte-for-byte on a second run (§14.9 B10's determinism requirement applies to the live code paths too, not
just the backtest).

### 6.18 Step 19 — Backtest harness

See §8 of this plan (separate, larger track).

### 6.19 Step 20 — Parameter freezing (§15)

After Phase 1 completes: `scripts/freeze_params.py` (one-off, not scheduled) hashes `config.py`'s constant
block, writes `config/frozen_params.json`. `verify_frozen_params()` (§5 above) is added to every job's
`main()` at this point — not before, since there's nothing to verify against yet during Phase 0/1.

---

## 7. Testing Plan Summary

| Layer | Approach |
|---|---|
| `strategy/*`, `execution/fills.py` | Pure-function unit tests, synthetic `pd.Series`/`dict` fixtures, no network, no filesystem |
| `execution/ledger.py`, `results/run_io.py` | `tmp_path`-based tests (pytest fixture), asserting append-only behavior and idempotency skip logic |
| `quality/gates.py` | Table-driven: one test per §13 row, both pass and fail sides |
| `reporting/monthly_report.py` | Snapshot-style: render against a fixed fixture, assert required substrings (disclaimer, decision table) present |
| Integration | `test_integration_monthly_cycle.py` — full cycle, deterministic rerun check |
| Backtest harness | Separate — see §8.6 below (this is where the real bias/robustness/stress-window testing lives, not in `p21_momentum/tests/`) |

No test hits the network. `data/prices.py`, `data/earnings.py`, `data/universe.py` are exercised only
through mocked/injected downloader instances in unit tests — actual Yahoo/Wikipedia access is verified once
manually during Step 0 smoke-testing, not in CI.

---

## 8. Backtest Harness (§14) — Separate Build Track

This is materially more work than the live pipeline's job wiring and is scoped as its own track because it
gates deployment (§15 Phase 0) but shares no job-scheduling concerns with it.

### 8.1 Universe construction decision

Spec §14.3 offers three options. **Recommendation: start with Option A (current S&P 500 applied backward)**
— it's free, it's what `data/universe.py` (§6.3 above) already produces with no extra work, and §14.3 is
explicit that mechanical metrics (the actual Phase 0 acceptance criteria, §14.9) are largely unaffected by
survivorship bias. Option B (point-in-time via Norgate/Sharadar, ~$50–70/mo) only matters if return figures
from the backtest are meant to inform a real decision — and §14.1/§14.11 are explicit that they should not
be. **This needs your confirmation before Step 19 starts**, since it's a recurring-cost decision, not a
code decision — flagged in §10 below.

If Option A is confirmed, `PHASE0_REPORT.md`'s banner (spec §14.3 exact wording) is a hard-coded template
string in `runner.py`, rendered unconditionally — not optional, not something a flag can suppress.

### 8.2 `runner.py`

Iterates `pandas_market_calendars` XNYS trading days from `2005-01-01` to `2026-06-30`, calling the same
`strategy/`+`execution/` modules as the live pipeline against `backtest/p21_momentum/data/prices.parquet`
instead of live `YahooDataDownloader` calls. The frozen-panel fetch (`YahooDataDownloader.get_ohlcv_batch()`
once, §14.2's "price adjustment drift" mitigation) is a one-time script, not part of `runner.py`'s per-run
path — re-running `runner.py` must never re-download.

Missing-data handling (§14.4: 3-day forward-fill max, never back-fill, −30% delisting haircut) lives here,
not in the live-pipeline `data/prices.py`, since live data doesn't have this class of gap.

### 8.3 `robustness.py` (§14.5)

Runs the 3^6 = 729-combination grid (§14.5 Rule 2/3), computes the deflated-Sharpe band, and — critically —
implements **Rule 4's out-of-sample discipline mechanically**, not as a documentation-only convention:
`runner.py` refuses to evaluate the 2017-01→2026-06 window more than once per grid-search session unless
`--acknowledge-oos-reaccess` is passed, and every evaluation (in-sample or out) appends a timestamped line to
`oos_access_log.md` regardless. This is the one place in the whole plan where a process rule from the spec
is turned into an enforced code constraint rather than an operator instruction, because §14.5 is explicit
that this is "where backtests are most often ruined."

### 8.4 `cost_sensitivity.py` (§14.7)

Four slippage levels × the `hold_rank` turnover-curve sweep. Reuses `runner.py`'s single-config path in a
loop — no separate simulation logic.

### 8.5 `stress_windows.py` (§14.6)

Slices the base-case run's output by the nine named windows, reports A/B/C/D/E + realized `regime_scalar`
path per window. Table-driven from the spec's §14.6 window list, so adding a tenth window later is a
one-line change.

### 8.6 Acceptance gate (§14.9)

`PHASE0_REPORT.md` generator evaluates B1–B10 against the run's own output and prints a pass/fail table as
the document's lead section (before any performance number, per spec). **B10 (bit-identical reruns) is
enforced by CI-style self-check**: `runner.py --verify-determinism` runs the base case twice and diffs the
two `nav_daily.csv` outputs byte-for-byte, failing loudly on any difference — this is the single highest-
value test in the whole backtest track, since non-determinism here would otherwise surface as an
unreproducible bug months into live paper trading.

---

## 9. Deployment Checklist (ties spec §15 to this plan's artifacts)

1. `requirements.txt` updated with `pandas_market_calendars`; `pip install -r requirements.txt` in whatever
   env runs the scheduler.
2. `config/pipeline/p21_exclusions.json` seeded (empty `{"exclusions": []}` is a valid starting state).
3. Backtest harness (§8) run to completion; `PHASE0_REPORT.md`'s §14.9 table all-pass (or documented,
   accepted failures with fixes applied per the table's "Response on failure" column).
4. Step 17's cron-vs-scheduler-capability question (§6.16) resolved and `register_jobs.py` run once.
5. Phase 1 dry run (spec §15): one month, `--dry-run`-equivalent (jobs run, `LEDGER`/`POSITIONS` written,
   but flagged so `monthly_execute` doesn't feed forward into next month's `current_positions.json` as real
   capital) — **this needs a `--paper-dry-run` flag threaded through `execution/ledger.py`**, not currently
   in the module list above; add during Step 10 once Step 17's scheduling mechanics are settled, since the
   exact flag semantics depend on how the scheduler distinguishes a dry-run invocation from a real one.
6. `config/frozen_params.json` written (§6.19) at the end of Phase 1.
7. Phase 2: 12 months, no parameter changes, `config/pipeline/p21_changelog.md` created (empty, ready for
   any exceptional entries) at the start of Phase 2, not before.

---

## 10. Open Decisions Not Resolved by This Plan

These need a human call before or during the corresponding step — flagging now rather than guessing:

| # | Decision | Affects | Default if not answered |
|---|---|---|---|
| 1 | Backtest universe: Option A (free, biased, banner-flagged) vs Option B (~$50–70/mo, point-in-time) | §8.1, Step 19 | Proceed with Option A — spec itself recommends it unless "returns matter to you at all," and §0/§14.11 argue they structurally can't yet |
| 2 | Can `scheduler_service.py` express "last/first trading day of month" natively, or does the job need a self-guarding daily cron? | §6.16, Step 17 | Assume self-guarding daily cron (safer, no scheduler-internals dependency) unless investigation during Step 17 finds native support |
| 3 | Exact semantics of Phase 1's "dry run" — does `monthly_execute` write to the *same* `_state/current_positions.json` the real Phase 2 run will later read, or a Phase-1-only sandboxed copy? | Deployment checklist item 5 | Sandboxed copy (`_state/current_positions.dryrun.json`), promoted to the real path only when Phase 2 begins — avoids Phase 1 test data silently becoming Phase 2's opening position |

Everything else in the spec (§4–§13, §16) is unambiguous enough to build directly from the pseudocode given.

---

*This plan implements `pipeline-specification.md` v1.2. Any spec change (especially to §10.1's provisional
adjusted-close-only decision) should be reflected here before continuing implementation, not worked around
ad hoc in code.*
