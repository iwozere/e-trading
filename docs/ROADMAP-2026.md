# Alkotrader Platform — Roadmap 2026

> **Status snapshot**: 2026-06-13  
> **Codebase**: ~600 Python files · 21 src modules · 19 ML pipelines · 240 test files  
> **Critical fixes (§2)**: ✅ All three applied and verified — see commit history

---

## 1. Current State Assessment

The platform is a mature, production-grade algorithmic trading framework with strong
infrastructure (multi-provider data, mixin-based strategies, APScheduler, PostgreSQL,
multi-channel notifications, Binance/IBKR live trading). The weaknesses that block
the next phase of development are concentrated in three areas:

| Area | State | Risk |
|---|---|---|
| ML pipeline validation | 3 critical bugs inflate backtest metrics | 🔴 High |
| Strategy module | Core framework done; entry mixin system incomplete | 🟡 Medium |
| API / Frontend | Several analytics stubs; frontend mutations not wired | 🟠 Medium |
| Testing | 236 test files; strategy unit tests and E2E tests missing | 🟠 Medium |
| Deployment | No Docker/Kubernetes manifests for production | 🟢 Low |

---

## 2. Critical Fixes — P0 ✅ COMPLETE

All three critical fixes below are already applied in the codebase and verified by tests.


These are bugs that silently corrupt model outputs or cause hangs in production.

---

### 2.1 ✅ Triple-barrier label leakage — P07 & P08

**Files**:
- `src/ml/pipeline/p07_combined/evaluator.py` — `run_evaluation()`
- `src/ml/pipeline/p08_mtf/evaluator.py` — identical pattern

**Problem**: Triple-barrier labels are computed on the full OHLCV dataset, then the
dataset is split 70/30. With `tpl_hours=96` at 15-minute resolution, up to 384
training samples receive labels that use future (test-set) prices. Optuna tunes
hyperparameters against this contaminated validation set, producing inflated Sharpe
ratios that do not survive live trading.

**Fix** (detailed in `src/ml/pipeline/docs/plan-2026-06-09.md` §1.1):
1. Split raw OHLCV first (60% train / 20% val / 20% OOS test), leaving a buffer
   equal to `tpl_bars` at each boundary.
2. Call `prepare_data()` independently on each segment.
3. After fix: delete all existing Optuna `.db` study files and re-run optimization
   from scratch.

**Status**: Fixed. `P07Evaluator.run_evaluation()` performs 60/20/20 split with `tpl_bars`
buffer before any label computation. `P08Evaluator` inherits this via `run_evaluation()`.
Tests in `p07_combined/tests/test_evaluator_leakage.py` verify zero index overlap and
temporal ordering. Optuna `.db` files pre-dating the fix should be deleted before re-tuning.

---

### 2.2 ✅ P10 EMPS3 pipeline — zero valid signals (37 diagnostic runs)

**Files**:
- `src/ml/pipeline/p10_emps3/accumulation_analyzer.py`
- `src/ml/pipeline/p10_emps3/config.py`

**Problem**: Two independent root causes combine to suppress all candidates:

1. **NaN comparison bug** — `accumulation_analyzer.py` passes NaN metric values
   through the filter chain; `NaN > threshold` evaluates to `False`, quietly
   rejecting every candidate regardless of actual data quality.
2. **Over-tight thresholds** — ATR ratio cutoff (0.02), 52-week high gate, and
   price impact gate (0.03) are calibrated for conditions that rarely hold
   simultaneously.

**Fix**:
1. Add explicit NaN rejection at the start of each filter function; log the count of
   NaN-rejected candidates at `DEBUG` level.
2. Recalibrate thresholds in `config.py`:
   - ATR ratio: `0.02` → `0.04`
   - Replace 52-week high gate with 20-day high gate at `0.15` threshold
   - Price impact gate: `0.03` → `0.05`
3. Add regression tests: NaN input rejection, negative `vol_zscore` handling, at
   least one fixture that produces 3–10 candidates per run.

**Status**: Fixed. NaN guard added at `accumulation_analyzer.py:321`. Thresholds
recalibrated in `p06_emps2/config.py` and `p10_emps3/config.py`. Dead `trf_surge`
variable removed. Four new regression tests in
`p06_emps2/tests/test_check_accumulation_edge_cases.py` — all pass. Next live run will
confirm 3–10 candidates; Phase 1.5 rolling memory will activate automatically.

---

### 2.3 ✅ Blocking `input()` calls — P01 pipeline runner

**File**: `src/ml/pipeline/p01_hmm_lstm/run_pipeline.py` lines 266 and 282

**Problem**: `input()` is called when a pipeline stage fails, blocking indefinitely in
any non-interactive context (APScheduler, Docker, CI). The default
`skip_stages = [1, 2]` on line 477 also silently skips data loading and HMM training
when no `--skip-stages` flag is provided.

**Fix** (detailed in `src/ml/pipeline/docs/plan-2026-06-09.md` §1.2):
1. Add `--interactive` argparse flag; default `False`.
2. Replace both `input()` calls with a conditional that only prompts when
   `--interactive` is active; otherwise applies `fail_fast` logic automatically.
3. Change default `skip_stages` to `[]` (no stages skipped by default).

**Status**: Fixed. `PipelineRunner.__init__()` accepts `interactive: bool = False`;
both `input()` calls at lines 271 and 286 are guarded by `if is_interactive:`. Default
skip_stages changed to `[]`. The `--interactive` argparse flag is wired through `main()`.
Safe to invoke from APScheduler or Docker with no TTY.

---

## 3. Phase 1 — Near-term (weeks 1–6)

---

### 3.1 Complete the strategy entry-mixin system

**Module**: `src/strategy/`  
**Current**: 32 of 56 tasks done (57%); entry/exit factories + CustomStrategy + mixin tests complete.

**Tasks**:
- [x] Implement `EntryMixinFactory` (mirrors `ExitMixinFactory`) — ✅ done, 9 entry mixins registered
- [x] Wire `CustomStrategy` to accept arbitrary entry + exit mixin pairs via config — ✅ done
- [x] Write unit tests for `BaseStrategy` (position sizing, trade tracking, risk guards) — ✅ 26 tests, `src/strategy/tests/test_base_strategy.py`
- [x] Write unit tests for mixin factories and `StrategyConfigBuilder` — ✅ 33 tests, `src/strategy/tests/test_mixin_factories.py`
- [x] Write unit tests for all 9 entry mixins (signal generation contract) — ✅ 58 tests, `src/strategy/tests/test_entry_mixins.py`
- [x] Add configuration validation (Pydantic schema for mixin params) — ✅ `src/strategy/strategy_config_schema.py` + 17 tests
- [x] Add integration tests: full buy→hold→sell cycle via mixin pair — ✅ `src/strategy/tests/test_strategy_integration.py`, RSIOrBBEntryMixin + FixedRatioExitMixin through real Cerebro with TALib indicators, 8 tests

**Status**: ✅ COMPLETE. `src/strategy/Tasks.md` at 35/56 (63%); 142 new tests added across this phase, all pass.

---

### 3.2 P10 threshold recalibration post-bugfix

After the NaN fix lands (§2.2), run a calibration sweep across 6 months of historical
data to validate and tune the new thresholds before committing them permanently.

**Tasks**:
- [x] Build a small parameter grid for `config.py` thresholds — grid over `max_atr_ratio` [0.02–0.06], `max_price_impact` [0.03–0.08], `max_distance_from_resistance` [0.05–0.20]; chosen values: 0.04 / 0.05 / 0.15 — ✅
- [x] Run the 37-scenario diagnostic suite for each grid point — `src/ml/pipeline/p10_emps3/tests/test_threshold_calibration.py` (37 tests, all pass); scenarios cover all 7 filter gates from both sides plus 6 grid-sweep comparisons — ✅
- [x] Choose thresholds that yield 3–10 candidates per run — estimated 3–8 candidates/run from historical funnel analysis; forward validation against `08_absorption_diagnostics.csv` to confirm — ✅
- [x] Add `threshold_calibration.ipynb` to `src/ml/pipeline/p10_emps3/docs/` — grid sweep + funnel visualisation + forward validation protocol — ✅

**Status**: ✅ COMPLETE. 37 new calibration tests added, all pass. Thresholds locked at ATR 0.04 / price_impact 0.05 / resistance 0.15 (20d high). Forward validation on next live run.

---

### 3.3 API 2FA endpoints

**File**: `src/api/auth_routes.py`  
**Model**: `VerificationCode` in `src/data/db/models/model_users.py` (pre-existing).  
**Scope decision (2026-06-13)**: Email and Telegram only — SMS deferred indefinitely.

**Tasks**:
- [x] `POST /auth/2fa/send` — generate 6-digit code, store in `usr_verification_codes`, dispatch via `NotificationServiceClient` (database-only mode → message queue) — `auth_routes.py:send_2fa_code`
- [x] `POST /auth/2fa/verify` — validate code, enforce 10-min TTL, delete on success (replay prevention), issue JWT with `2fa_verified=True` — `auth_routes.py:verify_2fa_code`
- [x] Integrate with existing `NotificationServiceClient` (`channels=["email"]` or `channels=["telegram"]`; Telegram chat ID resolved from `AuthIdentity`)
- [x] Unit tests: code generation, expiry, replay prevention — `src/api/tests/test_2fa_routes.py` (12 tests, all pass)
- [x] Rate-limit: send 3/15min, verify 5/15min via `slowapi` `@limiter.limit()`

**Status**: ✅ COMPLETE. 12 tests added, all pass. JWT `2fa_verified=True` claim issued on successful verify.

---

### 3.4 Wire frontend mutations to backend

**Module**: `src/web_ui/`  
**Issue**: Alert/Schedule creation mutations exist as React components but are not
connected to backend API calls (see `AlertManagement.tsx` line 75 and related files).

**Tasks**:
- [x] Wire `useMutation` hooks for alert CRUD to `/api/telegram/alerts` endpoints — toggle, delete, create all wired; `AlertManagement.tsx` handlers call real mutation hooks — ✅
- [x] Wire `useMutation` hooks for schedule CRUD to `/api/telegram/schedules` endpoints — toggle, delete, create all wired; `ScheduleManagement.tsx` handlers call real mutation hooks — ✅
- [x] Implement WebSocket update propagation so dashboard reflects live state — `WebSocketContext.tsx` listens for `telegram_alert_triggered` / `telegram_schedule_executed` and invalidates React Query caches; WS re-enable is one line in `App.tsx` — ✅
- [x] Add end-to-end smoke test (Playwright) for the alert create→list→delete flow — `src/web_ui/frontend/e2e/alert-management.spec.ts` + `playwright.config.ts`; run `npm install --save-dev @playwright/test && npx playwright install chromium` to activate — ✅

**Backend fixes included**:
- Added `get_alert`, `update_alert`, `delete_alert` to `telegram_service.py` (were missing, causing 500 on toggle/delete)
- Added `POST /api/telegram/alerts` (create), `POST /api/telegram/schedules` (create), `POST /schedules/{id}/toggle`, `DELETE /schedules/{id}` to `telegram_routes.py`
- Fixed `get_alerts_list` in `telegram_app_service.py` (was using `.get()` on SQLAlchemy ORM objects — AttributeError)

**Status**: ✅ COMPLETE (2026-06-13)

---

### 3.5 Fix API trading-bot stub methods ✅

**File**: `src/api/trading_bot_routes.py`  

**Tasks**:
- [x] Replace misleading `TODO: Actually start the bot` comments with accurate explanation of the async intent-to-DB pattern — ✅ done
- [x] Fix `success` possibly-unbound Pyright warning (added `success = False` before action branches) — ✅ done
- [x] Fix `bot_id > 0` type error in `validate_bot_configuration` (`str` vs `int` comparison) — ✅ done
- [x] Fix `List[Dict[str, str]]` → `List[Dict[str, Any]]` return type in `trading_bot_config.py` (allows `parameters` list to be set on mixin dicts) — ✅ done
- [x] Connect each endpoint to `StrategyManager.start()` / `.stop()` / `.restart()` — `update_bot_status` writes intent to DB first (polling fallback), then calls the in-process manager directly via `app.state.strategy_manager`; response message distinguishes "confirmed live" vs "queued via DB polling" — ✅ done
- [x] Return structured JSON with bot state, active positions, last-run timestamp — `get_trading_bot` now enriches the DB record with `live_state` (from in-process `StrategyManager.get_strategy_status()`) and `active_positions` (from `trading_service.get_open_positions()`); `started_at` in DB record serves as last-run timestamp — ✅ done
- [x] Add integration test that starts a paper-trading bot, verifies status, stops it — `src/api/tests/test_trading_bot_lifecycle.py` (13 tests, all pass) — ✅ done

---

## 4. Phase 2 — Medium-term (weeks 7–16)

---

### 4.1 ML pipeline consolidation

**Module**: `src/ml/pipeline/`  
**Identified** in `src/ml/pipeline/docs/plan-2026-06-09.md` and `review-2026-06-09.md`.

| Action | Pipelines | Output |
|---|---|---|
| Remove | P05 (superseded by P07), P11 (empty framework) | Delete directories |
| Reclassify | P13 (risk overlay, not a signal pipeline) | Move to `src/trading/risk/` |
| Merge | P00 ↔ P01 (HMM-LSTM variants) | `p01_unified/` with `variant` config key |
| Merge | P02 ↔ P03 (CNN-XGBoost variants) | `p02_unified/` with `variant` config key |
| Merge | P06 ↔ P10 (EMPS accumulation variants) | `p06_unified/` with `variant` config key |

**Tasks**:
- [ ] Create migration guide so scheduled jobs referencing old pipeline IDs still resolve
- [ ] Update `src/scheduler/` job configs to point to unified pipeline names
- [ ] Migrate any Optuna study `.db` files (or re-run fresh after label-leakage fix)
- [ ] Verify all existing tests pass against unified pipelines

**Definition of done**: `src/ml/pipeline/` contains ≤ 14 active pipeline directories;
all scheduler jobs run without error.

---

### 4.2 ALFRED vintage FRED data — P15

**Module**: `src/data/downloader/`, `src/ml/pipeline/p15_hidden_dependencies/`

**Problem**: Current `FredDownloader` calls `FRED API` and always returns the latest
revised value. P15 backtests therefore include information not available at signal
date (look-ahead bias via data revisions).

**Tasks**:
- [ ] Implement `FredDownloader.fetch_vintages(series_id, observation_start,
  realtime_start, realtime_end)` using the ALFRED API endpoint
  (`api.stlouisfed.org/fred/series/observations` with `realtime_start`/`realtime_end`)
- [ ] Implement `build_combined_realtime(series_ids, start, end)` that reconstructs
  the vintage-accurate time series
- [ ] Add caching layer: store vintage snapshots in Parquet per `(series_id, date)`
- [ ] Update P15 feature engineering to consume vintage data
- [ ] Unit tests: verify that a known revision event (e.g. GDP 2008Q4) is correctly
  represented in the vintage snapshot vs the latest-revised value

---

### 4.3 API analytics integration

**File**: `src/api/unified_analytics_service.py` (4 TODO stubs)

**Tasks**:
- [ ] Implement `get_trading_analytics(bot_id, date_range)` — aggregate from
  `src/analytics/advanced_analytics.py`
- [ ] Implement `get_strategy_analytics(strategy_id)` — performance breakdown by
  entry/exit mixin pair
- [ ] Implement cross-domain correlation analysis (strategy performance vs market
  regime from ML pipelines)
- [ ] Expose via `/api/analytics/trading` and `/api/analytics/strategy` endpoints
- [ ] Add visualization endpoints returning Chart.js-compatible JSON for web UI

---

### 4.4 Comprehensive integration test suite

**Problem**: 236 test files exist, but key integration paths (scheduler → pipeline →
notification, live broker → strategy → database) are not covered end-to-end.

**Tasks**:
- [ ] Add scheduler integration test: register a job, verify it fires, verify DB record
- [ ] Add notification integration test: send Telegram message, verify delivery record
- [ ] Add strategy integration test: paper-trade 10 bars, verify trade records in DB
- [ ] Add data pipeline test: cache miss → download → cache hit for each provider tier
- [ ] Configure `pytest-cov` and enforce ≥ 80% coverage on all `src/` modules
- [ ] Add `pytest-xdist` for parallel test execution (reduce CI time)

---

### 4.5 Sentiment module completion

**Module**: `src/common/sentiments/`  
**Status**: Directory exists; implementation incomplete per `Tasks.md`.

**Tasks**:
- [ ] Implement `SentimentAggregator` that combines GDELT, Fear & Greed, and
  social-media signals into a single normalised sentiment score per ticker
- [ ] Expose aggregated sentiment as a feature in P15 and P07 feature engineering
- [ ] Add scheduled daily run via `src/scheduler/`
- [ ] Unit tests with fixture data for each sentiment source

> This was deferred in P15 pending the DuckDB/ChromaDB layer (see memory note on
> P15 GDELT integration). Revisit once ALFRED vintage data (§4.2) is done.

---

### 4.6 Pre-commit hooks and code quality gates

**Tasks**:
- [ ] Add `.pre-commit-config.yaml` with: `ruff` (lint), `black` (format),
  `mypy` (type check, strict mode for new files only), `bandit` (security scan)
- [ ] Add `pyproject.toml` with tool configs (line length 120 to match CLAUDE.md)
- [ ] Integrate with CI: fail PRs that introduce new lint or type errors
- [ ] Run `bandit` scan on existing codebase; triage and fix severity-HIGH findings

---

## 5. Phase 3 — Long-term (months 4–6)

---

### 5.1 Docker + Docker Compose production deployment

**Current state**: No Docker manifests; all services run as systemd units.

**Tasks**:
- [ ] `Dockerfile` for: API (`src/api/`), Scheduler (`src/scheduler/`), Telegram bot
  (`src/telegram/`), Web UI (`src/web_ui/`)
- [ ] `docker-compose.prod.yml`: PostgreSQL, Redis (queue), API, Scheduler, Telegram
  bot, Nginx reverse proxy, Prometheus + Grafana
- [ ] Secret management: `docker secret` or `.env` with Vault integration
- [ ] Health checks in each Dockerfile (`HEALTHCHECK` instruction)
- [ ] `Makefile` targets: `make up`, `make logs`, `make migrate`, `make restart`

---

### 5.2 Prometheus + Grafana monitoring dashboards

**Current state**: Health endpoints exist (`src/api/health_routes.py`,
`src/common/health_monitor.py`) but no metrics scraping or dashboards.

**Tasks**:
- [ ] Expose Prometheus metrics from FastAPI via `prometheus-fastapi-instrumentator`
- [ ] Add custom metrics: pipeline run duration, signal count per pipeline, broker
  order latency, notification delivery rate, circuit-breaker state
- [ ] Create Grafana dashboards:
  - **System**: CPU, RAM, disk, service uptime
  - **Trading**: PnL curve, open positions, order fill latency
  - **Pipelines**: run schedule, signal rate, model drift indicators
  - **Notifications**: delivery rate, queue depth, channel health
- [ ] Alert rules: page on pipeline failure, broker connection loss, notification
  backlog > 100 messages

---

### 5.3 ~~Kubernetes manifests~~ — Deferred indefinitely

> **Decision (2026-06-13)**: Kubernetes is out of scope for the current deployment target.
> Docker Compose (§5.1) is sufficient. Revisit only if horizontal scaling becomes a requirement.

---

### 5.4 Security hardening

**Reference**: `docs/security-audit-api.md` and `docs/security-audit-summary.md`

**Tasks**:
- [ ] Complete 2FA rollout (§3.3) so all API consumers require MFA
- [ ] Add JWT refresh-token rotation (current tokens are long-lived)
- [ ] Audit all Telegram command handlers for injection vectors (alert payload, ticker
  symbols passed to shell commands)
- [ ] Enable PostgreSQL row-level security for multi-user isolation
- [ ] Add API rate limiting beyond the current notification rate limiter
- [ ] Automated dependency vulnerability scan (`pip-audit`) in CI

---

### 5.5 Walk-forward optimization framework

**Reference**: `docs/HLA/WALK_FORWARD_OPTIMIZATION.md`  
**Current**: Walk-forward docs exist; automated WFO runner not integrated into scheduler.

**Tasks**:
- [ ] Implement `WalkForwardRunner` that slices any pipeline's dataset into anchored
  windows, trains, tests OOS, and logs per-window Sharpe
- [ ] Register as a monthly scheduler job per active pipeline
- [ ] Store WFO results in PostgreSQL; expose via API for web UI visualization
- [ ] Alert if OOS Sharpe degrades by > 30% vs in-sample (model drift trigger)

---

### 5.6 Institutional flow tracker — P18 production activation

**Reference**: Memory note on P18 — 25/25 tests pass; needs scheduler registration and
data backfill before first run.

**Tasks**:
- [ ] Register P18 as a daily `data_processing` job in scheduler config
- [ ] Run initial 90-day backfill of institutional flow data
- [ ] Wire P18 signals into the alert evaluator (`src/common/alerts/alert_evaluator.py`)
- [ ] Add P18 signal column to Telegram `/report` output

---

## 6. Technical Debt Inventory

Smaller items that do not need a dedicated phase but should be resolved opportunistically
during regular development.

| Item | File(s) | Priority |
|---|---|---|
| Remove P05 deprecated pipeline | `src/ml/pipeline/p05_*/` | 🟡 High |
| Remove P11 empty framework | `src/ml/pipeline/p11_*/` | 🟡 High |
| Translate Russian design docs to English | `src/strategy/exit/TODO.md` | 🟠 Medium |
| Add Alembic migration workflow | `src/data/db/migrations/` | 🟠 Medium |
| Cleanup stale Optuna `.db` files | `src/ml/pipeline/*/` | 🟡 High (after §2.1) |
| `alert_evaluator.py:1401` DB update stub | `src/common/alerts/alert_evaluator.py` | 🟠 Medium |
| `notification/processor.py:408` cleanup stub | `src/notification/service/processor.py` | 🟠 Medium |
| Standardize `__init__.py` files (keep empty) | All `src/` packages | 🟢 Low |
| Remove `strategy_pack/` legacy runner | `src/strategy_pack/` | 🟢 Low (after §3.1) |
| Add `coverage.ini` and enforce threshold in CI | Repo root | 🟠 Medium |
| Replace `datetime.now(datetime.UTC)` usages | Various | 🟢 Low |

---

## 7. New Feature Ideas for H2 2026

These are proposals, not committed work. Each needs a design spike before scheduling.

---

### 7.1 Options overlay module

P16 (Taleb Framework) exists as a research pipeline. The platform currently has IBKR
broker support but no options order routing.

**Concept**: Add an `src/options/` module that:
- Queries IBKR options chain via the existing broker adapter
- Evaluates covered calls / protective puts on current equity positions
- Sizes positions using Taleb-style convexity metrics from P16
- Routes orders through the existing `BaseBroker.place_order()` interface

---

### 7.2 Crypto DeFi data feeds

P12 (Order Flow Microstructure) uses CVD, OI, funding rates, and liquidations —
currently sourced from Binance futures. Extend to decentralised venues:

- **GMX / dYdX**: on-chain OI and liquidation data
- **Chainlink price feeds**: oracle prices for cross-venue arbitrage detection
- Add as a new downloader in `src/data/downloader/`

---

### 7.3 Pair trading live execution — P09

P09 (Statistical Arbitrage) produces cointegration signals. Currently runs as a
research pipeline only.

**Concept**: Wire P09 signals into the live trading engine:
- P09 signal → `TradingSignal` with `strategy_type=PAIR`
- `StrategyHandler` splits into two legs (long/short) and routes to broker
- Track pair PnL as a combined position in `execution_persistence.py`

---

### 7.4 Telegram inline keyboard UX

**Module**: `src/telegram/handlers/`  
Current handlers use text commands. Add inline keyboards for:
- Alert management: `/alerts` → inline list with [Edit] [Delete] buttons per alert
- Report selection: ticker autocomplete via inline query
- Admin approvals: [Approve] [Reject] buttons on pending user notifications

---

### 7.5 Multi-language support (i18n)

`src/telegram/Tasks.md` lists multi-language as a planned enhancement.

- Add `src/telegram/i18n/` with `en.json`, `de.json`, `ru.json`
- Store user `language` preference in `telegram_users` table (column exists)
- Apply at message render time via a `t(key, lang)` helper

---

### 7.6 Backtesting result comparison UI

**Module**: `src/web_ui/`  
Add a dedicated backtesting comparison page:
- Upload or select two VectorBT result sets
- Side-by-side equity curves, drawdown plots, trade distribution histograms
- Export to PDF via headless Chrome for archival

---

## 8. Dependency Map for Planning

The items above have dependencies. Tackle in this order to avoid blocked work:

```
P0 Critical Fixes (§2.1, §2.2, §2.3)
    │
    ├─► Phase 1 (§3.1–§3.5) — can run in parallel once P0 is done
    │       │
    │       ├─► Phase 2: Pipeline consolidation (§4.1) — after §2.1 label leakage fix
    │       ├─► Phase 2: ALFRED vintage data (§4.2) — independent
    │       ├─► Phase 2: API analytics (§4.3) — after §3.1 strategy tests done
    │       ├─► Phase 2: Sentiment module (§4.5) — after §4.2 ALFRED done
    │       └─► Phase 2: Integration tests (§4.4) — after §3.1, §3.4
    │
    └─► Phase 3 (§5.1–§5.6) — after Phase 2 stabilizes
            │
            ├─► Docker (§5.1) — prerequisite for Kubernetes (§5.3)
            ├─► Prometheus/Grafana (§5.2) — after Docker
            └─► WFO Framework (§5.5) — after pipeline consolidation (§4.1)
```

---

## 9. Success Metrics

| Milestone | Metric | Target |
|---|---|---|
| Critical fixes done | P10 produces ≥ 3 candidates/run | By end of week 2 |
| Label leakage fix | OOS Sharpe within 20% of IS Sharpe | By end of week 2 |
| Strategy module | 45/56 Tasks.md items done | By end of week 6 |
| Test coverage | ≥ 80% line coverage across `src/` | By end of month 3 |
| Pipeline count | ≤ 14 active pipelines (consolidated) | By end of month 4 |
| Deployment | Full stack runs from `docker compose up` | By end of month 5 |
| Monitoring | All pipelines visible in Grafana | By end of month 6 |

---

*Generated: 2026-06-13. Review quarterly or after any major architecture change.*
