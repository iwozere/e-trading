# Docs Cleanup & HLA Consolidation — Implementation Plan
**Date:** 2026-04-30

## Goal
Remove outdated and redundant files from `docs/`, merge surviving content into `docs/HLA/`, and leave `docs/` as a thin entry-point layer that points into HLA.

---

## Phase 1 — Hard Deletes (no content loss)

These files document completed work, one-off fixes, or planning artifacts. Their content either belongs in git history, the issue tracker, or is already superseded.

| File | Reason |
|------|--------|
| `docs/BACKTRADER_MIGRATION_SUMMARY.md` | Completed migration; historical artifact |
| `docs/FIXES_APPLIED.md` | Session-based maintenance log; belongs in CHANGELOG or git |
| `docs/MONITORING_ENDPOINT_DESIGN.md` | Design record superseded by `HLA/system-health-monitoring.md` |
| `docs/POSTGRESQL_MIGRATION.md` | Migration complete; any permanent notes belong in `HLA/database-architecture.md` |
| `docs/implementation_plan.md` | Task planning; belongs in issue tracker |
| `docs/notification_review.md` | Code-review artifact; open items belong in issue tracker |
| `docs/TIMESTAMP_INDEX_ISSUE.md` | Bug-fix record; context already in code + commit history |
| `docs/TIMEZONE_FIX_EXPLANATION.md` | Bug-fix record; same as above |

**Action:** `git rm` all 8 files.

---

## Phase 2 — Merge into HLA (content must be preserved)

Root docs that contain real, ongoing reference value not yet fully covered in HLA.

### 2a. Merge into `HLA/modules/trading-engine.md`
- `docs/ADVANCED_ATR_EXIT.md` — ATR exit strategy details
- `docs/LIVE_TRADING_BOT.md` — bot orchestration & lifecycle
- `docs/RISK_MANAGEMENT.md` — pre/real-time/post-trade controls
- `docs/BOLLINGER_BANDS_ATTRIBUTES.md` — indicator attribute reference (add as appendix)

### 2b. Merge into `HLA/modules/ml-analytics.md`
- `docs/ADVANCED_ANALYTICS.md` — Monte Carlo, metrics
- `docs/ADVANCED_ML_FEATURES.md` — MLflow, feature engineering
- `docs/HMM_LSTM_BACKTESTING_GUIDE.md` — neural-net backtesting walkthrough
- `docs/WALK_FORWARD_GUIDE.md` — out-of-sample testing (consolidate with `HLA/WALK_FORWARD_OPTIMIZATION.md`)
- `docs/KB.md` — trading-domain knowledge base (regimes, neural nets)

### 2c. Merge into `HLA/modules/data-management.md`
- `docs/DATA_DOWNLOADERS.md` — provider summary
- `docs/DATA_CACHE_REFRESH_GUIDE.md` — cache structure & refresh procedures
- `docs/QUICK_DATA_REFRESH.md` — quick-reference section at end
- `docs/YFINANCE_BATCH_DOWNLOAD.md` — batch download patterns
- `docs/LIVE_DATA_FEEDS.md` — Binance / Yahoo / IBKR live feeds

### 2d. Merge into `HLA/notification-services.md`
- `docs/NOTIFICATION_SYSTEM.md` — async delivery overview (563 lines; check for gaps vs 967-line HLA file)
- `docs/CHANNEL_OWNERSHIP.md` — service boundary definitions

### 2e. Merge into `HLA/logging-subsystem.md`
- `docs/CONTEXT_AWARE_LOGGING.md` — child-module logging inheritance
- `docs/MULTIPROCESSING_LOGGING.md` — queue-based logging

### 2f. Merge into `HLA/modules/infrastructure.md`
- `docs/ERROR_HANDLING.md` — resilience system (circuit breaker, retry)
- `docs/JOB_SCHEDULER.md` — APScheduler + Dramatiq
- `docs/METRICS.md` — KPI definitions
- `docs/WEBGUI.md` — Next.js web UI overview

### 2g. Merge into `HLA/system-health-monitoring.md`
- `docs/HEALTH_CHECK_CONSOLIDATION.md` — unified health system
- `docs/HEALTH_CHECK_REFACTORING.md` — DB-layer separation

### 2h. Merge into `HLA/modules/configuration.md`
- `docs/CONFIGURATION.md` — centralized config with Pydantic
- `docs/ENVIRONMENT_SETUP.md` — API keys & credentials setup

### 2i. Merge into `HLA/modules/communication.md`
- `docs/API.md` — REST API endpoints & JWT auth

### 2j. Add new HLA file: `HLA/deployment.md`
- `docs/RASPBERRY_PI_DEPLOYMENT.md`
- `docs/RASPBERRY_PI_QUICK_START.md`

### 2k. Add new HLA file: `HLA/strategy-framework.md`
- `docs/ADVANCED_STRATEGY_FRAMEWORK.md`
- `docs/UNIFIED_INDICATOR_SERVICE_ARCHITECTURE.md` (root version, 242 lines; consolidate with `HLA/UNIFIED_INDICATOR_ARCHITECTURE.md`)

### 2l. Merge into `HLA/modules/communication.md` (alerts section)
- `docs/ALERT_SYSTEM.md` — smart alert rules engine

---

## Phase 3 — Root-level files to keep as-is (thin entry points)

These stay in `docs/` because they serve a different audience or purpose than HLA deep-dives.

| File | Reason to keep |
|------|---------------|
| `docs/CHANGELOG.md` | Release history; not architecture |
| `docs/ROADMAP.md` | Forward-looking; distinct from HLA |
| `docs/USER_GUIDE.md` | End-user quickstart; links into HLA |
| `docs/DEVELOPER_GUIDE.md` | Onboarding guide; links into HLA |
| `docs/_TODO.md` | Active task list |
| `docs/issues/` | Issue tracking folder |

After Phase 2, `docs/` will contain ~6 files + `HLA/` + `issues/`. All reference material lives in HLA.

---

## Phase 4 — HLA index update

After merges, update `HLA/INDEX.md` and `HLA/README.md` to reflect:
- New files: `deployment.md`, `strategy-framework.md`
- Updated files: all modules that received merged content
- Remove stale pointers to deleted root files
- Update `HLA/VALIDATION_REPORT.md` date and scope

---

## Execution Order

```
Phase 1  →  git rm 8 files (safe, no content loss)
Phase 2a →  Update HLA/modules/trading-engine.md
Phase 2b →  Update HLA/modules/ml-analytics.md
Phase 2c →  Update HLA/modules/data-management.md
Phase 2d →  Update HLA/notification-services.md
Phase 2e →  Update HLA/logging-subsystem.md
Phase 2f →  Update HLA/modules/infrastructure.md
Phase 2g →  Update HLA/system-health-monitoring.md
Phase 2h →  Update HLA/modules/configuration.md
Phase 2i →  Update HLA/modules/communication.md
Phase 2j →  Create HLA/deployment.md
Phase 2k →  Create HLA/strategy-framework.md
Phase 2l →  Update HLA/modules/communication.md (alerts)
Phase 3  →  Verify 6 root files remain untouched
Phase 4  →  Update HLA/INDEX.md + README.md + VALIDATION_REPORT.md
```

---

## Result

| | Before | After |
|-|--------|-------|
| `docs/` root .md files | 47 | 6 |
| `docs/HLA/` files | 30 | 32 (+2 new) |
| Deleted | — | 8 |
| Merged away | — | 33 |
