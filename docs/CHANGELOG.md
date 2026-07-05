# Changelog

All notable changes to the Alkotrader platform are recorded here.
Format: most recent first, grouped by theme.

---

## [2026-06] — Monitoring Dashboard & Bug Fixes

### Web UI — Monitoring Page
- Added **Services tab**: live systemd service status cards with `has_errors` badge
  (reads last 10 min of journald for Traceback / CRITICAL / OOM patterns).
- Added **Pipelines tab**: table of all scheduler pipelines with last-run status,
  duration, 10-run success rate, next scheduled run, and expandable run-history strip.
- Added **System tab**: CPU %, RAM %, uptime, version, and active-strategy count.
- All timestamps displayed in European format `dd.mm.yyyy hh:mm:ss`.

### Monitoring — False-Positive Fixes
- Removed `trading-api.service` duplicate unit (was a copy of `trading-webui.service` on port 5003).
- Loop devices (`/dev/loop*`) excluded from disk-usage threshold checks — they are
  always 100 % full by design (snap/AppImage mounts) and were generating false CRITICAL alerts.
- `check_service_logs` pattern `CRITICAL` tightened to `- CRITICAL -` so it matches
  only the Python log-level token, not the word inside a message body
  (e.g. `WARNING - System alert generated: CRITICAL - Disk usage…`).

### Bug Fix — `Message.updated_at` AttributeError
- `get_queue_health_for_channels()` in `database_optimization.py` referenced
  `Message.updated_at` and `Message.delivered_at`, which do not exist on the ORM model.
  Fixed: `updated_at` → `processed_at` (failed / stuck queries), `delivered_at` → `processed_at`.

### Portfolio — IBKR Symbol Mapping
- IBKR Flex Query exports non-US ETFs without exchange suffix (e.g. `VUSD` instead of `VUSD.L`).
  Added `_IBKR_SYMBOL_MAP` dict in `ibkr_xml_loader.py`; map applied at parse time so
  fresh exports are automatically translated without manual XML edits.

---

## [2026-05 / 2026-06] — Code Review & Quality Pass

- Full architecture review with Opus 4.8: P1 (critical) → P4 (low) issues triaged across
  all modules (data/db, downloader, notification, screeners, scheduler, telegram, trading, common).
- All P1/P2 issues resolved: circuit-breaker state transitions, log masking for sensitive data,
  dependency injection correctness, walk-forward pickle docs, screener consolidation.
- Removed legacy screener scripts from `src/util/` (UTIL-1 cleanup).
- Stock screeners wired into scheduler as `data_processing` jobs.

---

## [2026-05] — Portfolio PnL Alert Pipeline

- New module `src/portfolio/pnl_alert/` — daily end-of-day PnL alert for IBKR positions.
- **IBKR Flex Query loader** (`ibkr_xml_loader.py`): parses Open Positions XML export,
  merges multi-account positions via weighted-average cost basis.
- **Watchlist loader** (`watchlist_loader.py`): YAML-based supplementary watchlist
  (non-IBKR positions / overrides).
- **Price fetcher** (`price_fetcher.py`): thin wrapper around `DataManager.get_ohlcv`
  with per-symbol error isolation.
- **PnL evaluator** (`pnl_evaluator.py`): computes unrealised PnL % vs average cost.
- **Notifier** (`notifier.py`): formats and dispatches Telegram + email alert.
- **Runner** (`runner.py`): async orchestration pipeline; scheduled via `scheduler.service`
  at 21:30 UTC Mon–Fri.
- Non-US symbols require Yahoo/Tiingo exchange suffix (e.g. `VUSD.L`); documented in README.

---

## [2026-04 / 2026-05] — Web UI (React + FastAPI)

- Replaced Flask admin panel with a full **React 18 + Vite + MUI** frontend served by FastAPI.
- Frontend dev server on port **5002**; FastAPI backend on port **5003**
  (Vite proxies `/api`, `/auth`, `/ws` to backend in dev mode).
- Pages implemented: Dashboard, Strategies, Analytics, Telegram management suite
  (User Management, Audit Logs, Broadcast Center, Alert Management, Schedule Management,
  Config Builder), Administration, **Monitoring**.
- JWT-based session auth with HttpOnly cookie, rate limiting, and bcrypt password hashing.
- WebSocket endpoint (`/ws`) for real-time updates.
- `trading-webui.service` systemd unit replaces old `trading-bot.service` / `trading-admin.service`.

---

## [2026-04] — API Security Audit

- Removed `passlib`; enforced `bcrypt` directly for password hashing.
- All sensitive values (API keys, tokens) masked in logs via `SensitiveDataFilter`.
- Rate limiter added to FastAPI app.
- Internal log-alert endpoint (`/internal/log-alert`) restricted to localhost only.
- JWT secret and admin credentials moved to `config/donotshare/.env`.

---

## [2026-03 / 2026-04] — Vector Log Monitoring

- Deployed **Vector** (Rust binary) on Raspberry Pi to collect journald + Docker logs.
- Pipeline: journald → normalize → error-filter → fingerprint → 30-min throttle → HTTP POST
  to `/internal/log-alert` on port 5003.
- Alerts delivered via existing notification pipeline (PostgreSQL → `notification-bot.service` → Telegram).
- Excluded noisy patterns: apport, systemd condition-skip, IB Gateway Xlib errors.

---

## [2026-02 / 2026-03] — ML Pipelines (P15–P18)

### P15 — GDELT + FINRA Short Interest
- GDELT news-sentiment downloader: `.zip` files now stored as `.gz`.
- FINRA daily short-interest downloader bug-fixed (all tickers were downloaded daily instead of incrementally).
- Intraday downloader improvements; pipeline refactored.

### P16
- New ML pipeline added.

### P17
- New ML pipeline added.

### P18 — Institutional Flow Tracker
- Full pipeline implemented; 25/25 unit tests passing.
- Pending: scheduler registration + initial backfill before first production run.

---

## [2026-01 / 2026-02] — Scheduler Improvements

- `next_run_at` bug fixed (was not updated after successful run in some edge cases).
- Pipeline timeouts tuned for pipelines 06, 10, and 15.
- Scheduler pipeline table refactored; `ScheduleRun` records now store `job_snapshot`
  with `schedule_id` for reliable pipeline ↔ run matching on the Monitoring page.
- `notification-bot.service` and `scheduler.service` split into separate systemd units
  for independent restart and log isolation.

---

## [2025 and earlier] — Foundation

- Core database schema (PostgreSQL): users, strategies, schedules, schedule runs,
  notification messages (`msg_messages`, `msg_delivery_status`), Telegram user management.
- `DataManager` with multi-provider fallback (Tiingo → FMP → Yahoo Finance → cache).
- IBKR broker integration (`IBKRBroker`) via ib_insync.
- Backtesting framework (Backtrader integration).
- HMM + LSTM ML pipeline (P00/P01).
- CNN + XGBoost pipeline (P03).
- Short-squeeze scanner (P04).
- Notification system: async DB-backed queue, Telegram + email channels.
- Telegram bot: screener, alert monitoring, schedule management.
- Error handling: circuit breaker, retry manager, custom exception hierarchy.
- Multiprocessing-safe logging with `QueueHandler` / `QueueListener`.
