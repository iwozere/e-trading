# Submodule Reference

Alkotrader is organised as a monorepo under `src/`. Each top-level folder is an
independent submodule with its own tests and (where applicable) `docs/` folder.
This document gives a concise description of every submodule, its responsibilities,
key files, and integration points.

---

## Table of Contents

1. [API](#1-api)
2. [Web UI](#2-web-ui)
3. [Notification](#3-notification)
4. [Scheduler](#4-scheduler)
5. [Data](#5-data)
6. [Trading](#6-trading)
7. [Telegram](#7-telegram)
8. [Portfolio](#8-portfolio)
9. [ML Pipelines](#9-ml-pipelines)
10. [Strategy](#10-strategy)
11. [Backtester](#11-backtester)
12. [Indicators](#12-indicators)
13. [Common](#13-common)
14. [Config](#14-config)
15. [Error Handling](#15-error-handling)
16. [Monitoring (Infrastructure)](#16-monitoring-infrastructure)

---

## 1. API

**Path:** `src/api/`  
**Production entry point:** `src/api/main.py`  
**Port:** 5003  
**Systemd unit:** `trading-webui.service`

### Purpose
FastAPI application that serves as the single backend for both the Web UI and
internal system-to-system communication. Handles authentication, REST endpoints,
WebSocket updates, and health checks.

### Key Files
| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app, router registration, lifespan hooks |
| `auth.py` / `auth_routes.py` | JWT session auth, login/logout, bcrypt password hashing |
| `monitoring_routes.py` | `/api/monitoring/services` and `/api/monitoring/pipelines` |
| `jobs_routes.py` | Pipeline schedule CRUD |
| `notification_routes.py` | Notification message endpoints |
| `telegram_routes.py` | Telegram user and bot management |
| `trading_bot_routes.py` | Strategy / bot management |
| `health_routes.py` | `/api/health` and `/api/health/full` |
| `internal_routes.py` | `/internal/log-alert` (localhost-only, used by Vector) |
| `websocket_manager.py` | WebSocket broadcast to connected clients |
| `rate_limiter.py` | Per-IP rate limiting middleware |
| `services/monitoring_service.py` | CPU/RAM/disk metrics via psutil |
| `services/telegram_health_service.py` | Telegram bot health probe |
| `services/notification_health_service.py` | Notification queue health probe |

### Auth Flow
1. `POST /auth/login` validates credentials, sets HttpOnly JWT cookie.
2. All protected routes depend on `get_current_user` (reads cookie).
3. Passwords stored with bcrypt; no passlib dependency.

### Integration Points
- Reads/writes PostgreSQL via `src/data/db/services/`.
- Serves static files built from `src/web_ui/frontend/dist/` in production.
- Receives log alerts from Vector via `/internal/log-alert`.
- Sends WebSocket events to the frontend via `websocket_manager`.

---

## 2. Web UI

**Path:** `src/web_ui/`  
**Frontend:** `src/web_ui/frontend/` (React 18 + Vite + MUI)  
**Dev port:** 5002 (Vite); production static files served by FastAPI on 5003  
**Entry point:** `src/web_ui/run_web_ui.py`

### Purpose
Modern single-page application for system administration, monitoring, and
Telegram bot management. In dev mode, Vite proxies `/api`, `/auth`, and `/ws`
to `localhost:5003`.

### Pages
| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | Overview cards, strategy status |
| Strategies | `/strategies` | CRUD for trading strategies |
| Analytics | `/analytics` | Performance charts |
| Monitoring | `/monitoring` | System / Services / Pipelines tabs |
| Telegram — User Management | `/telegram/users` | Approve/block/manage users |
| Telegram — Audit Logs | `/telegram/audit` | Command audit trail |
| Telegram — Broadcast | `/telegram/broadcast` | Mass message to users |
| Telegram — Alerts | `/telegram/alerts` | Price/indicator alert rules |
| Telegram — Schedules | `/telegram/schedules` | Pipeline schedule management |
| Telegram — Config Builder | `/telegram/config` | Bot config editor |
| Administration | `/admin` | System administration |
| Login | `/login` | JWT auth form |

### Monitoring Page (added 2026-06)
- **System tab**: CPU %, RAM %, uptime, version.
- **Services tab**: live card per `SERVICES_TO_MONITOR` (systemd status + error badge).
- **Pipelines tab**: all schedules with last status, duration, success rate, next run,
  collapsible run history (coloured dots).

### Key Hooks
`src/web_ui/frontend/src/hooks/system/useSystemHealth.ts` — React Query hooks for
`/api/monitoring/*` endpoints with 30-second polling.

---

## 3. Notification

**Path:** `src/notification/`  
**Systemd unit:** `notification_bot.service`  
**Entry point:** `src/notification/notification_db_centric_bot.py`

### Purpose
DB-backed async notification delivery system. The bot polls the `msg_messages`
PostgreSQL table and delivers messages via Telegram and/or email channels.
Completely decoupled from the caller — producers write a row, the bot picks it up.

### Architecture
```
Producer (any module)
  └── NotificationService.create_message()
        └── INSERT INTO msg_messages (status=PENDING)
              ↑
        notification_bot polls every N seconds
              ↓
        MessageProcessor
          ├── TelegramChannel  → python-telegram-bot
          └── EmailChannel     → SMTP
              ↓
        UPDATE msg_messages (status=DELIVERED|FAILED)
        INSERT INTO msg_delivery_status (per-channel result)
```

### Key Files
| File | Responsibility |
|------|---------------|
| `notification_db_centric_bot.py` | Main polling loop and service lifecycle |
| `service/database_optimization.py` | Queue health queries, batch fetching |
| `channels/` | Telegram and email channel implementations |
| `logger.py` | Project-wide `setup_logger()`, multiprocessing-safe logging |
| `service_monitor.py` | `ServiceMonitor` — checks systemd service status + logs for the monitoring API |
| `model.py` | `Message`, `MessageDeliveryStatus`, `MessageStatus` enums |

### Message Priority
`LOW` → `NORMAL` → `HIGH` → `CRITICAL`. Critical messages bypass batching.

### Integration Points
- All modules import `NotificationService` and call `create_message()`.
- `src/api/monitoring_routes.py` calls `ServiceMonitor` on each `/api/monitoring/services` request.
- Vector posts to `/internal/log-alert` which calls `NotificationService.create_message()`.

---

## 4. Scheduler

**Path:** `src/scheduler/`  
**Systemd unit:** `scheduler.service`  
**Entry point:** `src/scheduler/main.py`

### Purpose
APScheduler-based pipeline runner. Reads `Schedule` rows from PostgreSQL, fires
jobs on their cron expressions, records `ScheduleRun` rows with status / error / duration.

### Job Types
| Type | Description |
|------|-------------|
| `data_download` | OHLCV data download via DataManager |
| `data_processing` | Screener / indicator computation |
| `ml_pipeline` | ML model training / inference |
| `pnl_alert` | Daily portfolio PnL notification |
| `system` | Health checks, maintenance |

### Key Files
| File | Responsibility |
|------|---------------|
| `scheduler_service.py` | Core APScheduler wrapper, job dispatch, run recording |
| `main.py` | Service entry point, graceful shutdown |
| `config.py` | Scheduler-level configuration |
| `cli.py` | CLI for manual job triggering |
| `deployment/` | Systemd unit file templates |

### Database Tables
- `schedules` — job definitions (name, cron, target, enabled, next_run_at).
- `schedule_runs` — execution history (started_at, finished_at, status, error, job_snapshot).

### Monitoring Integration
`monitoring_routes.py` joins `Schedule` + `ScheduleRun` and exposes them on
`/api/monitoring/pipelines` for the Web UI Pipelines tab.

---

## 5. Data

**Path:** `src/data/`

### Purpose
Market data acquisition, normalisation, caching, and database persistence.
Provides a unified `DataManager` interface with automatic provider fallback.

### Sub-components

#### 5.1 DataManager (`src/data/data_manager.py`)
Unified entry point. Call `dm.get_ohlcv(symbol, timeframe, start, end)` —
internally tries providers in priority order, fills gaps, caches results.

Provider priority: **Tiingo → FMP → Yahoo Finance → DB cache**

#### 5.2 Downloaders (`src/data/downloader/`)
| Downloader | Provider | Notes |
|-----------|---------|-------|
| `tiingo_data_downloader.py` | Tiingo | US equities, ETFs |
| `fmp_data_downloader.py` | FMP (Financial Modelling Prep) | US + international |
| `yahoo_data_downloader.py` | Yahoo Finance | Broad coverage, less reliable |
| `ibkr_flex_downloader.py` | IBKR Flex Query | Position exports |
| `edgar_downloader.py` | SEC EDGAR | Fundamentals, filings |
| `gdelt_downloader.py` | GDELT | News sentiment |
| `finra_downloader.py` | FINRA | Short interest |

All downloaders implement the `BaseDataDownloader` interface with `get_ohlcv(symbol, timeframe, start, end)`.

#### 5.3 Database Layer (`src/data/db/`)
- **Models** (`models/`): SQLAlchemy ORM — `Message`, `User`, `Schedule`, `ScheduleRun`,
  `Strategy`, `TelegramUser`, etc.
- **Repos** (`repos/`): Repository pattern, one class per model.
- **Services** (`services/`): Business-logic wrappers; `DatabaseService` provides
  `uow()` context manager (Unit of Work).
- **Core** (`core/base.py`): `Base` declarative model, engine factory.

#### 5.4 Cache (`src/data/cache/`)
Pipeline-level caching for expensive computations (e.g. ML feature matrices).

#### 5.5 Feeds (`src/data/feed/`)
Live streaming data feeds (Binance WebSocket, IBKR native API, Yahoo polling).

### Integration Points
- `DataManager` used by: `Scheduler`, `Portfolio/PnL alert`, `ML pipelines`, `Strategy`.
- DB models shared across all modules via `src/data/db/`.

---

## 6. Trading

**Path:** `src/trading/`  
**Systemd unit:** `trading.service`

### Purpose
Live trading execution — connects to Interactive Brokers, manages orders and
positions, integrates with the strategy framework.

### Key Files
| File | Responsibility |
|------|---------------|
| `broker/ibkr_broker.py` | IBKR broker implementation via ib_insync |
| `broker/base_broker.py` | Abstract broker interface |
| `risk/post_trade/pnl_attribution.py` | Post-trade PnL breakdown |
| `risk/post_trade/risk_reporting.py` | Risk metrics and reporting |

### IBKR Connection
- **Live trading**: TWS port 7496
- **Paper trading**: Gateway port 4797 (Docker container `ibgateway-docker.service`)
- Connection managed by `IBKRBroker.connect()` / `disconnect()`.

---

## 7. Telegram

**Path:** `src/telegram/`  
**Systemd unit:** `telegram_bot.service`

### Purpose
Telegram bot interface for end-users. Provides market screeners, alert management,
schedule control, and PnL notifications.

### Key Components
| Component | Description |
|-----------|-------------|
| `screener/` | Stock screener commands and results formatting |
| `services/` | Business-logic services called by bot handlers |
| `services/models.py` | Pydantic models for Telegram service layer |

### User Registration Flow
1. User sends `/start` to the bot.
2. Bot creates a `TelegramUser` record with `status=PENDING`.
3. Admin approves via Web UI (Telegram → User Management page).
4. User gains access to commands.

### Integration Points
- `src/api/telegram_routes.py` — Web UI manages users/alerts via REST.
- `src/notification/` — bot receives messages from the notification queue.
- `src/scheduler/` — users trigger pipeline runs via bot commands.

---

## 8. Portfolio

**Path:** `src/portfolio/pnl_alert/`

### Purpose
Daily end-of-day portfolio PnL alert. Merges IBKR positions (from Flex Query XML)
with an optional YAML watchlist, fetches current prices, and sends a Telegram +
email notification if any holding is above a configured threshold.

### Pipeline (runs at 21:30 UTC Mon–Fri)
```
load_ibkr_xml()          ← Open_Positions.xml (IBKR Flex Query export)
load_watchlist()         ← config/watchlist.yaml
merge_holdings()         ← IBKR wins over watchlist on same symbol
fetch_latest_closes()    ← DataManager.get_ohlcv() per symbol
evaluate()               ← compute unrealised PnL %
send_alert()             ← Telegram + email via NotificationService
```

### Key Files
| File | Responsibility |
|------|---------------|
| `ibkr_xml_loader.py` | Parse IBKR Flex Query XML; apply `_IBKR_SYMBOL_MAP` for exchange suffixes |
| `watchlist_loader.py` | Load YAML watchlist |
| `position_aggregator.py` | Merge multi-account positions, weighted-average cost basis |
| `price_fetcher.py` | Per-symbol price fetch with error isolation |
| `pnl_evaluator.py` | PnL % calculation, threshold filtering |
| `notifier.py` | Format and dispatch alert |
| `runner.py` | Async orchestration |
| `config.py` | `PnLAlertConfig` dataclass + YAML loader |

### Symbol Mapping
IBKR exports non-US ETFs without exchange suffix. `_IBKR_SYMBOL_MAP` in
`ibkr_xml_loader.py` translates at parse time:
```python
_IBKR_SYMBOL_MAP = {
    "VUSD": "VUSD.L",   # Vanguard FTSE All-World UCITS ETF (LSE)
}
```
Add entries here for any non-US symbol that gets a 404 from data providers.

### Configuration
`src/portfolio/pnl_alert/config/pnl_alert.yaml` — threshold %, channels, cron,
watchlist path, IBKR XML path, recipient user ID.

---

## 9. ML Pipelines

**Path:** `src/ml/pipeline/`

### Purpose
Machine-learning research and production pipelines for regime detection, signal
generation, and anomaly detection.

### Pipelines
| Pipeline | Path | Description | Status |
|----------|------|-------------|--------|
| P00 | `p00_hmm_3lstm/` | HMM regime detection + 3-LSTM per-regime training | Complete |
| P01 | `p01_hmm_lstm/` | HMM + single LSTM variant | Complete |
| P03 | `p03_cnn_xgboost/` | CNN embedding → XGBoost classifier | Complete |
| P04 | `p04_short_squeeze/` | Short-squeeze scanner (FINRA short interest + price action) | Complete |
| P06 | (scheduler job) | Data download / feature pipeline | Active |
| P10 | (scheduler job) | Data download / feature pipeline | Active |
| P15 | `p15_gdelt_sentiment/` | GDELT news sentiment + intraday | Deferred (needs DuckDB/ChromaDB) |
| P16 | (scheduler job) | New pipeline | Active |
| P17 | (scheduler job) | New pipeline | Active |
| P18 | (scheduler job) | Institutional Flow Tracker | Implemented, awaiting backfill |

### Common Utilities
- `src/ml/future/hmm_regime_detector.py` — HMM regime detection shared component.
- `src/ml/future/nn_regime_detector.py` — NN-based regime detection.
- `src/indicators/` — Technical indicator library used as features.

---

## 10. Strategy

**Path:** `src/strategy/`

### Purpose
Trading strategy framework. Strategies are Backtrader-compatible classes composed
from entry/exit mixin modules.

### Key Components
- `strategy_core.py` — base abstractions, signal types, risk controls, signal aggregation, regime detection.
- `multi_timeframe_engine.py` — multi-timeframe logic (higher TF trend + lower TF entry).
- `strategy_pack/` — concrete strategy implementations.

### Design Pattern
Strategies compose behaviour from mixins:
```python
class MyStrategy(RSIEntryMixin, ATRExitMixin, BaseStrategy):
    ...
```
This avoids deep inheritance hierarchies and makes parameter tuning straightforward.

---

## 11. Backtester

**Path:** `src/backtester/`

### Purpose
Historical strategy backtesting using Backtrader. Includes performance analytics
and walk-forward optimization support.

### Key Files
| File | Responsibility |
|------|---------------|
| `analyzer/bt_analyzers.py` | Custom Backtrader analyzers (Sharpe, drawdown, etc.) |
| `optimizer/` | Parameter optimisation (grid search, Optuna) |

### Performance Metrics Available
Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, recovery factor,
win rate, profit factor, expectancy.

---

## 12. Indicators

**Path:** `src/indicators/`

### Purpose
Technical indicator library. Wraps TA-Lib and custom implementations into a
consistent interface used by ML pipelines and strategies.

### Key Files
| File | Responsibility |
|------|---------------|
| `adapters/base.py` | Abstract indicator adapter |
| `utils.py` | Indicator calculation helpers |

### Indicator Categories
OHLCV-based: RSI, MACD, Bollinger Bands, ATR, Ichimoku, Stochastic, EMA/SMA/WMA.
Volume-based: OBV, VWAP, volume profile.
Custom: regime-detection features, short-squeeze score.

---

## 13. Common

**Path:** `src/common/`

### Purpose
Shared utilities used across multiple modules. Keeps cross-cutting concerns
out of individual submodules.

### Key Components
| Component | Description |
|-----------|-------------|
| `alerts/cron_parser.py` | Cron expression parser and next-run calculator |
| `health_cli.py` | CLI health-check utility |

---

## 14. Config

**Path:** `src/config/`

### Purpose
Centralised configuration management. Loads environment-specific YAML configs,
validates with Pydantic, and provides a config registry for hot-reload.

### Key Files
| File | Responsibility |
|------|---------------|
| `config_loader.py` | YAML loading, environment detection |
| `registry.py` | Config registry and hot-reload |
| `templates.py` | Config templates for common use cases |
| `migrate_configs.py` | Migration from legacy scattered config files |

### Secrets
Credentials live in `config/donotshare/donotshare.py` (not in version control).
Environment variables are loaded from `config/donotshare/.env`.

---

## 15. Error Handling

**Path:** `src/error_handling/`

### Purpose
Resilience infrastructure: circuit breaker, retry manager, custom exception
hierarchy, error monitoring.

### Components
| Component | Description |
|-----------|-------------|
| `exceptions.py` | Custom exception hierarchy with rich context |
| `circuit_breaker.py` | Circuit breaker (CLOSED → OPEN → HALF_OPEN state machine) |
| `retry_manager.py` | Configurable retry with exponential backoff |
| `recovery_manager.py` | Recovery strategies: fallback, degrade, ignore, alert |

### Usage Pattern
```python
from src.error_handling.retry_manager import RetryManager

retry = RetryManager(max_retries=3, backoff_base=2.0)
result = retry.execute(my_api_call, args=(...))
```

---

## 16. Monitoring (Infrastructure)

### Purpose
End-to-end log monitoring and alerting for the Raspberry Pi production server.

### Components

#### 16.1 Vector (Log Collector)
- Rust binary running as `vector.service` on the Pi.
- Reads journald (all systemd units) and Docker container logs.
- Filters lines matching `error|exception|critical|traceback|fatal` (case-insensitive).
- Deduplicates using a stable fingerprint (strips numbers/UUIDs/timestamps).
- Throttle: max 1 alert per unique fingerprint per 30 minutes.
- Posts to `http://127.0.0.1:5003/internal/log-alert`.

#### 16.2 ServiceMonitor (`src/notification/service_monitor.py`)
- Called by the API on each `/api/monitoring/services` request.
- Runs `systemctl is-active` per service and `journalctl --since "10 minutes ago"` for error patterns.
- Error patterns (intentionally narrow to avoid false positives):
  ```
  Traceback (most recent call last)
  SyntaxError
  - CRITICAL -        ← log-level token only, not the word in a message
  SystemExit
  OOM|out of memory
  ```

#### 16.3 Services Monitored
```
ibgateway-docker.service   IB Gateway (Docker)
notification_bot.service   Notification delivery bot
scheduler.service          Pipeline scheduler
telegram_bot.service       Telegram user-facing bot
trading.service            Live trading execution
trading-webui.service      FastAPI backend + React frontend
```

### Alert Flow
```
Vector / ServiceMonitor
  → POST /internal/log-alert (localhost)
    → NotificationService.create_message()
      → msg_messages table
        → notification_bot.service
          → Telegram alert to admin
```

---

## Cross-Module Dependency Map

```
Config ──────────────────────────────────► All modules
Error Handling ──────────────────────────► All modules
Data (DB layer) ─────────────────────────► API, Notification, Scheduler, Portfolio, Telegram
Data (DataManager) ──────────────────────► Scheduler, Portfolio, ML, Strategy
Notification ────────────────────────────► API (internal route), Scheduler (alerts), Portfolio
Scheduler ───────────────────────────────► ML pipelines, Data downloads, Portfolio PnL alert
API ─────────────────────────────────────► Web UI (serves static), WebSocket clients
Telegram bot ────────────────────────────► Notification (receives), API (managed via REST)
Trading ─────────────────────────────────► Strategy, Backtester (shares broker abstraction)
```
