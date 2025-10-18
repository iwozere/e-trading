---

# Trading System — High-Level Architecture

This document describes the **high-level architecture** of the trading system, built on the **Backtrader** framework.

---

## Overview

* **Framework:** Backtrader
* **Notification Channels:** Telegram, Email
* **Management Channels:** Telegram, Web UI
* **Purpose:** Unified system for automated trading, monitoring, alerts, and analytics.

---

## Web UI

The Web UI provides monitoring and management capabilities for both **trading** and **background services**.

### Trading Module

* Dashboard overview
* Trader management
* Per-trader metrics (P&L, portfolio stats, etc.)
* Portfolio overview and insights

### Background Jobs (Cron)

* Alerts
* Schedules
* Screeners

---

## Source Code Structure

### `src/data/db`

* Database access layer
* **Components:**

  * SQLAlchemy models
  * Repositories
  * Services

### `src/data/downloader`

* Data downloaders for OHLCV and fundamentals
* **Sources:** Binance, Yahoo Finance (`yfinance`), FMP, etc.

### `src/data/feed`

* Live data feeds for **paper** and **live trading**

---

### `src/telegram`

* **`telegram_bot.py`** — system service with a command parser
* Handles:

  * Incoming commands from Telegram bot
  * Command routing:

    * Stores commands in DB for background processing (alerts, schedules)
    * Processes commands directly (reports, screeners)
  * User registration, verification, and approval via DB services

---

### `src/web_ui/backend`

* **FastAPI service**
* REST API layer interacting with the database
* *(Consider moving to `src/api` for clarity)*

---

### `src/web_ui/frontend`

* Frontend for admins and traders
* Enables management of:

  * Trading (strategies, paper/live trading, optimization)
  * Background services (alerts, schedules, feedbacks, etc.)
  * User management
  * Audit logs,broadcas notifications, admin functions etc.

---

### `src/scheduler`

* Background services using **APScheduler** and **croniter**
* Handles commands from Telegram and Web channels:

  * Alerts
  * Schedules

---

### `src/trading`

* **`trading_bot.py`** — core trading system service
* Manages trading bot instances

#### Bot Configuration

* Each bot is defined as a row in the **`trading_bot_instances`** table (PostgreSQL)
* Configuration parameters are stored in a **JSONB** field
* Handles:

  * Loading configuration
  * Starting/stopping bots
  * Managing paper/live modes

---
