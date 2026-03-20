# Project Ports and Service Endpoints

This document maps all the ports and service endpoints used across the trading system for both development and production environments.

## Core Services

| Service | Port | Configuration Key | Notes |
|---------|------|-------------------|-------|
| **Web UI Backend (API)** | **5003** | `TRADING_API_PORT` | FastAPI application. Used by both the Web UI and the Telegram Bot's internal client. |
| **Web UI Frontend (Dev)** | **5002** | `TRADING_WEBGUI_PORT` | Vite development server port. In production, static files are served by the backend or Nginx. |
| **Telegram Bot API** | **5004** | `TELEGRAM_API_PORT` | Internal HTTP API for the Telegram bot to receive notifications/broadcasts from the system. |
| **PostgreSQL** | **5432** | `POSTGRES_PORT` | Main database for strategies, trades, and users. |
| **Redis** | **6379** | `REDIS_PORT` | Used for async tasks and caching if enabled. |

## External Integrations

| Integration | Port | Configuration Key | Notes |
|-------------|------|-------------------|-------|
| **IBKR TWS (Live)** | **7496** | `IBKR_PORT` | Interactive Brokers Trader Workstation (Live connection) |
| **IBKR Gateway (Paper)** | **4797** | `IBKR_PAPER_PORT` | Interactive Brokers Gateway (Paper trading connection) |
| **SMTP (Email)** | **587/465** | `SMTP_PORT` | Outgoing email notifications. |

## Development Environment Specifics

When running the system locally on Windows 11 using `python src/web_ui/run_web_ui.py --dev`:

- **Vite Frontend**: starts on `http://localhost:5002`
- **FastAPI Backend**: starts on `http://localhost:5003` (must match Vite's proxy)
- **Vite Proxy**: Configured in `vite.config.ts` to redirect `/api`, `/auth`, and `/ws` to `http://localhost:5003`.

> [!IMPORTANT]
> If you run the backend on a different port (like the default 8000), you **must** update the proxy target in `src/web_ui/frontend/vite.config.ts` or the frontend will not be able to communicate with the API.
