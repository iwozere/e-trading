# Trading System REST API Documentation

## Overview

This document provides comprehensive API documentation for the Advanced Trading Framework REST API, including:
- Complete endpoint reference for all system components
- Authentication and authorization methods
- Request/response models and examples
- WebSocket real-time communication

**API Location**: `src/api/` - Dedicated FastAPI backend module  
**Base URL**: `http://localhost:8000` (configurable via `TRADING_API_PORT`)  
**Documentation**: Auto-generated OpenAPI/Swagger at `/docs`

---

## Authentication

The API uses JWT (JSON Web Token) based authentication with role-based access control:

### Authentication Flow
1. **Login**: `POST /auth/login` with username/password
2. **Receive Tokens**: Access token (30min) + Refresh token (7 days)
3. **API Access**: Include `Authorization: Bearer <access_token>` header
4. **Token Refresh**: `POST /auth/refresh` with refresh token when access token expires

### User Roles
- **Admin**: Full system access including user management and system administration
- **Trader**: Trading operations, strategy management, and monitoring access
- **Viewer**: Read-only access to system status and reports

### Authentication Endpoints
- `POST /auth/login` - User authentication with JWT tokens
- `POST /auth/refresh` - Token refresh using refresh token
- `POST /auth/logout` - Session termination and token invalidation
- `GET /auth/me` - Current user profile information

---

## Bot Management

### Start a Bot

- **POST** `/start_bot` (api.py)  
- **POST** `/api/bots` (webgui/app.py)

**Request JSON:**
```json
{
  "strategy": "rsi_boll_volume",   // Name of the strategy (required)
  "id": "mybot1",                // Optional bot ID (if not provided, auto-generated)
  "config": {                     // Bot configuration (strategy-specific)
    "trading_pair": "BTCUSDT",
    "initial_balance": 1000.0,
    "...": "..."
  }
}
```

**Response:**
```json
{
  "message": "Started bot for rsi_boll_volume.",
  "bot_id": "mybot1"
}
```
or
```json
{
  "status": "success",
  "bot_id": "mybot1"
}
```

---

### Stop a Bot

- **POST** `/stop_bot` (api.py)  
- **DELETE** `/api/bots/<bot_id>` (webgui/app.py)

**Request JSON (api.py):**
```json
{
  "bot_id": "mybot1"
}
```

**Response:**
```json
{
  "message": "Stopped bot mybot1."
}
```
or
```json
{
  "status": "success"
}
```

---

### Get Status of All Bots

- **GET** `/status` (api.py)  
- **GET** `/api/bots` (webgui/app.py)

**Response:**
```json
{
  "mybot1": "running",
  "mybot2": "running"
}
```
or
```json
[
  {
    "id": "mybot1",
    "status": "running",
    "active_positions": 0,
    "portfolio_value": 1000.0
  }
]
```

---

### Get Trades for a Bot

- **GET** `/trades?bot_id=mybot1` (api.py)  
- **GET** `/api/bots/<bot_id>/trades` (webgui/app.py)

**Response:**
```json
[
  {
    "bot_id": 123456,
    "pair": "BTCUSDT",
    "type": "long",
    "entry_price": 10000,
    "exit_price": 10500,
    "size": 1,
    "pl": 5.0,
    "time": "2024-06-01T12:00:00"
  }
]
```

---

### Get Bot Logs

- **GET** `/log?strategy=rsi_boll_volume` (api.py)

**Response:**
```json
{
  "log": "Last 20 lines of log file..."
}
```

---

### Backtest a Strategy

- **POST** `/backtest` (api.py)

**Request JSON:**
```json
{
  "strategy": "rsi_boll_volume",
  "ticker": "BTCUSDT",
  "tf": "1h"
}
```

**Response:**
```json
{
  "message": "Backtesting rsi_boll_volume on BTCUSDT (1h)... [stub]"
}
```

---

### Bot Configuration (webgui/app.py only)

- **GET** `/api/config/bots` — List available bot configs
- **GET** `/api/config/bots/<bot_id>` — Get config for a bot
- **POST** `/api/config/bots/<bot_id>` — Save config for a bot
- **GET** `/api/config/bots/<bot_id>/parameters` — Get parameter template for a bot type
- **GET** `/api/config/bots/<bot_id>/archive` — Get archived configs for a bot

---

## API Documentation Generation

This project supports auto-generating API documentation from Python docstrings using either **Sphinx** or **MkDocs**.

### Using Sphinx

#### 1. Install Sphinx and Extensions
```bash
pip install sphinx sphinx-autodoc-typehints sphinx_rtd_theme
```

#### 2. Initialize Sphinx in the `docs/` Folder
```bash
cd docs
sphinx-quickstart
```
- Answer the prompts (project name, author, etc.).
- When asked, enable autodoc and type hints extensions.

#### 3. Configure Sphinx for Autodoc
- In `docs/conf.py`, add or ensure:
```python
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]
html_theme = 'sphinx_rtd_theme'
```

#### 4. Add API Reference to `index.rst`
Add this to your `index.rst`:
```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules
```

#### 5. Generate API Stubs
```bash
sphinx-apidoc -o api ../src
```

#### 6. Build the Docs
```bash
make html
```
- The HTML docs will be in `docs/_build/html/`.

### Using MkDocs (with mkdocstrings)

#### 1. Install MkDocs and Plugins
```bash
pip install mkdocs mkdocstrings[python] mkdocs-material
```

#### 2. Create or Edit `mkdocs.yml`
Example minimal config:
```yaml
site_name: Crypto Trading Platform API
nav:
  - Home: index.md
  - API Reference: api.md
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
```

#### 3. Create `docs/api.md`
Add:
```markdown
# API Reference

::: src.strategy.base_strategy
::: src.trading.base_trading_bot
# (Add more modules/classes as needed)
```

#### 4. Build the Docs
```bash
mkdocs build
```
- The HTML docs will be in `site/`.

#### 5. Serve Locally
```bash
mkdocs serve
```
- Visit `http://127.0.0.1:8000/` in your browser.

### Documentation Tips

- Keep your code docstrings up to date for best results.
- Regenerate docs after code changes.
- You can deploy the generated HTML to GitHub Pages or any static site host.

For more details, see the [Sphinx](https://www.sphinx-doc.org/) and [MkDocs](https://www.mkdocs.org/) documentation.

---

## Notes

- All endpoints that modify bots require authentication.
- The `strategy` parameter should match the name of the strategy/bot module (e.g., `rsi_boll_volume`).
- The `bot_id` is a unique identifier for each running bot instance. 