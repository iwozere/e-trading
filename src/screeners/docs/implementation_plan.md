# Implementation Plan: IBKR Scalable Screener

**Role:** Senior Python Software Engineer  
**Status:** Draft / Proposed  
**Objective:** Build a robust, modular 15m/1h screener for IBKR that minimizes API costs and reuses existing strategy/broker logic.

---

## 1. Project Structure & New Files
To maintain the existing project cleanliness, we will use the following structure:

```text
src/screeners/
├── discovery/
│   ├── base.py             # IDiscoveryProvider interface
│   ├── portfolio.py        # IBKR holdings discovery
│   └── static.py           # Watchlist (JSON) discovery
├── logic/
│   └── strategy_bridge.py  # Backtrader "Virtual Cerebro" wrapper
├── ibkr_screener_service.py # Main entry point & loop
└── docs/
    ├── architecture.md
    └── implementation_plan.md

src/data/downloader/
└── ibkr_downloader.py      # Inherits BaseDataDownloader

bin/screener/
├── run_screener.sh         # Shell launcher
├── ibkr_screener.service   # Systemd config
└── install_service.sh      # Deployment helper
```

---

## 2. Phased Milestones

### Phase 1: Core Data Infrastructure (`IBKRDownloader`)
**Goal:** High-performance, cached data retrieval.
1.  **Inheritance:** Extends `src.data.downloader.base_data_downloader.BaseDataDownloader`.
2.  **Delayed Data Support:** Implement `ib.reqMarketDataType(3)` as default.
3.  **Black Box Sync:**
    *   `_get_cached_data()`: Read from `DATA_CACHE_DIR` using `pandas`.
    *   `_sync_missing_bars()`: Identify the gap between `last_cached_timestamp` and `now()`.
    *   `get_ohlcv()`: Orchestrate the read-fetch-merge-save-return flow.

### Phase 2: The "Virtual Cerebro" Bridge (`StrategyBridge`)
**Goal:** Zero-rewriting of existing strategies.
1.  **Bridge Class:** Create `ScreenerStrategyBridge`.
2.  **Logic:**
    *   Accepts a `BaseStrategy` class (e.g., `HMMLSTMStrategy`) and its config.
    *   Instantiates a minimal `bt.Cerebro` in `preload` mode.
    *   Loads the DataFrame from Phase 1.
    *   Runs `cerebro.run()` and inspects the last bar's signals via the `entry_mixin` state.
3.  **Parity Ensure:** Automatically handles `warmup_period` by requesting extra data from the Downloader.

### Phase 3: Discovery & Orchestration
**Goal:** Modular symbol selection and the main loop.
1.  **SymbolDiscovery:**
    *   `PortfolioProvider`: Inject `IBKRBroker` to call `get_positions()`.
    *   `StaticProvider`: Load from `data/watchlists/screener.json`.
2.  **ScreenerService:** 
    *   Async loop using `apscheduler` or a simple `asyncio.sleep` wrapper.
    *   Concurrent processing of signals (limited to ~50 symbols).
    *   Result persistence to `results/screeners/ibkr/signals_{timestamp}.json`.

### Phase 4: Notifications & Hardening
**Goal:** User alerts and error recovery.
1.  **Notifier:** Bridge to `src.notification.notification_manager`.
2.  **Formatting:** Create a clean MD/HTML format for Telegram alerts showing:
    *   Symbol & Current Price
    *   Signal Logic (e.g., "LSTM Predicted +0.5% Change")
    *   Indicators (RSI, Regimes)
3.  **Resilience:** Implement exponential backoff for IBKR connectivity issues (TWS/Gateway drops).

---

## 3. Engineering Requirements

### Concurrency Model
*   We will use `asyncio` for the orchestrator to prevent blocking during IBKR networking calls.
*   The `StrategyBridge` (Backtrader) is CPU-bound; if 50 symbols take too long on a Pi, we will utilize `ProcessPoolExecutor` for the actual strategy math.

### Memory Management (Pi Specific)
*   Backtrader instances will be ephemeral. We will explicitly `del` the `Cerebro` and `Strategy` instances after each ticker scan to prevent memory leaks in the long-running service.
*   DataFrames will only be kept in memory for the duration of the scan for that specific ticker.

### Data Validation
*   The `Downloader` must validate OHLCV integrity (check for daily gaps, zero-volume bars) before passing it to the `StrategyBridge`.

---

## 4. Risks & Mitigations
*   **Rate Limits:** IBKR has a "60 requests per 10 minutes" limit for historical data. 
    *   *Mitigation:* The `Orchestrator` will use a `Semaphore(5)` and a delay between batches to ensure we never hit pacing violations.
*   **Stale Cache:** If the Pi is offline for 2 days, the `Downloader` must be able to perform a multi-day "Fill-in" request without overloading the API.

---

## 5. Deployment & Persistence (Linux/Pi)
To ensure the screener survives reboots and operates as a background process on the Raspberry Pi, we will utilize `systemd`.

### Operational Scripts: `bin/screener/`
All deployment-related files will be centralized in `bin/screener/`.

1.  **`run_screener.sh`**:
    *   Handles environmental setup (VirtualEnv activation).
    *   Sets `PYTHONPATH` to the project root.
    *   Executes the orchestrator as a non-blocking process.
2.  **`ibkr_screener.service`**:
    *   A standard Linux unit file.
    *   Configured with `Restart=always` and `RestartSec=60`.
    *   Ensures the service starts only after the network is available.
3.  **`install_service.sh`**:
    *   Automates symlinking the service file and reloading `systemctl`.

---

## 6. Definition of Done (DoD)
- [ ] `IBKRDownloader` successfully merges local CSV with IBKR Delayed data.
- [ ] `StrategyBridge` produces the same signal as the Backtester for a given dataset.
- [ ] Screener runs a full loop of 20 symbols under 2 minutes.
- [ ] Initial signal is successfully delivered to Telegram.
- [ ] Screener automatically restarts after a simulated system reboot.
