This refined architecture moves away from a "monolithic script" toward a **modular, event-driven framework**. In this setup, the `Downloader` acts as a **Data Provider (Hardware Abstraction Layer)**, while the `Screener` acts as the **Orchestrator (Business Logic)**.

### 1. The Modular Architecture

#### A. The Provider (`ibkr_downloader.py`)

* **Responsibility:** Connectivity, Protocol (IBKR), and Data Normalization.
* **Features:**
* `fetch_account_holdings()`: Returns a list of current contracts.
* `fetch_historical_data(contract, duration, timeframe)`: Standardized OHLCV.
* `stream_data(contracts)`: Real-time or delayed updates.
* **Internal Persistence:** Handles the "Check Cache -> Merge -> Save" logic internally so the screener doesn't have to worry about file I/O.



#### B. The Orchestrator (`ibkr_screener.py`)

* **Responsibility:** Initialization and Loop Management.
* **Features:**
* Loads the Downloader.
* Loads pluggable Strategy modules (Exit/Entry logic).
* Configures the interval (e.g., 5 min, 15 min, 1 hour).
* Triggers the "Data Update -> TA-Lib Calculation -> Signal Check" pipeline.



#### C. The Logic Plugin (`strategy_base.py`)

* **Responsibility:** Pure Business Logic.
* **Interface:** Receives a Clean DataFrame + Metadata, returns a Signal (Buy/Sell/Hold/Alert).

---

### 2. Implementation Prototyping (Conceptual)

#### The Downloader Plugin (`src/providers/ibkr_downloader.py`)

```python
class IBKRDownloader:
    def __init__(self, config):
        self.ib = IB()
        self.cache_path = config['cache_dir']

    async def get_active_contracts(self):
        """Returns contracts from IBKR account positions."""
        positions = await self.ib.positionsAsync()
        return [p.contract for p in positions]

    async def sync_and_get_data(self, contract, timeframe):
        """
        The 'Black Box' logic: 
        1. Read CSV 
        2. Fetch missing bars from IBKR
        3. Merge and Save
        4. Return full DataFrame
        """
        # ... logic as discussed in data-management.md ...
        return df

```

#### The Screener Orchestrator (`src/screeners/ibkr_screener.py`)

```python
class Screener:
    def __init__(self, downloader, strategy, interval_minutes=15):
        self.downloader = downloader
        self.strategy = strategy
        self.interval = interval_minutes * 60

    async def run(self):
        await self.downloader.connect()
        
        while True:
            # 1. Get current holdings
            contracts = await self.downloader.get_active_contracts()
            
            for contract in contracts:
                # 2. Downloader handles the heavy lifting of caching/fetching
                df = await self.downloader.sync_and_get_data(contract, "15 min")
                
                # 3. Pluggable Logic
                signal = self.strategy.analyze(df)
                
                if signal:
                    self.execute_signal(signal)
            
            await asyncio.sleep(self.interval)

```

---

### 3. Benefits of this Approach

1. **Isolation:** If you decide to switch from IBKR to Alpaca or Binance, you only replace the `Downloader` plugin. The `Screener` and `Strategy` remain untouched.
2. **Configurability:** The interval and the list of stocks are decoupled. You can run one instance of the screener for "Long Term" (1 day bars) and another for "Scalping" (5 min bars) using the same downloader class.
3. **Cross-Platform Performance:** On your Raspberry Pi SSD, the `Downloader` can be optimized for SQLite or high-speed CSV writes without cluttering the "Trading Logic" code.
4. **Testability:** You can pass "Fake Data" to the `Screener` to backtest your `Strategy` plugin without ever connecting to IBKR.

---

### Next Step for the Prompt

We need to finalize the **Interface** (the names of the methods that the Downloader *must* have).
