## 2. requirements.md

### Technical Environment
* **Language:** Python 3.9+
* **Required Libraries:** `pandas`, `numpy`, `yfinance`, `matplotlib`, `quantstats` (for tear sheets).

### Data Requirements
* **Source:** Yahoo Finance API (`yfinance`).
* **Timeframe:** Daily (1d) resolution.
* **History:** Minimum 20 years (2006 – 2026).
* **Columns:** Use `Adj Close` for all price calculations to ensure dividends/splits are included.

### Functional Requirements
* **R1 (Data Handling):** Handle missing data (NaNs) by forward-filling stock prices but dropping dates where VIX data is missing.
* **R2 (Scaling Logic):** The engine must support an arbitrary number of "Entry Tiers." The developer should make the Z-score thresholds and investment percentages configurable in a dictionary.
* **R3 (Portfolio Simulation):** * Assume an initial capital of \$100,000.
    * Apply a 0.1% slippage/commission fee per trade to simulate realistic costs.
* **R4 (Output):** * Generate a CSV of trade logs (Date, Ticker, Action, Z-Score, Price).
    * Generate a Matplotlib chart comparing Cumulative Returns.

### Error Handling & Validation
* Validate that the ticker exists before attempting download.
* Ensure the Z-score lookback window (e.g., 30 days) does not create "look-ahead bias" (always use `.shift(1)` for signals to ensure we trade on the next day's open or current day's close).

---

### Pro-Tip for your Dev Agent:
Ask the agent to implement **"Volatility Normalization."** In years like 2008 or 2020, the Z-score can stay elevated for weeks. The agent should ensure the code doesn't "re-buy" every day the Z-score is high, but rather "checks" if the target exposure is already met.