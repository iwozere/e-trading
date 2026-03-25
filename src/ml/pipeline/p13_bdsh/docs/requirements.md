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

Adding a hard stop-loss and visual markers is the best way to separate a "backtest" from a "trading tool." The triangles will help you visually audit whether the VIX was actually timely or if you were catching a knife for too long.

Here are the specific updates for your `requirements.md`.

---

## Updated requirements.md (Additions)

### R5: Risk Management (Volatility-Adjusted Stop-Loss)
* **ATR Stop-Loss:** Implement a volatility-adjusted stop-loss using the formula: $StopLoss = 2 \times ATR(14)$.
* **Logic:** If the current price of a ticker drops below the **Average Acquisition Price** minus $2 \times ATR$ (measured at the time of entry/scaling), the system triggers an immediate "SELL ALL".
* **Cooldown:** After a stop-loss is triggered, the system should not re-enter that specific ticker until the VIX Z-score resets (crosses below 0.0) and generates a new Buy Tier 1 signal.

### R6: Enhanced Visualization (Signals)
* **Price Charting:** Use `matplotlib` or `plotly` to plot the `Adj Close` price.
* **Trade Markers:**
    * **Buy Signals:** Render **Green Upward Triangles** (`^`) at the price point for every Tier entry (1, 2, and 3). 
    * **Sell Signals:** Render **Red Downward Triangles** (`v`) at the price point for a VIX-based exit.
    * **Stop-Loss Signals:** Render **Black "X" Markers** if a trade was closed via the R5 stop-loss logic.
* **Subplots:** Create a dual-axis chart:
    1.  Top Plot: Share Price with Trade Markers.
    2.  Bottom Plot: VIX Z-Score with horizontal lines at thresholds (1.5, 2.5, 3.5).

### R7: Production Signaling Logic
* **State Persistence:** The script must save the current "Active Tiers" and "Average Entry Price" to a local `state.json` file.
* **Daily Output:** On execution, the script should print a summary:
    > **Ticker:** [AAPL] | **Current VIX Z:** [2.1] | **Status:** [HOLD - Tier 1 Active] | **Stop-Loss:** [$185.20]

---

### Implementation Note for the Dev Agent:
To handle the markers correctly in `matplotlib`, the agent should use a logic similar to this snippet to ensure the triangles don't overlap and are clearly visible:

```python
# Logic Example for the Developer:
plt.plot(df.index, df['Adj Close'], label='Price', alpha=0.5)
plt.scatter(df.index[df['buy_signal']], df['Adj Close'][df['buy_signal']], 
            marker='^', color='g', label='Entry', s=100)
plt.scatter(df.index[df['sell_signal']], df['Adj Close'][df['sell_signal']], 
            marker='v', color='r', label='Exit', s=100)
```

### Strategic Consideration:
The **10% stop-loss** is a double-edged sword with the VIX. During high-volatility events (like 2020), price swings of 10% are common before the recovery. You might want to ask the developer to make the stop-loss **"Volatility Adjusted"** (e.g., $StopLoss = 2 \times ATR$) so you don't get shaken out of a good trade just because the market is "normally" wild.

Does a flat 10% stop-loss feel right for your risk tolerance, or should we make it wider to give the VIX room to work?