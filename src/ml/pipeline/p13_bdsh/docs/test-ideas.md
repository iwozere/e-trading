To test your VIX strategy effectively over 20 years, you must avoid **Survivorship Bias**. If you only test stocks that are famous today (like Nvidia or Apple), your results will be artificially high because you’ve already filtered out the companies that failed.

A "true" test needs a mix of **Survivors**, **Stagnators**, and **Losers**. Here is how to select them for your Python pipeline:

### 1. The "Survivors" (High Beta Growth)
These stocks move aggressively. Since your strategy buys on VIX spikes (fear), these are the tickers that usually "tank" the hardest during a crash but "rip" the highest during the recovery.
* **NVDA (Nvidia):** The gold standard of the last 20 years.
* **AMZN (Amazon):** Survives multiple massive drawdowns to reach new highs.
* **AAPL (Apple):** A consistent performer across cycles.
* **NFLX (Netflix):** High volatility, perfect for testing VIX-based entry points.

### 2. The "Losers" (Value Destroyers & Bankruptcies)
You need to see if your VIX strategy would have accidentally "bought the dip" on a company that never came back.
* **F (Ford):** It survived 2008, but its 20-year chart is a rollercoaster of pain. It tests if your strategy can handle "fake" recoveries.
* **C (Citigroup):** A "Blue Chip" that lost ~90% of its value in 2008 and never recovered to its pre-crash highs.
* **GE (General Electric):** Once the biggest company in the world, it spent 20 years in a slow, painful decline. 
* **INTC (Intel):** A giant that stagnated while its peers soared.
* **Failed Tickers (Manual Entry):** To truly test losers, you may need to look at companies that were delisted. While `yfinance` makes this hard (since they remove delisted tickers), you can simulate this by testing **OTC tickers** or companies like **LU (Lucent)** or **Sears** if your data provider supports historical delisted data.

### 3. The "Stagnators" (Defensive/Low Vol)
These stocks don't react as wildly to the VIX. Testing these helps you see if the strategy is even worth the effort compared to just holding them.
* **KO (Coca-Cola):** Low volatility, steady dividends.
* **WM (Waste Management):** "Boring" business that tends to hold up during VIX spikes.
* **JNJ (Johnson & Johnson):** The classic defensive play.

---

### Selection Strategy for your `requirements.md`
Tell your development agent to include a **"Ticker Universe"** dictionary. This allows you to categorize your tests:

| Category | Tickers to Input | Purpose of the Test |
| :--- | :--- | :--- |
| **The Aggressive** | `NVDA`, `TSLA`, `AMD` | Can the strategy capture "V-shaped" recoveries? |
| **The Reliable** | `SPY`, `QQQ`, `BRK-B` | Does the strategy beat a simple index fund? |
| **The Traps** | `C`, `GE`, `WBA` | Does the VIX signal trick you into buying a "dead" stock? |
| **The Defensive** | `XLU` (Utilities), `GLD` (Gold) | Does the VIX correlation even exist for non-tech stocks? |

### Pro-Tip for 20-Year Accuracy
When selecting your "Losers," look for **"Historical S&P 500 Constituents."** In 2006, the top stocks were companies like **ExxonMobil (XOM)**, **Citigroup (C)**, and **AIG**. In 2026, the list is dominated by Tech. Testing your strategy on **AIG** or **C** from 2007-2009 is the "Final Boss" level for your Python pipeline—if the strategy survives the 2008 Financial Crisis on bank stocks, you have a winner.
