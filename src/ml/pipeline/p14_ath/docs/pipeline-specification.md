This is a classic quantitative finance task often used to analyze "drawdown recovery" and volatility patterns. Identifying these peaks and troughs sequentially allows you to see how a stock behaves after it hits a milestone.

Below is the pipeline specification designed for an AI agent to execute this logic.

---

## Pipeline Specification: Sequential ATH & Drawdown Analysis

### 1. Objective
To extract sequential All-Time Highs (ATH) and their subsequent maximum drawdowns for a list of tickers over a 10-year period, resulting in a structured dataset and a clear visual timeline.

### 2. Data Requirements
*   **Timeframe:** `Today - 10 Years` to `Today`.
*   **Data Source:** Yahoo Finance (`yfinance`) or similar API.
*   **Price Type:** Adjusted Close (to account for dividends and splits).

### 3. Core Logic (The "Greedy" Peak-Trough Algorithm)
The agent must follow this state-machine logic for each ticker:
1.  **Initialize:** Set `global_ath = 0`.
2.  **Find ATH:** Iterate through the time series. If `Price[i] > global_ath`, mark this as a potential new ATH.
3.  **Find Max Dropdown:** Once a new ATH is established, monitor all subsequent prices until a *higher* ATH is reached. 
4.  **Capture Trough:** Identify the absolute lowest price point reached between $ATH_{n}$ and $ATH_{n+1}$.
5.  **Record:** Log the ATH details and the specific max dropdown details found in that window.



---

### 4. Technical Stack
*   **Language:** Python 3.x
*   **Libraries:** `pandas`, `yfinance`, `matplotlib` (or `plotly` for interactive charts).
*   **Output:** `ath_drawdown_analysis.csv`.

### 5. CSV Schema
Each row in the exported file should contain:
| Column | Description |
| :--- | :--- |
| `Ticker` | The stock symbol. |
| `ATH_Date` | Date the ATH was achieved. |
| `ATH_Price` | Price at that ATH. |
| `Max_Drop_Date` | Date of the lowest point before the next ATH. |
| `Max_Drop_Price` | The price at that lowest point. |
| `Days_To_Drop` | Integer count of days between ATH and Max Drop. |

---

### 6. Visualization Requirements
*   **Primary Line:** Closing price of the ticker.
*   **ATH Markers:** Green upward triangles (`^`) at every identified sequential ATH.
*   **Dropdown Markers:** Red downward triangles (`v`) at the lowest point of each drawdown period.
*   **Styling:** Grid enabled, Y-axis in log scale (optional, but recommended for 10-year views).

---

### 7. Execution Python Snippet (Logic Template)
The agent should implement the loop logic roughly as follows:

```python
import yfinance as yf
import pandas as pd

def analyze_ticker(ticker):
    df = yf.download(ticker, period="10y")['Adj Close']
    results = []
    
    current_ath_val = 0
    current_ath_date = None
    
    # Logic to iterate and find sequential peaks and the deepest 
    # valley before the next peak exceeds the previous one.
    # ... (Agent to implement iteration)
    
    return pd.DataFrame(results)
```