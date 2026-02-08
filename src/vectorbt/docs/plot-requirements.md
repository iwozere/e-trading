To reach your **50%+ APR** goal while utilizing high leverage in futures, your reports need to move beyond simple PnL. You need to visualize **survivability** and **efficiency**.

Here are the specific metrics and plot types I recommend including in the requirements for your agent to implement in `src/vectorbt/`:

---

## 1. Essential Performance Metrics

The agent should configure the `pf.stats()` output to include these specific institutional-grade crypto metrics:

* **Calmar Ratio (Primary):** Since you are targeting 50%+ profit, this is your most important metric. It measures . A Calmar > 2.0 is the "gold standard" for leveraged crypto strategies.
* **Sortino Ratio:** Unlike Sharpe, Sortino only penalizes "bad" (downside) volatility. This is crucial for crypto because you want to keep the "good" volatility (huge pumps).
* **Max Gross Exposure:** In futures, you need to track how much of your total collateral is tied up. If this hits 100%, you are at high risk of a margin call.
* **Profit Factor:** Total Gross Profit / Total Gross Loss. Aim for > 1.5.
* **Expectancy:** The average USDT you expect to make per trade. If this is smaller than your exchange fees + slippage, the strategy will fail in live trading.

---

## 2. Recommended Plot Types

Vectorbt's Plotly integration allows for interactive dashboards. The requirements should specify these four layouts:

### A. The "Under-the-Hood" Dashboard

This is for technical validation.

* **Cumulative Returns vs. Benchmark:** Compare your strategy against "Buy & Hold BTC."
* **Underwater Plot:** Visualizes drawdowns as a "valley." It shows not just how deep the loss was, but how long it took to recover (Time to Recovery).
* **Net Exposure:** A line chart showing your current position size ( for full long,  for full short). With leverage, this line will scale (e.g.,  for 5x long).

### B. Liquidation & Margin Risk Plot

* **Distance to Liquidation:** A custom plot showing the gap between the current price and the liquidation price. If this gap narrows to , the report should flag the trial as "High Risk."
* **Margin Usage:** A bar chart showing the percentage of your wallet used as margin over time.

### C. Optimization Heatmaps

To find the "Stable Plateau" (avoiding over-fitting):

* **2D Heatmap:** Plot `EMA_Fast` on the X-axis and `EMA_Slow` on the Y-axis.
* **Leverage Sensitivity Map:** A heatmap showing how Profit and Max Drawdown correlate with increasing leverage. You are looking for the "Sweet Spot" before the risk-to-reward ratio breaks.

### D. Trade Distribution (Histogram)

* **Trade PnL Histogram:** Visualizes the "Fat Tails." You want to see a few "Home Runs" (massive winners) and many small, controlled losses. If you see a "Left Fat Tail" (huge sudden losses), the strategy's stop-loss logic is failing.

---

## 3. Storage & Export Requirements

Since you are using **SQLite** for trials and **PostgreSQL** for production:

* **Best Trial Snapshot:** For the top 5 trials of every study, the agent must save a static `.png` of the equity curve and a full `.html` interactive report.
* **JSON Meta-Data:** The JSON saved to Postgres should include the `Expectancy` and `Max Drawdown` seen during the Walk-Forward period so the live bot can shut itself down if real-world performance deviates too far from the test.

---

### Suggested Requirement Update for your Agent:

> "The Research Module must generate a `Performance_Report.html` for every 'Promoted' strategy. This report must include a subplot of **Cumulative Returns**, **Underwater Drawdowns**, and a **Net Exposure** chart. Additionally, the SQLite database must log the **Calmar Ratio** and **Sortino Ratio** for every trial to facilitate filtering in the `PromoteStrategy` script."
