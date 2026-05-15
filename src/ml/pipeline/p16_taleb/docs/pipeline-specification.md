# Technical Specification: Taleb Barbell Put Options Pipeline
### S&P 500 Deep OTM Put Strategy — Backtesting, Optimization & Visualization
**Version:** 1.0  
**Format:** Jupyter Notebook + Python modules  
**Language:** Python 3.10+

---

## 1. Objective

Build a full quantitative pipeline that:

1. Ingests historical OHLCV (S&P 500), VIX, GDELT, and optional options-chain data
2. Prices deep out-of-the-money put options using Black-Scholes (and optionally market prices)
3. Simulates systematic monthly purchase of OTM puts across a grid of strike levels
4. Optimizes strike selection across multiple criteria (ROI, Sharpe-like ratio, max drawdown coverage)
5. Produces interactive visualizations of cumulative P&L, premium decay, and crisis payoffs
6. Exports a structured results report

The strategy is modeled after Nassim Taleb's barbell approach: small recurring premium bleed offset by convex payoff during tail events.

---

## 2. Data Sources

### 2.1 Primary Market Data (OHLCV + VIX)

Use your existing connectors to pull:

| Dataset | Ticker / Symbol | Frequency | Period |
|---|---|---|---|
| S&P 500 | `^GSPC` or `SPY` | Daily | 2010-01-01 → present |
| VIX Index | `^VIX` | Daily | 2010-01-01 → present |
| 3-Month Treasury (risk-free rate) | `^IRX` | Daily | 2010-01-01 → present |

Expected OHLCV columns: `date, open, high, low, close, volume`  
Store as: `data/raw/sp500_daily.parquet`, `data/raw/vix_daily.parquet`

### 2.2 Historical Options Price Data

This is the hardest part. Ranked by quality:

#### Tier 1 — Best quality (paid)
- **CBOE DataShop** — `datashop.cboe.com`  
  SPX options data from 2004. Full bid/ask chains, implied vol surface. Pay-per-dataset.  
  Download format: CSV with columns `[quote_date, expiration, strike, option_type, bid, ask, iv, delta, gamma, vega, theta, open_interest]`

- **OptionMetrics / Ivy DB** — academic access via university or direct subscription  
  Most complete dataset. Greeks, IV, forward prices all included.

- **Polygon.io Options API** — `api.polygon.io/v3/snapshot/options/{underlyingAsset}`  
  Historical snapshots available. ~$200/month for full history.  
  Endpoint for historical chain: `GET /v3/snapshot/options/SPY?date=YYYY-MM-DD`

#### Tier 2 — Medium quality (lower cost)
- **Nasdaq Data Link** — `data.nasdaq.com`, package `OPT/SPX`  
  EOD options data. Suitable for daily-resolution backtesting.

- **Tradier API** — `developer.tradier.com`  
  Historical options chains via REST. Requires brokerage account.

#### Tier 3 — Free / Approximation
- **CBOE VIX term structure archive** — `cboe.com/tradable_products/vix/vix_historical_data`  
  Only aggregate vol metrics, not individual strikes. Use for vol surface calibration.

- **Black-Scholes synthetic pricing (default fallback)**  
  When no market data is available, compute theoretical option prices using:
  - `sigma = VIX / 100` as the implied volatility proxy  
  - This is the standard approach in academic research and is a valid approximation  
  - VIX is defined as the 30-day implied vol of SPX options, so it anchors BS pricing correctly

**Implementation rule:** The pipeline must support both modes — real market prices when available, BS synthetic prices as fallback. The mode is controlled by a config flag `USE_MARKET_PRICES = True/False`.

### 2.3 GDELT Data (optional signal layer)

GDELT provides a daily global event tone score that can serve as a macro stress indicator.

- **Source:** `data.gdeltproject.org` or BigQuery `gdelt-bq.gdeltv2.events`
- **Key fields:**  
  - `GoldsteinScale` — event stability score (−10 to +10)  
  - `AvgTone` — average media tone  
  - `NumArticles` — media attention volume  
- **Use in pipeline:** As an optional feature for regime detection (high tension → raise put allocation)
- **Frequency:** Daily aggregated
- **Storage:** `data/raw/gdelt_daily.parquet`

---

## 3. Repository Structure

```
taleb_puts/
├── data/
│   ├── raw/                    # Downloaded source data (parquet)
│   │   ├── sp500_daily.parquet
│   │   ├── vix_daily.parquet
│   │   ├── rates_daily.parquet
│   │   ├── options_chains/     # Optional: real market option prices
│   │   │   └── spx_options_YYYY.parquet
│   │   └── gdelt_daily.parquet
│   └── processed/
│       ├── master_daily.parquet    # Merged feature dataset
│       └── simulation_results/     # Per-strike simulation outputs
├── src/
│   ├── data_loader.py          # Data ingestion and validation
│   ├── features.py             # Feature engineering (drawdown, regime, etc.)
│   ├── pricing.py              # Black-Scholes pricer + Greeks
│   ├── simulator.py            # Barbell strategy simulation engine
│   ├── optimizer.py            # Strike grid optimization
│   └── visualizer.py           # Plotly chart library
├── config.yaml                 # All parameters (see Section 5)
├── notebook.ipynb              # Main Jupyter notebook (see Section 6)
└── report/
    └── output.html             # Exported results
```

---

## 4. Module Specifications

### 4.1 `src/pricing.py` — Option Pricing Engine

#### Black-Scholes Put Pricer

```python
def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Parameters
    ----------
    S     : current spot price
    K     : strike price
    T     : time to expiry in years (e.g. 90/252)
    r     : annualized risk-free rate (e.g. 0.04)
    sigma : annualized implied volatility (e.g. VIX/100)
    
    Returns
    -------
    float : theoretical put price
    """
```

#### Greeks (required for position sizing and risk management)

Implement all five Greeks as a single function returning a dict:

```python
def bs_greeks(S, K, T, r, sigma) -> dict:
    # Returns: {"delta": float, "gamma": float, "theta": float,
    #           "vega": float, "rho": float}
```

#### Volatility Surface Interpolation (optional, Tier 1 data only)

If real options data is available, build a simple vol surface interpolator:

```python
def build_vol_surface(options_df: pd.DataFrame, date: str) -> callable:
    """
    Given a DataFrame of real options on a specific date,
    return a function vol_surface(K, T) -> implied_vol
    using scipy.interpolate.RectBivariateSpline or similar.
    """
```

### 4.2 `src/features.py` — Feature Engineering

All features computed on the daily master dataframe. Required columns:

| Feature | Formula | Description |
|---|---|---|
| `drawdown` | `(close - close.cummax()) / close.cummax()` | Rolling drawdown from peak |
| `ret_1d` | `close.pct_change(1)` | 1-day return |
| `ret_5d` | `close.pct_change(5)` | 1-week return |
| `ret_21d` | `close.pct_change(21)` | 1-month return |
| `ret_63d` | `close.pct_change(63)` | 1-quarter return |
| `vix_ma20` | `vix.rolling(20).mean()` | VIX 20-day moving average |
| `vix_regime` | `pd.cut(vix, bins=[0,15,20,30,40,100])` | VIX regime label |
| `vol_ratio` | `vix / vix_ma20` | VIX vs its own average (vol-of-vol signal) |
| `gdelt_tone_ma5` | `gdelt_avgtone.rolling(5).mean()` | Smoothed global media tone |
| `stress_flag` | `(vix > 30) \| (drawdown < -0.10)` | Binary stress indicator |

### 4.3 `src/simulator.py` — Simulation Engine

Core function signature:

```python
def simulate_barbell(
    df: pd.DataFrame,
    moneyness: float = 0.85,       # K/S — e.g. 0.85 = 15% OTM
    T_days: int = 90,              # Option expiry in calendar days
    rebalance_days: int = 21,      # How often to buy new puts (21 = monthly)
    put_budget_pct: float = 0.02,  # % of capital per period spent on puts
    initial_capital: float = 100_000,
    r_col: str = "rate_3m",        # Column name for risk-free rate
    sigma_col: str = "vix",        # Column to use as IV proxy
    use_market_prices: bool = False,  # If True, look up real prices
    options_df: pd.DataFrame = None,  # Real options data (optional)
) -> pd.DataFrame:
    """
    Returns one row per rebalance period with columns:
    date, S, K, T, sigma_used, put_price, put_price_pct_of_S,
    budget_spent, num_contracts, S_at_expiry, drawdown_at_expiry,
    intrinsic_value_at_expiry, payoff, pnl, pnl_pct,
    cum_cost, cum_payoff, cum_pnl
    """
```

#### Edge cases to handle:
- If `put_price <= 0` or `put_price > S * 0.15`: skip period and log warning
- If expiry date falls on weekend/holiday: roll to next business day
- If `use_market_prices=True` but price not found: fall back to BS and flag row with `price_source = "bs_fallback"`

### 4.4 `src/optimizer.py` — Strike Optimization

Run `simulate_barbell` across a parameter grid and compute summary statistics:

```python
def optimize_strikes(
    df: pd.DataFrame,
    moneyness_grid: list = None,      # Default: np.arange(0.70, 0.97, 0.01)
    T_days_grid: list = [60, 90, 120],
    rebalance_days_grid: list = [21],
    budget_pct: float = 0.02,
    initial_capital: float = 100_000,
) -> pd.DataFrame:
    """
    Returns summary DataFrame with one row per (moneyness, T_days, rebalance_days):
    
    moneyness, strike_otm_pct, T_days,
    total_cost, total_payoff, total_pnl, net_roi_pct,
    win_rate_pct,          # % of periods with positive payoff
    avg_premium_pct,       # average cost of put as % of S
    max_single_payoff,     # largest single payoff
    payoff_to_cost_ratio,  # total_payoff / total_cost
    crisis_capture_rate,   # % of drawdown events >15% where put paid out
    sharpe_analog,         # mean(pnl_pct) / std(pnl_pct)
    n_periods
    """
```

#### Optimization criteria (compute all, let user choose):

1. **Max total ROI** — pure return maximizer
2. **Max Sharpe analog** — risk-adjusted return
3. **Max crisis capture** — maximize hit rate during drawdown > 15%
4. **Min cost with positive ROI** — cheapest strategy that still profits
5. **Pareto frontier** — (win_rate, total_roi) tradeoff curve

### 4.5 `src/visualizer.py` — Plotly Chart Library

All charts must be:
- Interactive (Plotly)
- Dark-mode compatible
- Exportable to HTML and PNG

Required charts:

#### Chart 1: S&P 500 Price + Drawdown (dual axis)
- Left axis: S&P 500 price as line
- Right axis: drawdown % as filled area (red when below −10%)
- Annotations: vertical lines at major crisis dates (2011, 2015, 2018, 2020, 2022)
- VIX as secondary subplot below

#### Chart 2: Strike Optimization Heatmap
- X-axis: `strike_otm_pct` (5% → 30%)
- Y-axis: `T_days` (60, 90, 120)
- Color: `net_roi_pct` or `crisis_capture_rate` (toggle via dropdown)
- Hover: show all metrics

#### Chart 3: Cumulative P&L Comparison
- Multiple lines, one per selected strike (e.g. 10%, 15%, 20% OTM)
- X-axis: date
- Y-axis: cumulative net P&L in $
- Shaded bands for drawdown events > 10%
- Annotations at payoff spikes showing crisis event name

#### Chart 4: Payoff Distribution
- Histogram of per-period P&L for the selected strategy
- Log scale on X-axis to show fat tail
- Overlay: normal distribution fit (to highlight how fat the tail is)
- Annotate: mean loss (bleed), max gain (tail event)

#### Chart 5: Premium Cost Over Time
- Area chart: rolling 12-month total premium paid
- Overlay: rolling 12-month total payoff received
- The "bleeding" pattern should be clearly visible

#### Chart 6: VIX vs Option Price Scatter
- X: VIX level on purchase date
- Y: put price as % of S
- Color: `moneyness`
- Shows the relationship between vol regime and option cost

#### Chart 7: GDELT Tone vs Crisis Events (optional, if GDELT connected)
- Time series of `gdelt_avgtone` with rolling mean
- Overlay: S&P 500 drawdown
- Tests whether GDELT deterioration precedes market drawdowns

#### Chart 8: Pareto Frontier (Win Rate vs ROI)
- Scatter plot: X = win_rate_pct, Y = total_roi_pct
- Each point is a (moneyness, T_days) combination
- Efficient frontier highlighted
- Quadrant lines: median win rate, break-even ROI

---

## 5. Configuration File (`config.yaml`)

```yaml
data:
  sp500_path: "data/raw/sp500_daily.parquet"
  vix_path: "data/raw/vix_daily.parquet"
  rates_path: "data/raw/rates_daily.parquet"
  gdelt_path: "data/raw/gdelt_daily.parquet"
  options_path: "data/raw/options_chains/"
  start_date: "2010-01-01"
  end_date: "2024-12-31"

pricing:
  use_market_prices: false       # Set true if real options data is loaded
  iv_proxy_col: "vix"            # Column to use as sigma (vix or realized_vol_21d)
  iv_skew_adjustment: 1.1        # Multiply IV by this factor for OTM puts (skew approx)
  risk_free_rate_col: "rate_3m"  # Column for r; if null, use constant below
  risk_free_rate_const: 0.04     # Used if rate column not available

strategy:
  initial_capital: 100000
  put_budget_pct: 0.02           # Fraction of capital spent on puts per period
  rebalance_days: 21             # Trading days between purchases
  T_days: 90                     # Option expiry (calendar days)
  moneyness: 0.85                # Default single-run strike (K/S)

optimization:
  moneyness_grid: [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82,
                   0.84, 0.85, 0.86, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95]
  T_days_grid: [60, 90, 120]
  rebalance_days_grid: [21]
  objective: "net_roi_pct"       # Primary metric to maximize

output:
  results_path: "data/processed/simulation_results/"
  report_path: "report/output.html"
  charts_path: "report/charts/"
```

---

## 6. Jupyter Notebook Structure (`notebook.ipynb`)

The notebook is the primary deliverable. Structure it in the following cells/sections:

### Section 0 — Setup
```python
import yaml, pandas as pd, numpy as np, plotly.graph_objects as go
from src.data_loader import load_all
from src.features import build_features
from src.pricing import bs_put_price, bs_greeks
from src.simulator import simulate_barbell
from src.optimizer import optimize_strikes
from src.visualizer import *

config = yaml.safe_load(open("config.yaml"))
```

### Section 1 — Data Loading & Validation
- Load all datasets via `load_all(config)`
- Print shape, date range, missing value summary
- Show first/last 5 rows of master dataframe
- Assert: no gaps longer than 5 business days

### Section 2 — Feature Engineering
- Run `build_features(df)`
- Plot: drawdown timeseries with crisis annotations
- Plot: VIX regime distribution
- Print: table of major drawdown events (date, magnitude, duration, VIX at trough)

### Section 3 — Black-Scholes Primer (educational cell)
- Interactive widget (ipywidgets):  
  Sliders for S, K, T, r, sigma  
  Live output: put price, delta, vega, theta, breakeven move
- Show BS price surface as 3D plot over (K/S, T)

### Section 4 — Single Strategy Simulation
- Run `simulate_barbell()` with default config params
- Print: total cost, total payoff, net P&L, win rate
- Plot: Chart 3 (cumulative P&L) for this single strategy
- Plot: Chart 5 (premium bleed vs payoff)
- Plot: Chart 4 (P&L distribution)
- Show: top 10 best payoff events with date and context

### Section 5 — Strike Optimization
- Run `optimize_strikes()` over full grid
- Show: sorted results table (all metrics)
- Plot: Chart 2 (heatmap)
- Plot: Chart 8 (Pareto frontier)
- Print: recommended strike based on each objective

### Section 6 — Multi-Strike Comparison
- Run simulation for 5 selected strikes (from optimization output)
- Plot: Chart 3 overlaid for all 5 strategies
- Show: comparison table (cost, payoff, ROI, win rate, max single payoff)
- Highlight: which strategy would have maximized payout in each specific crisis

### Section 7 — Regime Analysis
- Split results by VIX regime: Low (<15), Normal (15-25), High (25-35), Extreme (>35)
- For each regime: average put price, win rate, average payoff
- Key insight table: "When VIX is high, puts are expensive but crises are more likely"
- Optional: GDELT tone analysis if data available (Chart 7)

### Section 8 — Sensitivity Analysis
- Vary `put_budget_pct`: [0.5%, 1%, 2%, 3%, 5%]  
  Show: how total capital allocation changes ROI and absolute P&L
- Vary `T_days`: [30, 60, 90, 120, 180]  
  Show: short-dated puts are cheaper but expire before recovery
- Vary `rebalance_days`: [5, 10, 21, 42]  
  Show: frequency effect on total cost and coverage

### Section 9 — Real Options Prices vs BS (conditional cell)
Only execute if `config.pricing.use_market_prices = true`:
- Load real options chains for 5 sample dates
- Compare: BS theoretical price vs market mid-price
- Compute: pricing error distribution
- Show: bid/ask spread as % of mid — this is the real transaction cost
- Key metric: `effective_cost = market_ask_price / bs_price` — the "overpay factor"

### Section 10 — Summary & Recommendations
- Auto-generate text summary from results
- Show: optimal strategy parameters
- Show: total cost over full period vs total insurance value received
- Show: what $100k portfolio would look like with/without puts during 2020 COVID crash
- Export: `report/output.html` with all charts embedded

---

## 7. Key Formulas Reference

### Black-Scholes Put (European)

```
d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T

Put = K·e^(−rT)·N(−d2) − S·N(−d1)
```

### Greeks

```
Delta   = N(d1) − 1              (negative for puts, range [−1, 0])
Gamma   = N'(d1) / (S·σ·√T)
Theta   = −[S·N'(d1)·σ / (2·√T)] + r·K·e^(−rT)·N(−d2)   (per year, divide by 365)
Vega    = S·N'(d1)·√T            (per 1 unit of vol; divide by 100 for per 1%)
```

Where `N()` = standard normal CDF, `N'()` = standard normal PDF

### Moneyness

```
OTM % = (1 − K/S) × 100
e.g. K = 0.85·S → 15% OTM
```

### Intrinsic Value at Expiry

```
Payoff = max(0, K − S_expiry) × num_contracts
Net P&L = Payoff − Premium_paid
```

### Annualized Cost of Protection

```
Annual_cost = (put_budget_pct × rebalances_per_year) × 100%
e.g. 2% × 12 = 24% per year — this is the "insurance premium" against tail risk
```

---

## 8. Dependencies

```
# requirements.txt
pandas>=2.0
numpy>=1.24
scipy>=1.11
plotly>=5.18
ipywidgets>=8.0
pyyaml>=6.0
pyarrow>=14.0      # parquet support
jupyter>=1.0
notebook>=7.0
nbformat>=5.9
```

---

## 9. Implementation Notes for the Agent

### Do NOT:
- Use `yfinance` for options data — it only provides current chains, not historical
- Assume VIX = realized volatility — VIX is implied vol, not historical; they diverge significantly during crises (exactly when it matters most)
- Use simple `DataFrame.iloc` indexing for date lookups — always use `.loc` with datetime index after setting it
- Hard-code dates — all date ranges must come from `config.yaml`

### DO:
- Cache all raw downloads to parquet on first run; subsequent runs load from cache
- Use `pd.business_day_calendar` for expiry date rolling
- Log every simulation run with parameters to `data/processed/run_log.jsonl`
- When BS sigma > 1.5 (150% vol), clamp to 1.5 and log a warning — numerical instability
- Add `iv_skew_adjustment` multiplier (config default: 1.1) to account for the fact that deep OTM puts trade at a vol premium vs ATM (the "vol skew" / "smirk") — without this, BS underprices real OTM puts

### IV Skew Adjustment Rationale
In practice, SPX implied volatility is not flat across strikes. Deep OTM puts trade at higher IV than ATM options — this is the "volatility skew" or "put skew." Without adjustment, Black-Scholes will systematically underprice the puts we are simulating.

A rough calibration (from academic literature):
- 5% OTM put: IV ≈ ATM IV × 1.05
- 10% OTM put: IV ≈ ATM IV × 1.10–1.15
- 15% OTM put: IV ≈ ATM IV × 1.15–1.25
- 20% OTM put: IV ≈ ATM IV × 1.20–1.35
- 30% OTM put: IV ≈ ATM IV × 1.40–1.60

The config parameter `iv_skew_adjustment` applies a flat multiplier. For more accurate results, implement a simple linear skew model: `sigma_adjusted = vix/100 × (1 + skew_slope × OTM_pct)` where `skew_slope ≈ 0.015` is a reasonable default for SPX.

---

## 10. Validation Checklist

Before declaring the pipeline complete, verify:

- [ ] S&P 500 data loads cleanly with no gaps > 5 business days
- [ ] VIX data aligns exactly with S&P 500 dates
- [ ] BS put price for ATM option (K=S, T=90/252, sigma=0.20, r=0.04) returns approximately `0.056 × S` (known analytical result)
- [ ] Put delta is negative and in range [−1, 0] for all inputs
- [ ] Simulation total cost = `n_periods × put_budget_pct × initial_capital`
- [ ] Win rate for 15% OTM puts over 2010–2024 is approximately 10–15%
- [ ] During March 2020: puts with K ≤ 0.85·S purchased in early February should show positive payoff
- [ ] Chart 3 shows sharp upward jumps at crisis dates (2011, 2015-16, 2018 Q4, 2020 Mar, 2022)
- [ ] Optimizer returns higher ROI for deeper OTM strikes (mathematical property: convexity)
- [ ] Exported HTML report opens correctly without local server

---

## 11. Optional Extensions

These are out of scope for v1.0 but recommended for v2.0:

1. **Dynamic allocation** — increase `put_budget_pct` when `vol_ratio > 1.3` (VIX spike signal)
2. **GDELT-driven regime switch** — if GDELT tone drops below −2σ, double the put allocation
3. **Straddle simulation** — add OTM calls for upside convexity (full barbell)
4. **Transaction cost model** — deduct bid/ask spread from payoff; estimate as `0.10 × put_price` for liquid strikes, `0.25 × put_price` for deep OTM
5. **Monte Carlo validation** — simulate 1000 synthetic S&P paths calibrated to historical vol; verify strategy statistics are robust
6. **Portfolio integration** — model 90% bonds + 10% puts allocation; compute full barbell portfolio metrics (total return, max drawdown, Sharpe)
7. **Real-time pricing hook** — connect to live options API for current chain visualization

---

*End of specification — v1.0*
