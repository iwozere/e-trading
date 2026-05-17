# P16 Taleb Barbell Pipeline — Implementation Plan

**Version:** 1.0  
**Date:** 2026-05-17  
**Status:** Planning

---

## 1. Data Sources — What P15 Already Provides

P16 reads from the shared `DATA_CACHE_DIR` populated by the P15 daily pipeline.
No new scheduled downloaders are needed for the core pipeline.

| P16 Requirement | P15 Source | Cache Path |
|---|---|---|
| S&P 500 OHLCV (`SPY`) | yfinance job | `ohlcv/SPY/1d/YYYY.csv.gz` |
| VIX (`^VIX`) | yfinance job | `ohlcv/^VIX/1d/YYYY.csv.gz` |
| Risk-free rate | FRED `DGS3MO` | `fred/DGS3MO.csv.gz` |
| GDELT macro sentiment | Events job | `gdelt/events/YYYYMMDD.events.csv.gz` |

Historical options chains (2010–2024) are **not available** from P15 — P15 captures
only forward-looking daily snapshots via yfinance, which does not expose historical
chains. The pipeline defaults to Black-Scholes synthetic pricing (`USE_MARKET_PRICES = false`),
which is the correct and documented fallback from the spec.

---

## 2. Key Decision Points

### 2.1 Risk-Free Rate: DFF vs DTB3

**What they are:**

- **DFF — Federal Funds Rate (Effective)**
  The overnight interbank lending rate. Banks lend their reserve balances to each other
  at this rate. It is a *policy rate*: the Fed sets a target range and moves it in
  discrete steps (typically 25 bp). DFF changes rarely and is already tracked by P15.

- **DTB3 — 3-Month Treasury Bill Secondary Market Rate**
  The yield on 3-month US Treasury bills traded in the secondary (open) market. This is
  what you would actually earn by lending money risk-free for 90 days. It fluctuates
  daily with supply and demand at Treasury auctions and in the secondary market.

**Why DTB3 is the theoretically correct input for Black-Scholes:**
Black-Scholes requires `r` = the continuously compounded risk-free rate for the same
horizon as the option's time-to-expiry. For a 90-day put that horizon is exactly the
3-month T-bill horizon, so DTB3 is the textbook-correct choice.

**Why DFF is an acceptable substitute here:**

1. *Spread is small in normal conditions.* DFF and DTB3 typically differ by 5–30 bp.
   During crises (2008, 2020) the gap can widen to 50–100 bp as flight-to-safety demand
   drives T-bill yields below the fed funds rate.

2. *Option price is almost insensitive to r for deep OTM puts.* The rho (∂P/∂r) of a
   European put is negative and small in magnitude for OTM options. For a 15% OTM put
   with T = 90/365, S = 4000, r = 4%, σ = 0.20: rho ≈ −0.15 per 1% change in r.
   A 50 bp DFF–DTB3 divergence moves the put price by < 0.08, which on a $4000 underlying
   is 0.002% — far below bid/ask noise.

3. *DFF is already in P15.* DTB3 and DGS3MO both required adding new FRED series to
   the P15 downloader — which has now been done (see §FRED expansion below).

**Decision:** Use `DGS3MO` (3-Month Treasury CMT, bond-equivalent yield) from
`fred/DGS3MO.csv.gz`. DGS3MO is preferred over DTB3 because it is already expressed
on a bond-equivalent (semi-annual) yield basis — the same convention used in standard
Black-Scholes implementations — whereas DTB3 is on a bank-discount basis requiring a
conversion step. Both DTB3 and DGS3MO have been added to P15's `FRED_SERIES` dict
and will be backfilled automatically. `config.yaml` sets `rate_source: "DGS3MO"` and
the `rate_source` field allows switching to any other series without code changes.

---

### 2.2 GDELT Aggregation — Daily Scalar Cache

**Source files:** `DATA_CACHE_DIR/gdelt/events/YYYYMMDD.events.csv.gz`

Each file contains one row per `(EventCode, EventRootCode, QuadClass)` for that day,
with columns: `num_events`, `num_mentions`, `num_articles`, `avg_tone`,
`goldstein_scale_avg` — already aggregated by the P15 GdeltDownloader.

**Aggregation logic for P16:**
To collapse from per-EventCode rows to a single daily scalar, weight each event type
by its `num_articles` (media volume) — this gives larger events proportionally more
influence on the daily tone signal.

```
daily_avgtone      = Σ(avg_tone × num_articles) / Σ(num_articles)
daily_goldstein    = Σ(goldstein_scale_avg × num_articles) / Σ(num_articles)
daily_num_articles = Σ(num_articles)
daily_num_events   = Σ(num_events)
```

**Output cache:** `DATA_CACHE_DIR/gdelt/gdelt_p16_daily.csv.gz`

Stored as a plain CSV (gzipped) so it is human-readable with any spreadsheet tool.
Written by `data_loader.load_gdelt()` on first run; subsequent runs skip the rebuild
if the file already exists (or extend incrementally if new Events files have appeared).

Schema:

| Column | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Calendar date (index) |
| `avgtone` | float | Article-weighted mean tone across all events (−100 to +100) |
| `goldstein_scale` | float | Article-weighted mean Goldstein stability score (−10 to +10) |
| `num_articles` | int | Total media article count for the day |
| `num_events` | int | Total GDELT event records for the day |

**Coverage:** GDELT 2.0 Events start 2015-02-18. For the backtest period 2010–2015 the
GDELT signal will be NaN; `features.py` will forward-fill or zero-fill during that window
and mark rows with `gdelt_available = False`. Charts 3 and 7 annotate the GDELT start date.

**Why Events (not GKG) for this signal:**
GKG files are aggregated per `(date, theme)`. To get a daily scalar from GKG we would
need a second groupby across all themes — adding complexity. Events files already capture
the GoldsteinScale (event stability, −10 to +10) which is the primary macro stress signal
the spec requires. GKG tone is better suited for theme-filtered analysis (e.g. "financial
sector tone only"), which is out of scope for P16 v1.0.

---

### 2.3 Volatility Skew Model — Linear per-Strike

Black-Scholes with flat ATM volatility (VIX/100) systematically underprices deep OTM
puts because the real SPX implied vol surface exhibits a "put skew" or "vol smirk":
deep OTM puts trade at a premium to ATM implied vol.

**Rejected: flat multiplier (`iv_skew_adjustment = 1.1`)**
A flat 10% uplift applies the same correction at 5% OTM and 30% OTM. In practice the
skew is approximately linear in OTM percentage, so the flat multiplier over-corrects near
ATM and under-corrects at extreme strikes.

**Adopted: linear skew model**

```
otm_pct   = (1 - K/S)                       # 0.15 for a 15% OTM put
sigma_adj = (VIX / 100) × (1 + skew_slope × otm_pct × 100)
```

Default `skew_slope = 0.015` (config key `pricing.skew_slope`), meaning:
- 5% OTM  → sigma × 1.075  (+7.5%)
- 10% OTM → sigma × 1.15   (+15%)
- 15% OTM → sigma × 1.225  (+22.5%)
- 20% OTM → sigma × 1.30   (+30%)
- 30% OTM → sigma × 1.45   (+45%)

These values are consistent with historical SPX vol skew calibrations from academic
literature (Carr & Wu 2003, Dennis & Mayhew 2002) and align with the ranges in the
spec's §9 IV Skew Adjustment Rationale.

The flat multiplier config key (`iv_skew_adjustment`) is removed from `config.yaml`.
Replaced by `skew_slope: 0.015`. The function signature in `pricing.py`:

```python
def skew_adjusted_sigma(
    vix_level: float,
    otm_pct: float,       # e.g. 0.15 for 15% OTM — pass as decimal, not percent
    skew_slope: float = 0.015,
) -> float:
    """Return skew-adjusted implied vol for a given OTM put."""
    return (vix_level / 100.0) * (1.0 + skew_slope * otm_pct * 100.0)
```

---

## 3. Directory Structure

```
src/ml/pipeline/p16_taleb/          ← source code only; no data files
├── docs/
│   ├── pipeline-specification.md   ← original spec (do not modify)
│   └── implementation-plan.md      ← this document
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── pricing.py
│   ├── simulator.py
│   ├── optimizer.py
│   └── visualizer.py
├── tests/
│   ├── test_pricing.py
│   ├── test_features.py
│   ├── test_simulator.py
│   └── test_optimizer.py
├── config.yaml
├── notebook.ipynb
└── README.md

results/p16_taleb/                  ← all pipeline outputs (gitignored)
├── master_daily.csv.gz             ← merged SP500 + VIX + rates + GDELT
├── simulation_results/             ← per-run optimizer output csv.gz files
├── charts/                         ← Plotly HTML chart files
└── report/
    └── output.html                 ← full HTML report
```

---

## 4. Module Specifications

### 4.1 `src/data_loader.py`

Reads directly from `DATA_CACHE_DIR` (P15 cache). All input data stays in its original
`csv.gz` format — no intermediate copies are written inside the pipeline source tree.
All outputs go to `results/p16_taleb/` under the project root.

Data source paths are constructed directly from `sp500_ticker`, `vix_ticker`, and
`rate_source` — no redundant path keys in config. Only `gdelt_cache` (DATA_CACHE_DIR
relative) and `master_path` (project root relative) are stored in config.

```python
def load_sp500(cache_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Read SPY OHLCV from per-year ohlcv/SPY/1d/YYYY.csv.gz files."""

def load_vix(cache_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Read ^VIX close from ohlcv/^VIX/1d/YYYY.csv.gz; rename close → vix."""

def load_rates(cache_dir: Path) -> pd.DataFrame:
    """Read fred/DGS3MO.csv.gz; rename to rate_3m; convert percent → decimal; forward-fill weekends."""

def load_gdelt(cache_dir: Path, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Aggregate GDELT 2.0 Events to daily scalars.

    Reads DATA_CACHE_DIR/gdelt/events/YYYYMMDD.events.csv.gz files.
    On first run: streams all files, computes article-weighted avgtone
    and goldstein_scale, writes DATA_CACHE_DIR/gdelt/gdelt_p16_daily.csv.gz.
    On subsequent runs: reads the cached file and extends with any new days.
    Returns DataFrame with columns: date, avgtone, goldstein_scale,
    num_articles, num_events.
    """

def load_all(config: dict, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and merge all sources into master_daily.csv.gz.

    Merge strategy: left join on SP500 trading days (excludes weekends/holidays).
    Validates: no gaps > 5 consecutive business days.
    Writes: results/p16_taleb/master_daily.csv.gz (project root, human-readable).
    """
```

### 4.2 `src/pricing.py`

```python
def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price. T in years."""

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Returns dict: delta, gamma, theta (per day), vega (per 1% vol), rho."""

def skew_adjusted_sigma(
    vix_level: float,
    otm_pct: float,
    skew_slope: float = 0.015,
) -> float:
    """
    Linear skew model for OTM put implied vol.

    otm_pct: decimal (0.15 = 15% OTM).
    Returns: adjusted sigma as decimal (e.g. 0.25 for 25% vol).
    """
```

Numerical guards (from spec §9):
- Clamp `sigma > 1.5` → log warning, use 1.5
- Skip period if `put_price <= 0` or `put_price > S × 0.15`

### 4.3 `src/features.py`

Pure function `build_features(df: pd.DataFrame) -> pd.DataFrame`.
Adds columns to the master DataFrame:

| Column | Formula |
|---|---|
| `drawdown` | `(close - close.cummax()) / close.cummax()` |
| `ret_1d` / `ret_5d` / `ret_21d` / `ret_63d` | `close.pct_change(n)` |
| `vix_ma20` | `vix.rolling(20).mean()` |
| `vix_regime` | `pd.cut(vix, [0, 15, 20, 30, 40, 100], labels=["low","normal","elevated","high","extreme"])` |
| `vol_ratio` | `vix / vix_ma20` |
| `gdelt_tone_ma5` | `avgtone.rolling(5).mean()` |
| `stress_flag` | `(vix > 30) \| (drawdown < -0.10)` |
| `gdelt_available` | `avgtone.notna()` |

### 4.4 `src/simulator.py`

Core loop iterates over rebalance dates from the master DataFrame.
On each date:
1. Compute `K = moneyness × S`
2. Compute `otm_pct = 1 - moneyness`
3. Get `sigma = skew_adjusted_sigma(vix, otm_pct, skew_slope)`
4. Get `T = T_days / 365.0`
5. Price put: `put_price = bs_put_price(S, K, T, r, sigma)`
6. Guard checks (skip if invalid)
7. Roll expiry to next business day (`pd.offsets.BDay`)
8. At expiry: `payoff = max(0, K - S_expiry) × num_contracts`

Output: one row per cycle with all columns specified in spec §4.3.

### 4.5 `src/optimizer.py`

Grid sweep over `moneyness_grid × T_days_grid × rebalance_days_grid`.
Uses `concurrent.futures.ProcessPoolExecutor` for parallelism.
Computes 8 summary statistics per combination (spec §4.4).
Returns tidy DataFrame sorted by `net_roi_pct` descending.

### 4.6 `src/visualizer.py`

Eight Plotly charts (all dark-mode compatible, HTML/PNG exportable):

| # | Function | Key Design |
|---|---|---|
| 1 | `chart_sp500_drawdown(df)` | Dual-axis; drawdown area red below −10%; VIX subplot; crisis verticals |
| 2 | `chart_heatmap(opt_df)` | X = OTM%, Y = T_days, colour = metric; dropdown toggle (roi / crisis_capture) |
| 3 | `chart_cumulative_pnl(results)` | Multi-line per strike; shaded recession bands; spike labels |
| 4 | `chart_payoff_distribution(sim_df)` | Histogram log-X; normal overlay; bleed/tail annotations |
| 5 | `chart_premium_bleed(sim_df)` | Rolling 12M cost vs payoff area; gap = net drag |
| 6 | `chart_vix_vs_cost(sim_df)` | Scatter VIX × put_price%, colour = moneyness |
| 7 | `chart_gdelt_tone(df)` | GDELT 5-day MA vs drawdown dual-axis; annotate GDELT start date |
| 8 | `chart_pareto(opt_df)` | Win-rate × ROI scatter; efficient frontier in gold; median crosshairs |

---

## 5. Phased Build Order

### Phase 1 — Data & Pricing Foundation
1. `src/data_loader.py` + `load_all()` → validate `master_daily.csv.gz` for 2010–2024
2. `src/pricing.py` → unit tests against known BS analytical values
3. `src/features.py` → unit tests for drawdown, regime labels

### Phase 2 — Simulation Core
4. `src/simulator.py` → single-run test: 15% OTM, 90-day puts, 2010–2024
5. Validate: March 2020 puts purchased in early February show positive payoff
6. Notebook Sections 1–4 (data loading → single strategy)

### Phase 3 — Optimization & Charts
7. `src/optimizer.py` with `ProcessPoolExecutor` (~50 combinations, < 60 s)
8. `src/visualizer.py` — all 8 charts
9. Notebook Sections 5–10

### Phase 4 — Export & Documentation
10. HTML report export (`report/output.html`)
11. `README.md`, module `docs/` files, unit tests

---

## 6. `config.yaml` — Final Version

```yaml
data:
  sp500_ticker: "SPY"
  vix_ticker: "^VIX"
  rate_source: "DGS3MO"       # 3M Treasury CMT (bond-equivalent yield); best for BS pricing
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  gdelt_cache: "gdelt/gdelt_p16_daily.csv.gz"   # relative to DATA_CACHE_DIR
  master_path: "results/p16_taleb/master_daily.csv.gz"  # relative to project root

pricing:
  use_market_prices: false
  skew_slope: 0.015            # linear skew: sigma × (1 + slope × otm_pct × 100)
  risk_free_rate_const: 0.04   # fallback if FRED data unavailable

strategy:
  initial_capital: 100000
  put_budget_pct: 0.02
  rebalance_days: 21
  T_days: 90
  moneyness: 0.85

optimization:
  moneyness_grid: [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82,
                   0.84, 0.85, 0.86, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95]
  T_days_grid: [60, 90, 120]
  rebalance_days_grid: [21]
  objective: "net_roi_pct"

output:
  # All output paths relative to project root
  results_path: "results/p16_taleb/simulation_results/"
  report_path: "results/p16_taleb/report/output.html"
  charts_path: "results/p16_taleb/charts/"
```

---

## 7. Validation Checklist

- [ ] `master_daily.csv.gz` covers 2010-01-04 to 2024-12-31 with no gap > 5 business days
- [ ] BS put for ATM option (K=S, T=90/252, σ=0.20, r=0.04) ≈ 0.04 × S (trading-day convention)
- [ ] Put delta is negative and in [−1, 0] for all inputs
- [ ] `skew_adjusted_sigma(20.0, 0.15)` returns `0.20 × 1.225 = 0.245`
- [ ] March 2020 payoff positive for puts with K ≤ 0.85·S purchased in early February
- [ ] GDELT cache `DATA_CACHE_DIR/gdelt/gdelt_p16_daily.csv.gz` is readable as CSV
- [ ] Chart 3 shows upward jumps at 2011, 2015-16, 2018 Q4, 2020 Mar, 2022
- [ ] Optimizer returns higher ROI for deeper OTM strikes (convexity property)
- [ ] Exported HTML opens without a local server

---

*End of implementation plan — v1.0*
