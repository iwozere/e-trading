# Technical Specification v2: Vectorbt + Optuna US Equity Momentum Engine

Revision notes: v2 restores the missing formulas, fixes execution and look-ahead issues, adds data-integrity, validation and statistical-rigor requirements, and replaces the checklist with measurable acceptance tests.

---

## 1. Goal
Build a vectorized backtesting and optimization framework for an intraday (1h/4h) momentum strategy on US equities/ETFs (long-only), with regime filtering, earnings avoidance, realistic costs, and statistically honest validation. Research tool only; no live execution.

## 2. Environment
- **Python:** 3.12 recommended. vectorbt 1.1.2 (Oct 2026) declares Python >=3.11,<3.15; numba typically lags the newest Python, so choose the newest version that numba's support table lists and that passes the smoke test below.
- **Packages:** `vectorbt` (1.x, actively released), `numba`, `numpy`, `pandas`, `optuna`, `pyarrow`, `pytest`, `pyyaml`, `pandas_market_calendars`. TA-Lib optional (build issues); ADX/ATR/Donchian implemented in numba or vbt-native.
- **Setup procedure:**
  1. `uv venv --python 3.12`; `uv pip install vectorbt optuna pandas pyarrow pytest pyyaml pandas_market_calendars`. Let the resolver choose compatible numpy/numba; do not hand-pin them first.
  2. `uv pip check`; print versions of python, numba, numpy, pandas, vectorbt, optuna.
  3. Run `tests/test_env_smoke.py`: a tiny `Portfolio.from_signals` run with fees, slippage, `price=open`, `cash_sharing`, `group_by`, plus a `stats()` call.
  4. Freeze: `uv pip compile` to a lockfile; commit it. Ship a Dockerfile built from the lockfile.
- **API drift (vectorbt 1.x is a major version):** do not assume 0.2x signatures. The execution, annualization and truncation tests in section 9 double as the compatibility suite and run in CI on every dependency change. Dependency upgrades happen only on a branch, never mid-research; if a future dependency breaks vectorbt, pin the last working lockfile.
- **Known breakage areas to check:** numpy 2 removed aliases (`np.float`, `np.bool`); pandas 2/3 deprecated uppercase freq strings (use `'h'`, `'min'`) and copy-on-write affects `.loc` assignment; numba first-call compile time (set `cache=True` and `NUMBA_CACHE_DIR`) and thread safety.
- **Optuna parallelism:** process-based (separate workers sharing a SQLite/Postgres storage), not threads, to avoid shared numba state and memory growth.
- All settings in `config.yaml` (universe, dates, costs, search space, seeds). No magic numbers in code.
- Global random seed; Optuna sampler seeded; results reproducible. Run manifest records all package versions.

## 3. Data Pipeline

### 3.1 Inputs (FMP)
1. Intraday OHLCV (1h; 4h built by resampling 1h, see 3.3).
2. SPY daily OHLCV.
3. Historical earnings dates per ticker (actual announced dates plus before-open / after-close flag).
4. Split and dividend history.
5. Point-in-time universe data (see 3.1b): historical index constituents and/or the full US common-stock list with delisted names.

### 3.1b Universe selection (frozen before any tuning)
Rule: the universe is defined by rules applied with data available at each date. Never use today's list of winners.

- **Mode A, point-in-time index (baseline):** historical S&P 500 membership as of each date (verify FMP historical-constituent endpoint coverage back to the start date).
- **Mode B, dynamic liquidity universe (recommended for breadth):** on the first trading day of each month, rank all US common stocks by trailing 60-day median dollar volume and keep the top N (default 200). The list is used for the following month and lagged by one day. A ticker leaving the list is closed at the next open unless config `hold_until_exit=true`.
- **Delisted names:** included. If intraday history for a delisted ticker is unavailable, log it and report the share of excluded names; the report must state residual survivorship bias.
- **Identifiers:** key all data on a stable ID (CIK or equivalent), not the ticker symbol, to handle renames and reused tickers; keep a symbol-history map.
- **Exclusions:** leveraged/inverse ETFs, ADRs (default), halted names, instruments with fewer than 252 trading days of history at the date of inclusion.
- **Filters:** price >= $5 and 20-day median dollar volume >= $20M, both trailing only.
- **ETF sleeve (optional):** plain ETFs tested as a separate run; earnings logic does not apply to them.
- **Concentration:** sector cap (default 30% of positions); optional per-sector ranking tie-break.
- **Governance:** the universe config is hashed and written to the run manifest; changing it invalidates prior optimization results. Universe rules may not be changed after viewing OOS or holdout performance.
- **Scale and memory:** 300 tickers x 20 years of 1h bars is about 10M rows per OHLCV array. Use float32, parquet cache, and chunked signal computation; the optimizer loads a prepared array once per worker.
- **Acceptance:** a universe test asserts that membership at date t uses only information available before t (truncation test applied to the universe builder), and that no ticker appears before its listing date or after its delisting date.

### 3.2 Data realities to verify before building (blocking)
- **History depth:** FMP provides ~20 years of 1h history (confirmed by owner). The loader must still log per-ticker first/last available timestamp and coverage % (missing sessions), and drop or flag tickers with <95% session coverage. The effective window per ticker is documented in the run manifest.
- **Adjustment:** `adjClose` alone is insufficient. Compute an adjustment factor (adjClose/close on the daily series) and apply it to O/H/L/C, or confirm intraday prices are split-adjusted. Volume must be inversely split-adjusted. Mixed adjusted/unadjusted OHLC corrupts breakout and ATR logic.
- **Survivorship bias:** universe must be point-in-time (include delisted names) or the limitation must be stated explicitly in every report.
- **Liquidity filter:** minimum 20-day median dollar volume (default $20M) and minimum price (default $5), evaluated using trailing data only.
- **Data quality checks:** duplicate timestamps, gaps within sessions, zero/negative prices, zero-volume bars, outlier returns (>25% in one bar flagged), timezone consistency. Fail loudly with a report; do not silently fill.

### 3.3 Session and timezone handling
- Convert all timestamps to `America/New_York`; keep regular trading hours only (09:30-16:00).
- 4h bars: define explicitly as session-anchored (e.g., 09:30-13:30 and 13:30-16:00 partial), not clock-aligned. Document the bar-label convention (label = bar open time) and ensure indicators use only bars that have closed.
- Handle half-days and holidays via an exchange calendar (`pandas_market_calendars`).

### 3.4 Daily-to-intraday alignment (anti-look-ahead)
Daily regime values computed from day T's close may only be used from day T+1's first bar.

```python
def align_daily_regime_to_intraday(daily: pd.Series, intraday_index: pd.DatetimeIndex) -> pd.Series:
    # daily index: tz-aware session dates; value known at close of that session
    s = daily.shift(1)                              # T uses T-1 close
    s.index = s.index.normalize()
    day_key = intraday_index.normalize()
    out = s.reindex(day_key, method="ffill")
    out.index = intraday_index
    return out.fillna(False).astype(bool)           # warm-up period => no trading
```

Mandatory test: truncating all data after time t must not change any signal at or before t (see 9).

## 4. Strategy

All signals are computed at bar close (bar t) and executed at the **open of bar t+1** (see 5).

### 4.1 Regime and quality filters (all must be TRUE to enter)
1. **SPY regime:** `SPY_close[T-1] > SMA(SPY_close, N_spy)[T-1]` (default N=200), aligned via 3.4.
2. **Trend strength:** `ADX(adx_period)[t] > adx_threshold` (default period 14, threshold 20). ADX period is a fixed config value (or optional search parameter).
3. **Relative volume (time-of-day normalized):** `Vol[t] > k_vol * mean(Vol at the same time-of-day over the prior 20 sessions)`. A plain 20-bar SMA is biased because intraday volume is U-shaped (open/close are always "spikes").
4. **Earnings handling (config `earnings_mode`, A/B tested):**
   Uses actual historical announcement timestamps (not estimates) with before-open / after-close flag.
   - `entry_block`: no new entry if earnings fall within the next M trading days (default 3). Open positions are held through the announcement (gap risk accepted).
   - `entry_block_force_exit`: as above, plus forced exit at the last bar before the announcement (after-close reports: exit before that session's close; pre-market reports: exit before the prior session's close). Minimizes gap risk but forfeits post-earnings drift.
   - `conditional_hold`: entry block as above; an open position is held through earnings only if unrealized profit >= `hold_cushion_atr` x ATR (default 2.0) at the pre-earnings decision bar, otherwise it is exited. Evaluated using data available at that bar only.
   - Baseline run default: `entry_block_force_exit`. The final choice is made from A/B results on train/WFO data only, never the holdout.
5. **Asset-level trend (optional, config flag):** `close > SMA(close, asset_sma_bars)`; spec previously mentioned asset-level filtering but never defined it.

### 4.2 Entry
- Raw: `close[t] > max(high[t-N .. t-1])` where N = `donchian_window`. The window **must exclude the current bar** (shift by 1), otherwise the breakout can never trigger as defined.
- Filtered: raw AND all filters in 4.1.

### 4.3 Exit (first to trigger)
1. `close[t] < SMA(close, exit_sma_window)[t]`.
2. ATR trailing stop: stop = highest close since entry minus `atr_mult * ATR(atr_period)`; evaluated on closes; exit executes next open (gap risk acknowledged).
3. Pre-earnings exit per `earnings_mode` (4.1.4).
4. Optional max holding period (`max_hold_bars`, default off).
5. Optional regime exit: SPY regime turns FALSE (config flag, default off; A/B test it).

### 4.4 Portfolio construction (previously unspecified)
- Long-only, no leverage, no shorting.
- Position sizing: equal weight, `size = equity * (1 / max_positions)`; `max_positions` default 10. Optional volatility-targeted sizing as an alternative (risk per trade = 0.5% of equity / (atr_mult * ATR)).
- Shared cash across assets (`cash_sharing=True`, `group_by=True`) so capital constraints are real. Per-asset independent runs are allowed only as diagnostics.
- Signal conflict rule when more candidates than free slots: rank by relative volume or ADX (config), deterministically.
- No pyramiding; one position per ticker.

## 5. Execution and Costs
- **Next-open execution:** pass `price=open` with entry/exit signals shifted by one bar. Do not rely on vbt's default (fills at the signal bar's close, which is look-ahead). Add a unit test asserting fill price equals next bar's open (plus slippage).
- Defaults: `fees=0.0005`, `slippage=0.0010` per side (about 0.30% round trip combined). Treat as conservative for liquid large caps; make them config values.
- **Cost sensitivity grid** in every final report: total cost per side at 0.5x, 1x, 2x, 3x. A strategy that fails at 2x is flagged fragile.
- `init_cash=100_000` (config), `freq` set correctly, and **annualization handled explicitly**: intraday bars per year is not `365*24`. Compute ratios from returns resampled to daily (preferred), or set `year_freq` and bars/day to the session grid (about 252 x 7 for 1h incl. partial bar, 252 x 2 for 4h). Report which was used.
- Report gross vs net return, total fees paid, total slippage cost, turnover and average holding time.

## 6. Optimization (Optuna)

### 6.1 Search space
| Parameter | Type | Range | Step |
|---|---|---|---|
| spy_sma_period | int | 100-200 | 20 |
| adx_threshold | float | 15-30 | 2.5 |
| donchian_window | int | 10-40 | 2 |
| exit_sma_window | int | 5-20 | 1 |
| vol_multiplier | float | 1.0-2.5 | 0.25 |
| atr_mult | float | 2.0-4.0 | 0.5 |
| earnings_buffer_days | int | 1-5 | 1 |
| hold_cushion_atr | float | 1.0-3.0 | 0.5 |

`hold_cushion_atr` is searched only when `earnings_mode=conditional_hold`. Fixed (not searched): ADX/ATR periods, fees, slippage, max_positions. Keep the space small: 7 parameters on noisy data overfits easily. `exit_sma_window` should be constrained below `donchian_window` (conditional sampling).

### 6.2 Objective
- Primary: Sortino or Sharpe (daily-return based, net of costs) on **train folds only**.
- Constraints, applied as smooth penalties rather than a constant (a flat penalty gives the sampler no gradient):
  - Minimum trades, scaled to universe size and applied per walk-forward train fold: `min_trades_fold` (default 30) and `min_trades_total` (default 200 across all train data). Smooth penalty: `score -= penalty * max(0, (min - n)/min)`. Rationale: the error of mean trade return scales ~1/sqrt(n); below ~100 trades Sharpe is mostly noise. Both values are config-tunable; flag (don't silently pass) any final result below 100 total trades.
  - Max drawdown > 25%: penalty proportional to the excess.
  - Exposure < 10% of time: reject.
- Alternative: multi-objective (Sharpe vs max drawdown) with Pareto front reporting.
- Use `TPESampler(seed=...)`, `MedianPruner` where folds allow early stopping, and a SQLite storage backend (`sqlite:///study.db`) so studies are resumable and memory stays bounded. Default 200-500 trials.

### 6.3 Validation scheme
- **Primary: Walk-Forward Optimization.** Rolling or anchored windows (e.g., 3y train / 6m test, step 6m). Re-optimize per window; stitch OOS segments into one OOS equity curve. Report per-window parameters and OOS metrics.
- **Purge/embargo:** gap between train and test of at least max indicator lookback plus max holding period, to prevent leakage via overlapping trades.
- **Holdout:** the final 20-30% of the timeline is touched **once**, after all design decisions are frozen. Never feed it back into tuning. (The previous 70/30 single split is acceptable only as a holdout, not as the sole validation.)
- **Robustness checks:**
  - Parameter-plateau analysis: neighbors of the best parameters should perform similarly (report sensitivity heatmaps for 2 key pairs).
  - Deflated Sharpe Ratio or a White's Reality Check / SPA test accounting for number of trials.
  - Bootstrap (block) confidence intervals on OOS Sharpe and on A/B difference.
  - Regime breakdown: performance by year and by SPY bull/bear, high/low VIX.

## 7. A/B Test and Benchmarks
Compare, with identical costs and sizing:
1. Unfiltered momentum (Donchian + exit only).
2. Regime-filtered (full 4.1).
3. Ablations: each filter removed individually (shows which filters earn their place).
3b. Earnings-mode comparison: `entry_block` vs `entry_block_force_exit` vs `conditional_hold`, reporting return, Sharpe, max drawdown, worst single-trade loss, and share of P&L from earnings-gap trades.
4. Benchmarks: SPY buy-and-hold, equal-weight universe buy-and-hold, and 200-day-SMA SPY timing.

Metrics: CAGR, annualized vol, Sharpe, Sortino, Calmar, max drawdown and duration, win rate, profit factor, expectancy per trade, average win/loss, trade count, exposure %, turnover, avg holding bars, gross vs net, cost drag, and alpha/beta vs SPY. Include statistical significance of the A/B difference (paired bootstrap).

## 8. Deliverables
```
config.yaml
requirements.in + lockfile (uv), Dockerfile
universe.py       # point-in-time / liquidity-ranked universe builder
data_loader.py    # FMP fetch/cache (parquet), adjustment, QC, calendar, alignment
indicators.py     # ADX, ATR, Donchian, time-of-day relative volume, SMA
strategy.py       # masks -> entries/exits -> vbt.Portfolio (next-open execution)
optimizer.py      # Optuna study, WFO loop, penalties, SQLite storage
validation.py     # robustness: DSR, bootstrap, sensitivity, cost grid
main.py           # CLI: backtest | ab-test | optimize | wfo | report
tests/            # pytest suite (section 9)
reports/          # metrics tables (CSV), charts (PNG/HTML), run manifest
```
Each run writes a manifest: git hash, config snapshot, data date range, package versions, seed.

## 9. Acceptance Criteria (testable)
- [ ] **No look-ahead (truncation test):** for random cut points t, signals computed on data truncated at t equal the signals from the full run for all times <= t.
- [ ] Daily regime test: a synthetic SPY crossing above SMA on day T produces TRUE only from day T+1's first bar.
- [ ] Execution test: every entry/exit fills at next bar's open adjusted by slippage; fee and slippage totals are non-zero and appear in the stats output.
- [ ] Breakout test: Donchian uses prior N bars only (synthetic series with known breakout bar).
- [ ] Earnings test: for each `earnings_mode`, no entries in the M-day pre-earnings window; under `entry_block_force_exit` no position is held across a known announcement; under `conditional_hold` only positions meeting the ATR cushion are held, using only data available at the decision bar.
- [ ] Annualization test: Sharpe on a synthetic constant-return series matches the analytical value for both 1h and 4h.
- [ ] Optimizer runs 200 trials on a 50+ ticker universe with stable memory (RSS growth <10% after warm-up) and is resumable from SQLite.
- [ ] WFO produces a stitched OOS curve with no overlap between any train and test window (asserted in code).
- [ ] A/B report, ablations, benchmarks and cost-sensitivity grid generated by a single `main.py` command from a clean checkout; same seed yields identical results.
- [ ] Universe test: membership at date t depends only on data before t; no ticker is traded before listing or after delisting.
- [ ] Environment smoke test passes on a clean machine from the lockfile (`uv pip check` clean, tiny `from_signals` run succeeds).
- [ ] Report states data limitations (history depth, survivorship, adjustment method) explicitly.

## 10. Out of Scope / Known Limitations
Live trading, short selling, options, crypto, intraday order-book simulation, tax effects. Fills assume liquid names; results for small caps are not valid without a spread/impact model. Backtested results do not guarantee future performance.

## 11. Suggested Build Order
0. Environment setup, lockfile and smoke test.
1. Data loader plus universe builder plus QC plus alignment plus truncation tests.
2. Indicators and signals; strategy with next-open execution.
3. Baseline backtest and A/B with fixed default parameters (sanity check before any optimization).
4. Optuna on a single train window, then WFO.
5. Robustness suite, holdout evaluation (once), final report.
