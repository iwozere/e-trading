# P10 EMPS3 — Signal Analysis: Why the Pipeline Produces No Signals

_Analysis date: 2026-05-20 | Based on 37 runs (2026-03-13 → 2026-05-19) — 12,821 ticker-evaluations_

---

## 1. What the Pipeline Is Trying to Do

P10 EMPS3 detects the **"Coiled Spring"** pattern: a stock where institutional buyers are quietly absorbing the available float before a breakout. The signature is:

- **Volume spike** — elevated trading activity (z-score ≥ 1.5 above 20-day average)
- **Price stays quiet** — the volume does NOT translate into price movement (tight daily range, low ATR)
- **Near resistance** — the stock is pressing against its 52-week high, hinting that supply overhead is running thin

The pipeline then produces a rolling alert (Phase 1.5) when the same stock meets these criteria across three or more consecutive days, indicating sustained institutional absorption.

The idea is sound in theory. A stock can trade millions of shares in a narrow range if large buyers are absorbing every seller without letting price escape — this is classic Wyckoff accumulation logic. The problem is that the implementation makes it nearly impossible for any stock to qualify.

---

## 2. What Actually Happened Across 37 Runs

### 2.1 Funnel Summary (aggregated)

| Stage | Count | % of Evaluated |
|---|---|---|
| Total ticker-evaluations | 12,821 | 100% |
| ERRORS (code bugs) | 1,973 | 15.4% |
| FAILED: low vol z-score | 9,475 | 73.9% |
| FAILED: poor price compression | 1,068 | 8.3% |
| FAILED: insufficient bars | 230 | 1.8% |
| FAILED: low absorption ratio | 58 | 0.5% |
| FAILED: too far from 52w high | 12 | 0.09% |
| PASSED | 4 | 0.03% |

### 2.2 The 4 "Passed" Records Are False Positives

All 4 passes are for **XRXDW** (a Xerox warrant trading at ~$0.11) on consecutive days in April 2026.
Every single metric for this ticker is `NaN` or `0.0` — no real data. The ticker passed because:

```python
if vol_zscore <= self.config.min_vol_zscore:   # NaN <= 1.5  → False → does NOT filter out!
    return False, metrics, 'low_volume_zscore'
```

In Python, any comparison with `NaN` returns `False`. So `NaN <= 1.5` is `False`, meaning the guard condition never fires and the stock slips through all three filters. **The pipeline has never produced a legitimate signal.**

### 2.3 Filter-by-Filter Funnel (runs with valid data)

| Filter | Passes | % of prior stage |
|---|---|---|
| vol_zscore ≥ 1.5 | 961 / 12,821 | 7.5% |
| price_range_1d < 3% | 72 / 961 | 7.5% |
| atr_ratio < 2% | 23 / 72 | 31.9% |
| dist_52w_high ≤ 5% | 5 / 23 | 21.7% |
| **Final pass rate** | **5 / 12,821** | **0.04%** |

With 350 tickers in the fundamental-filtered universe, the expected signal count per run is `350 × 0.0004 ≈ 0.14`. Nearly zero on every day.

---

## 3. Root Cause Analysis

### 3.1 Bug: NaN Values Pass All Filter Guards

**Severity: Critical (data-corruption, not just missed signals)**

All three filter checks use `<=` or `>=` comparisons, which return `False` for NaN inputs, silently turning every filter into a pass:

```python
# All three are NaN-unsafe:
if vol_zscore <= self.config.min_vol_zscore:  # NaN <= 1.5 → False (passes!)
if price_range_1d >= self.config.max_price_impact or atr_ratio >= 0.02:  # NaN >= X → False
if ar <= self.config.min_vol_rv_ratio:        # NaN <= 1.5 → False (passes!)
```

Any ticker with missing intraday data (warrants, halted stocks, thin tickers) will always pass. The fix is a NaN pre-check at the top of `_check_accumulation`.

### 3.2 Bug: `_coerce_ohlcv_timestamp_column` AttributeError in Early Runs

**Severity: High (caused 1,898 incorrect ERROR outcomes)**

In early pipeline versions the method `_coerce_ohlcv_timestamp_column` was not defined as an instance method, so every ticker raised `AttributeError`. This is fixed in the current code, but it contaminated the historical result files and means the pipeline was effectively blind for its first several weeks.

### 3.3 Design Flaw: Volume Spike + Tight Range Are Mutually Contradictory at Daily Granularity

**Severity: High (fundamental)**

The hypothesis requires elevated volume WITHOUT proportional price movement. At the **daily bar** level this almost never happens:

- Out of 961 tickers with vol_zscore ≥ 1.5 across 37 runs, only 72 (7.5%) had a daily price range below 3%.
- The median price range for high-volume tickers was **8.9%** — three times the threshold.

High volume at the daily level typically causes price to move. The absorption effect the strategy is modelling is real, but it manifests **intraday** (the stock opens, trades heavily in a narrow band, closes near the open) while the daily OHLC bar looks "quiet" only if the buyer keeps price anchored all day. This is rare.

**Fix**: Compute price compression from intraday (1h) bars instead of, or in addition to, the daily OHLC bar. A stock can trade heavy intraday volume in a 0.5% band while the daily H-L looks like 2%.

### 3.4 Threshold: ATR Ratio < 2% Is Extremely Restrictive

**Severity: High**

After vol_zscore and daily range filters, 72 tickers remain. Only 23 (32%) have ATR(14) below 2% of price. The distribution among those 72 stocks:

| ATR/price threshold | Count passing |
|---|---|
| < 2% | 23 (32%) |
| < 3% | 49 (68%) |
| < 4% | 64 (89%) |
| < 5% | 69 (96%) |

Typical US small/mid-cap ATR/price is **2–5%**. A threshold of 2% filters out almost everything. Raising to 3.5–4% would triple the number of candidates at this stage at negligible quality cost.

### 3.5 Threshold: 52-Week High Proximity ≤ 5% Is Too Strict

**Severity: High**

After the first three filters, 23 tickers remain. Only 5 (22%) are within 5% of their 52-week high. Distribution:

| Distance threshold | Count passing |
|---|---|
| ≤ 5% | 5 (22%) |
| ≤ 10% | 7 (30%) |
| ≤ 20% | 13 (57%) |

The 52-week high filter serves to ensure the stock is in supply-depletion territory. But it conflates two different things:
- The **absolute 52-week high** is dominated by the market's peak from 12 months ago, which may be irrelevant today.
- What the strategy actually wants is a stock pressing against **its most recent local resistance** — the 20-day or 50-day high.

The code already computes `high_20` (20-day high) for scoring, but never uses it as a filter gate. This is the correct resistance level to use.

### 3.6 Redundant Double-Filter: Daily Range AND ATR

**Severity: Medium**

Both `price_range_1d` (single-day range as % of price) and `atr_ratio` (14-day average true range) measure the same phenomenon — price volatility. Using both as hard gates is redundant and reduces the pass rate more than either alone justifies. ATR is a smoother signal; the daily range is noisier (one quiet day doesn't make a setup). Keeping ATR (relaxed to ~4%) and removing or relaxing the daily range check would make the filter set more statistically robust.

### 3.7 Absorption Ratio Formula Produces Negative Values

**Severity: Medium**

```python
ar = vol_zscore / rv if rv > 0 else 0.0
```

When `vol_zscore < 0` (volume is below its 20-day average), `ar` is negative. The check `ar <= 1.5` then fires, correctly rejecting the ticker — but only because the vol_zscore check fires first. If the order of checks ever changes, negative AR values could slip through. The formula should semantically require `vol_zscore > 0` before computing AR.

### 3.8 Phase 1.5 Rolling Memory Is Starved of Inputs

**Severity: Medium (downstream consequence)**

The rolling memory scanner looks for tickers that appeared in the watchlist 3+ times over 5 days. Since Stage 3 (AccumulationAnalyzer) never produces any output, the rolling memory stage has never had any inputs to process. It is functionally dead as a result of the Stage 3 failures above.

---

## 4. Proposed Changes

### Tier 1 — Bug Fixes (implement first, no logic risk)

#### Fix 1.1: NaN-safe filter guards in `_check_accumulation`

Add at the top of `_check_accumulation` before any filter logic:

```python
if np.isnan(vol_zscore) or np.isnan(rv) or np.isnan(ar):
    return False, metrics, 'nan_metrics'
```

Also guard `price_range_1d` and `atr_ratio` since they can be NaN when the daily bar data is incomplete.

#### Fix 1.2: Require `vol_zscore > 0` before computing AR

```python
# Replace:
ar = vol_zscore / rv if rv > 0 else 0.0

# With:
ar = vol_zscore / rv if (rv > 0 and vol_zscore > 0) else 0.0
```

A negative vol_zscore means volume is below average — there is no accumulation happening regardless of price range.

---

### Tier 2 — Threshold Recalibration (high priority, data-driven)

Based on the filter funnel analysis across 37 runs, the following adjustments are expected to produce 5–30 legitimate candidates per run without materially increasing false positives:

| Parameter | Current | Proposed | Rationale |
|---|---|---|---|
| `atr_ratio` hard cutoff | 2% | 4% | Typical small-cap ATR is 2–5%; 2% excludes 68% of valid setups |
| `max_distance_from_resistance` | 5% (52w high) | 15% (20-day high) | Local resistance is more relevant; 5% of 52w high fires almost never |
| `max_price_impact` (daily range) | 3% | 5% or remove | Redundant with ATR; relaxing increases funnel yield dramatically |

Projected funnel after recalibration (estimated from historical data):
- vol_zscore ≥ 1.5: ~961 (unchanged)
- price_range < 5%: ~175 (vs 72 today)
- atr_ratio < 4%: ~155 (vs 23 today)
- dist_20d_high ≤ 15%: ~60–90 (vs 5 today)
- Expected signals/run: **3–8 candidates**

---

### Tier 3 — Logic Improvements (medium priority)

#### Improvement 3.1: Use intraday range for price compression

The "coiled spring" manifests intraday. Replace or supplement the daily OHLC range check with the standard deviation of **intraday (1h) bar ranges** over the past 3 days:

```python
intraday_ranges = df_intra['high'].values[-20:] - df_intra['low'].values[-20:]
recent_rv_intraday = np.std(intraday_ranges) / last_price
```

A stock with heavy 1h volume in a 0.3% intraday band is a stronger signal than a stock with a narrow daily bar that might simply not have traded.

#### Improvement 3.2: Replace 52-week high gate with "near local high" gate

Replace `dist_52w_high <= max_distance_from_resistance` with:

```python
# Use 20-day high instead of 52w high
dist_20d_high = (high_20 - daily_curr['close']) / high_20
if dist_20d_high > config.max_distance_from_local_high:  # e.g. 0.08
    return False, metrics, 'too_far_from_local_high'
```

The 20-day high represents the most recent supply zone that sellers have defended. A stock pressing this level with increasing volume is exactly what the strategy looks for.

#### Improvement 3.3: Add a "tightening trend" confirmation

Rather than a single-bar compression check, validate that the stock has been compressing over multiple days:

```python
# Check that ranges have been shrinking over the last 5 daily bars
recent_ranges = (df_daily['high'].values[-5:] - df_daily['low'].values[-5:]) / df_daily['close'].values[-5:]
slope = np.polyfit(range(5), recent_ranges, 1)[0]
range_tightening = slope < -0.0005  # ranges trending narrower
```

This is the same logic the rolling memory Phase 1.5 already uses for ATR slope — bring it into the single-day evaluation as a soft boost to the score rather than a hard gate.

#### Improvement 3.4: Use BB width trend, not just current BB width

The current BB squeeze check is `bb_current_width <= bb_12m_min * 1.1`. This only fires when the BB is at a 12-month extreme. A more useful check:

```python
# BB width trending down (contracting) even if not yet at annual minimum
bb_contracting = bb_width[-1] < bb_width[-5]  # tighter than 5 days ago
```

---

### Tier 4 — Architecture / Code Quality (lower priority)

These are already catalogued in `refactoring.md` but listed here for completeness in the context of making the pipeline operational:

1. **Fix async double-call bug** in `alerts.py` — the notification client is closed after the first `asyncio.run()` call, so email/Telegram alerts are never both delivered.
2. **Remove dead `trf_surge` variable** — computed but never used in scoring or filtering.
3. **Extract shared modules** from p06 imports — coupling makes p10 fragile if p06 is ever reorganized.
4. **Add minimum slope magnitude** to Phase 1.5 trend detection — current threshold `max_atr_slope: -0.0001` accepts trivially small slopes.

---

## 5. Recommended Implementation Order

```
Priority  Change                                          Effort    Expected Impact
────────  ──────────────────────────────────────────────  ────────  ──────────────────────────────────
1         Fix NaN guard in _check_accumulation            30 min    Eliminates false positives
2         Raise ATR threshold 2% → 4%                     15 min    ~3x more candidates reach next gate
3         Switch resistance gate to 20-day high ≤ 15%     30 min    ~12x more candidates at final gate
4         Relax or remove daily range gate                 15 min    ~2.5x more at compression stage
5         Fix async double-call in alerts.py               30 min    Alerts actually delivered
6         Add vol_zscore > 0 guard before AR calc          15 min    Semantically correct AR values
7         Intraday range compression (Improvement 3.1)     2h        Better signal quality
8         Tightening trend confirmation (Improvement 3.3)  2h        Fewer one-day noise signals
```

After items 1–4, the pipeline is expected to produce 3–10 candidates per run on normal market days, and 10–30+ candidates on high-volume market days.

---

## 6. How to Validate Changes

### 6.1 Backtesting approach

Run the recalibrated pipeline against existing result dates where diagnostics are available (`results/p10_emps3/*/08_absorption_diagnostics.csv`). Count how many tickers would have passed under the new thresholds:

```python
# Quick threshold sweep on existing diagnostics:
passz = df[df['vol_zscore'] >= 1.5]
passpr = passz[passz['price_range_1d'] < 0.05]   # new: 5%
passatr = passpr[passpr['atr_ratio'] < 0.04]      # new: 4%
passres = passatr[passatr['dist_52w_high'] <= 0.15]  # proxy for local high
print(f'{len(passres)} candidates would have passed (vs 5 actual)')
```

### 6.2 Forward validation

After deploying the changes, monitor `08_absorption_diagnostics.csv` for two weeks. Target range: **3–20 candidates per run**. If consistently > 30, tighten one threshold. If still < 1 per run, proceed to Tier 3 improvements.

### 6.3 Signal quality check

For any candidates that do pass, verify manually:
- Chart shows volume building while price is consolidating
- Stock is above SMA20 (not in a downtrend)
- No recent earnings, dilution, or halt activity

---

_This document should be updated after each threshold iteration with new funnel statistics._
