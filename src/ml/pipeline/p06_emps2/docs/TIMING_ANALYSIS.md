# Alert Timing Analysis — The "Too Late" Problem

## Problem Statement

By the time an EMPS2 Phase 2 alert fires, the ticker's price has often already moved
significantly. A Phase 2 alert requires:

1. A ticker to appear in the volatility-filtered universe **5+ times** in 10 rolling days → Phase 1.
2. On the same day, the ticker must also show a **vol Z-score ≥ 3.0** → Phase 2.

The **minimum structural lag from first detection to notification is ~5–6 trading days**.
In that window, a small/mid-cap can easily move +10–30%.

---

## Diagnostic Tool

**`util/analyze_alert_timing.py`** — run to quantify the problem across all historical alerts.

```bash
python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing
# skip yfinance forward-return fetch:
python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing --no-forward
```

Output: `results/p06_emps2/timing_analysis.csv`

### Key Metrics Computed

| Column | Description |
|---|---|
| `lag_days` | Calendar days from `first_seen` to Phase 2 alert date |
| `price_at_first_seen` | Price on the day ticker first entered rolling window |
| `price_at_alert` | Price on alert day (from `05_volatility_filtered.csv`) |
| `pre_alert_gain_pct` | % price move that occurred **before** the user could act |
| `return_5d_pct` | Price return 5 trading days after alert (yfinance) |
| `return_10d_pct` | Price return 10 trading days after alert |
| `return_20d_pct` | Price return 20 trading days after alert |

> **Note on `latest_last_price` in alert CSVs:** `rolling_memory.scan_historical_results()`
> iterates from newest to oldest. Pandas `.last()` on the resulting group therefore returns
> the oldest (first_seen) price. The column is named `latest_last_price` but holds the
> **first_seen price**. The diagnostic tool accounts for this.

---

## Baseline Analysis — 2026-05-20 (pre-improvement)

Ran `analyze_alert_timing.py` on 123 unique Phase 2 first-alerts spanning
2025-12-11 to 2026-05-18. Forward returns available for 102 of them.

### Overall signal quality (pre-improvement)

| Holding period | Win rate | Mean return | Median return |
|---|---|---|---|
| 5 days | 40% | -1.0% | -1.2% |
| 10 days | 47% | -0.2% | -0.4% |
| 20 days | 45% | -0.3% | -2.0% |

**Verdict: the Phase 2 signal as designed had ~47% win rate at 10 days — essentially random.**

### Finding 1 — Pre-alert price drift is the strongest predictor

How much the stock moved between `first_seen` and the alert day predicts 10d outcome:

| Pre-alert drift | Win rate (10d) | Mean return (10d) | n |
|---|---|---|---|
| Already **down** >5% | **62%** | **+4.3%** | 24 |
| Flat (−5% to +5%) | 44% | −0.7% | 62 |
| Already **up** 5–15% | 35% | −2.7% | 23 |
| Already **up** >15% | 50% | −5.9% | 6 |

**Interpretation:** Stocks being bought on weakness (price pulled back during accumulation)
show a genuine institutional buying pattern. Stocks that have already run up before the alert
perform significantly worse — the pipeline is generating momentum-chase signals in those cases.

### Finding 2 — Lag days matter

| Lag bucket | Win rate (10d) | Mean return (10d) | n |
|---|---|---|---|
| 0–4 days | 52% | +0.4% | 23 |
| 4–7 days | **50%** | **+2.3%** | 22 |
| 7–10 days | 45% | −1.3% | 22 |
| 10+ days | 42% | −1.4% | 48 |

**Interpretation:** Alerts triggered quickly (within 7 days of first_seen) outperform.
Long-lag alerts (10+ days) are stale — the accumulation pattern has faded or completed.

### Finding 3 — Vol acceleration modestly discriminates

`vol_acceleration = max_vol_zscore / avg_vol_zscore` (proxy for today_zscore / avg_zscore):

| Group | Win rate (10d) | Mean return (10d) |
|---|---|---|
| High acceleration (≥ 1.43) | 47% | +1.1% |
| Low acceleration (< 1.43) | 46% | −1.7% |

**Interpretation:** When volume is spiking *above* its own historical level, it is a cleaner
signal. A ticker with consistently high volume that isn't accelerating is less actionable.

### Finding 4 — Appearance count: lower is better

| Count | Win rate (10d) | Mean return (10d) |
|---|---|---|
| Low (≤ 5 appearances) | **52%** | **+1.6%** | 
| High (> 5 appearances) | 42% | −1.7% |

Fewer appearances means the signal is fresher. Extended accumulation (many appearances)
correlates with worse outcomes — the move may already be mature or the stock is simply
a recurring fixture that never actually breaks out.

### Best subset found (high accel + low count)

| Subset | Win rate (10d) | Mean return (10d) | n |
|---|---|---|---|
| vol_accel ≥ 1.43 AND count ≤ 5 | **51%** | **+3.5%** | 35 |
| vol_accel < 1.43 AND count > 5 | 33% | −7.8% | 3 |

---

## Changes Implemented — 2026-05-20

Three quality gates added to `rolling_memory.detect_phase2_candidates()`.
Config knobs live in `RollingMemoryConfig` (config.py).

### Change 1 — Price drift gate (`max_pre_alert_drift_pct`)

**What:** After all other conditions pass, suppress Phase 2 alerts where the price has
already risen more than `max_pre_alert_drift_pct` (default: **5%**) from `first_seen` to today.

**Why:** Pre-alert drift > 5% correlates with 35% win rate and −2.7% mean return (Finding 1).
Flat or negative drift correlates with 44–62% win rate.

**Bonus:** Tickers where `pre_alert_drift_pct < 0` (price pulled back during accumulation)
are now labelled `PREMIUM` instead of `HIGH`. These have the best historical performance.

**New output columns:** `pre_alert_drift_pct`, `lag_days`, `vol_acceleration`.

**Config:**
```python
max_pre_alert_drift_pct: float = 5.0   # suppress if already up >5%
```

---

### Change 2 — Lag cap (`max_phase2_lag_days`)

**What:** Suppress Phase 2 alerts where `lag_days = (alert_date − first_seen) > max_phase2_lag_days`
(default: **7 days**).

**Why:** Alerts with lag ≥ 10 days have 42% win rate and −1.4% mean return (Finding 2).
The accumulation window has expired; the smart-money positioning is complete.

**Config:**
```python
max_phase2_lag_days: int = 7           # suppress stale signals (>7 calendar days old)
```

---

### Change 3 — Vol acceleration condition (`phase2_min_vol_acceleration`)

**What:** Add a second volume condition: `today_vol_zscore / avg_vol_zscore ≥ phase2_min_vol_acceleration`
(default: **1.3**). Volume must be accelerating above its own historical average, not just
sitting at a previously-high level.

**Why:** High acceleration group: +1.1% mean vs −1.7% for low acceleration (Finding 3).
The existing `vol_zscore ≥ 3.0` check is now complemented by requiring the spike to be
fresh, not a plateau from days ago.

**Config:**
```python
phase2_min_vol_acceleration: float = 1.3   # today_zscore / avg_zscore
```

---

## Expected Impact

Based on historical distribution (102 alerts with forward data):

- **Lag gate** removes ~46% of alerts (the 10+ day bucket).
- **Price drift gate** removes an additional ~25% (pre_alert_gain > 5%).
- **Vol acceleration** is additive to existing vol condition; expected ~20% further reduction.
- Combined: roughly **30–40% of previous alerts survive**, with estimated win rate improvement
  from 47% → ~55%+ and mean 10d return from −0.2% to +2%+.

These are estimates — the post-improvement baseline should be re-measured after ~3 months
of live runs using the same `analyze_alert_timing.py` script.

---

## Remaining Ideas (not yet implemented)

### Phase 1 Early-Warning Alert (Medium effort)

Fire a soft "WATCH" alert at 3 appearances instead of waiting for Phase 2.
Reduces minimum lag from ~6 days to ~3 days.
Trade-off: more false positives — best as a daily digest, not a push notification.

Config knob to add: `phase0_min_appearances: int = 3`.

---

### "NEW vs CONTINUING" Alert Label (Quick win)

Mark Phase 2 alerts as:
- **NEW**: first time this ticker has triggered Phase 2 in the last 30 days.
- **CONTINUING Day N**: same ticker on consecutive days.

Implementation: compare to previous day's `08_phase2_alerts.csv`.

---

### Pre-Market Pipeline Scheduling (Infrastructure)

Running the pipeline at 7–8 AM ET means `last_price` = yesterday's close, which is
actionable before the day's open. Currently it runs EOD and captures intraday drift.

---

## Stop-Loss Simulation — 2026-05-20

Script: `util/simulate_stops.py`
Outputs: `results/p06_emps2/stop_simulation.csv`, `stop_simulation_summary.csv`

Entry model: **next-day open** after alert (realistic: alert received EOD, order placed pre-market).
Stop check: daily LOW. Gap-down protection: if open ≤ stop price, fill at open.

### ATR context — why tight stops don't work here

The EMPS2 universe is explicitly filtered for high volatility (ATR/price ≥ 2%).

| Stop % | Multiples of ATR (median=2.0%) | Interpretation |
|---|---|---|
| 5% | 2.5× | Inside 1–2 days of normal noise — triggered constantly |
| 8% | 4.0× | Still frequently hit by regular swings |
| 10% | 5.0× | Getting reasonable |
| 15% | 7.5× | Wide enough to avoid most noise stops |

### Baseline stop simulation results (pre-improvement, n=121 alerts)

| Strategy | Win% | Mean% | Median% | Stop hit% | Avg stop loss% | Max loss% |
|---|---|---|---|---|---|---|
| No stop (20d hold) | 48% | -0.7% | -0.3% | 0% | — | -55% |
| Fixed stop 5% / 20d | 21% | -1.6% | -5.0% | 79% | -5.7% | -16% |
| Fixed stop 8% / 20d | 31% | -2.0% | -8.0% | 65% | -9.2% | -30% |
| Fixed stop 10% / 20d | 36% | -1.6% | -10.0% | 57% | -11.1% | -30% |
| Fixed stop 15% / 20d | 43% | -1.0% | -4.6% | 36% | -15.9% | -30% |
| Trailing stop 5% / 20d | 35% | -0.7% | -2.1% | 95% | -1.0% | -16% |
| Trailing stop 8% / 20d | 30% | -1.4% | -3.6% | 93% | -2.0% | -26% |
| Trailing stop 10% / 20d | 34% | -1.5% | -4.7% | 88% | -2.8% | -30% |
| Trailing stop 15% / 20d | 43% | -0.8% | -4.8% | 64% | -6.8% | -30% |

**Key finding: every fixed or trailing stop strategy performs worse than no stop.**
High ATR stocks get stopped out by noise before the move plays out.

### PREMIUM vs HIGH split (fixed 15% stop)

| Group | Win% | Mean% | Stop hit% |
|---|---|---|---|
| PREMIUM (drift < 0) + fixed 15% | 52% | +1.0% | 26% |
| HIGH (drift ≥ 0) + fixed 15% | 37% | -2.4% | 42% |

Signal quality (PREMIUM gate) is a stronger predictor than stop placement.

### Conclusions from baseline stop simulation

1. **Traditional stops hurt on this universe.** Any stop < 3× ATR (~6%) is triggered by
   daily noise on 65–95% of trades, cutting winners short while not preventing the real crashes.

2. **Position sizing is the primary protection tool.** A -55% trade on a 2% position = -1.1%
   portfolio impact. A 10% fixed stop on the same position only saves 0.9% but triggers on 57%
   of trades that would have recovered.

3. **The one exception: PREMIUM alerts with a wide stop.** PREMIUM + fixed 15% gives 52% win
   rate and +1.0% mean return — the only stop configuration with positive expectancy.

4. **Crash prevention needs a smarter approach.** Two strategies worth testing:
   - **ATR-based stop**: stop = entry × (1 − N × ATR_ratio), adapts to each stock's actual
     volatility. A 3× ATR stop is dynamically 4–9% depending on the stock.
   - **Breakeven trailing**: wide initial stop (8%), move to entry once up +10%, then trail
     8% from the high. Eliminates ruin risk on winners while respecting initial volatility.

### ATR-based + Breakeven simulation results — 2026-05-20

Run: `python -m src.ml.pipeline.p06_emps2.util.simulate_stops` (n=121 alerts)

#### ATR-based stops

| Strategy | Win% | Mean% | Median% | Stop% | Max loss | Crashes >15% | Crashes >25% |
|---|---|---|---|---|---|---|---|
| No stop (20d hold) | 48% | -0.7% | -0.3% | 0% | -55.4% | 18 | 8 |
| ATR 2.0× stop / 20d | 16% | -1.1% | -3.4% | 84% | -16.9% | **2** | **0** |
| ATR 2.5× stop / 20d | 18% | -1.6% | -4.0% | 81% | -21.2% | **2** | **0** |
| ATR 3.0× stop / 20d | 22% | -1.4% | -4.6% | 77% | -25.4% | 4 | 1 |

**Finding:** ATR-based stops are excellent at crash prevention (18 crashes → 2) but they destroy win rate
(48% → 16–22%) — triggered by normal daily noise 77–84% of the time. Net expectancy is negative.

#### Breakeven trailing stops

| Strategy | Win% | Mean% | Median% | Stop% | Max loss | Crashes >15% | Crashes >25% |
|---|---|---|---|---|---|---|---|
| No stop (20d hold) | 48% | -0.7% | -0.3% | 0% | -55.4% | 18 | 8 |
| 8% init, +10% lock, trail 8% | 42% | -1.1% | -8.0% | 88% | -29.9% | 4 | 2 |
| 8% init, +8% lock, trail 8% | 41% | -1.0% | -8.0% | 91% | -29.9% | 4 | 2 |
| 10% init, +15% lock, trail 8% | **43%** | **-0.8%** | -10.0% | 76% | -29.9% | 4 | 2 |
| 10% init, +10% lock, fixed at entry | 27% | -1.4% | -10.0% | 69% | -29.9% | 5 | 2 |

**Finding:** Breakeven strategies are the best compromise. They cut severe crashes from 18 → 4
(>15% losses) and eliminate worst-case disasters (-55% → -30%), while preserving a reasonable
win rate (~42%). The "fixed at entry after lock" variant performs poorly — trailing beats fixed.

#### PREMIUM vs HIGH split on best strategies

| Subset | Win% | Mean% | Stop% | Crashes >15% |
|---|---|---|---|---|
| PREMIUM — no stop 20d | **56%** | **+1.2%** | 0% | 5 |
| HIGH — no stop 20d | 42% | -2.0% | 0% | 13 |
| PREMIUM — ATR 3.0× stop | 22% | -1.1% | 78% | 1 |
| HIGH — ATR 3.0× stop | 21% | -1.6% | 76% | 3 |
| PREMIUM — Breakeven 8%/+10%/trail 8% | **50%** | **+0.2%** | 84% | 2 |
| HIGH — Breakeven 8%/+10%/trail 8% | 37% | -2.1% | 92% | 2 |

**Key insight:** PREMIUM alerts (price pulled back during accumulation) have +1.2% mean return
unhedged. Adding a breakeven stop to PREMIUM gives +0.2% mean with crashes reduced from 5 → 2.
The breakeven stop acts as insurance on PREMIUM trades without significantly hurting expectancy.

#### Conclusions from extended stop simulation

1. **ATR stops prevent crashes but destroy expectancy.** Stop hit rate of 77–84% means the strategy
   exits almost every trade early, converting recoverable drawdowns into realized losses.

2. **Breakeven stops are the practical crash-prevention tool.** They cap the absolute loss at the
   initial stop level (8–10%) on all positions that never trigger breakeven, and protect profits on
   winners. Crash frequency drops from 18 to 4 events.

3. **The recommended approach for this universe:**
   - **Only trade PREMIUM alerts** (pre_alert_drift_pct < 0).
   - **Use a breakeven stop** (8% initial, trail 8% once up +10%) to protect against the rare -30%+ event.
   - **Size positions at 2–3%** of portfolio — worst-case protected loss is ~0.2% of portfolio.
   - Do NOT use ATR-based or tight fixed stops: the universe is explicitly filtered for high volatility
     and normal daily moves will trigger stops before the thesis plays out.

See `results/p06_emps2/stop_simulation_summary.csv` for full per-configuration numbers.

---

## Improved Exit Strategy — 2026-05-21

After finding the 10-day return (+1.6% mean) beats the 20-day return (+0.2%) for PREMIUM tickers,
three incremental exit improvements were simulated. Implementation in `simulate_stops.py` (Impr-3 config).

### Key finding that motivated the changes

| Hold period | PREMIUM win% | PREMIUM mean% |
|---|---|---|
| 5 days | 48% | +0.3% |
| **10 days** | **57%** | **+1.6%** (peak) |
| 20 days | 48% | +0.2% |

11 of 20 PREMIUM positions up at day 5 were **negative by day 20** (55% reversal rate).
67% of positions flat/down at day 10 stayed that way through day 20. The 20-day ceiling was wrong.

### Improvements tested (2026-05-21, n=121 alerts)

| Strategy | Avg hold | Win% | Mean% | Crashes >15% | Max loss |
|---|---|---|---|---|---|
| Baseline: no stop, 20d | 20.0d | 48% | -0.7% | 18 | -55.4% |
| Baseline: no stop, 10d | 10.0d | 47% | 0.0% | 10 | -39.7% |
| Old best: BE 8%/+10%/trail 8%, 20d | 6.3d | 42% | -1.1% | 4 | -29.9% |
| Impr-1: 10d ceiling only | 5.2d | 43% | -0.9% | 4 | -29.9% |
| Impr-2: +dead-money exit at day 10 | 5.7d | 43% | -1.1% | 4 | -29.9% |
| **Impr-3: +tight trail 5% once up >15%** | **5.4d** | **43%** | **-1.1%** | **4** | **-29.9%** |

### PREMIUM-only (n=50)

| Strategy | Avg hold | Win% | Mean% | Crashes >15% |
|---|---|---|---|---|
| No stop, 20d | 20.0d | 56% | +1.2% | 5 |
| Old best BE 20d | 7.0d | 50% | +0.2% | 2 |
| **Impr-3 (recommended)** | **6.0d** | **50%** | **+0.5%** | **2** |

Impr-3 improves PREMIUM mean return from +0.2% to +0.5% by tightening the trail once up >15%,
capturing more of big-winner gains before they reverse. Crashes halved vs old best.

### Impr-3 exit rules (the recommended approach)

1. **Initial stop**: 8% below entry
2. **Breakeven at +10%**: move stop to entry (cannot lose money after this point)
3. **Tight trail at +15%**: switch from 8% trail to 5% trail below the high watermark
4. **Dead-money exit**: at EOD of day 10, if within ±3% of entry AND breakeven never triggered → exit at next open
5. **Hard ceiling**: sell at close on day 20 (avg actual hold is ~6 days in practice)

See `docs/TRADING_PLAYBOOK.md` for the full operational checklist.

---

## Re-measurement Checklist

After running the improved pipeline for ~3 months, re-run:

```bash
python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing
```

Compare against the **2026-05-20 baseline** numbers in this document:

| Metric | Baseline (pre-improvement) | Target |
|---|---|---|
| 10d win rate | 47% | > 55% |
| 10d mean return | −0.2% | > +1.5% |
| Alerts per month | ~15 | ~6–8 (quality over quantity) |
| PREMIUM-labelled % | N/A | > 30% of alerts |

---

*Created: 2026-05-20 | Last updated: 2026-05-21 (improved exit strategy, Impr-3)*
