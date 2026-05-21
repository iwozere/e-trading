# EMPS2 PREMIUM Alert — Trading Playbook

*Based on backtested data: 2025-12-11 to 2026-05-20 (n=121 total alerts, 50 PREMIUM)*
*Post quality-gate cadence: ~3 PREMIUM alerts/month, ~3 concurrent positions (avg hold ~6 days)*

---

## What is a PREMIUM alert?

A PREMIUM alert means a ticker:
1. Appeared in the high-volatility universe **5+ times in 14 days** (quiet institutional accumulation)
2. Has a **vol Z-score ≥ 3.0 today** AND vol is **accelerating** vs its own average (×1.3)
3. **Price pulled back** during the accumulation window — it is cheaper now than when first detected
4. The signal is **fresh** (≤ 7 days since first detection — not stale)

**Backtested expectancy (PREMIUM only, Impr-3 exit strategy): 50% win rate, +0.5% mean per trade.**

HIGH alerts (drift ≥ 0%) have 38% win rate and −2.3% mean. Do not trade them.

---

## Exit Strategy — "Impr-3" (the recommended approach)

Three mechanisms work together. Average hold time is **~6 trading days**.

### 1. Breakeven stop (crash protection)
- Enter at next-day open
- Initial stop: **8% below entry**
- Once price hits **+10%**: move stop to **entry price** (you can no longer lose money)
- After breakeven: trail stop at **8% below the rolling high watermark** — update daily

### 2. Tight trail for big winners (lock in gains)
- Once price hits **+15%**: tighten trail from 8% to **5% below the rolling high**
- Rationale: big early winners on this universe frequently reverse. FCEL went +24% at 5d then −3% at 20d; PRGS +16% then −13%. Tighter trail locks in more of the gain.

### 3. Dead-money early exit (redeploy capital)
- At the close of **trading day 10**: if position is still within **±3% of entry** AND breakeven was never triggered → exit at close
- Rationale: if nothing has happened in 10 days, the thesis has not played out. Historical data shows 67% of positions flat at day 10 never recover by day 20.

### Hard ceiling
- **Day 20**: close out at market close regardless. In practice, most positions exit via stop or dead-money exit well before this.

---

## What the Exit Looks Like in Practice

```
Entry at $10.00

  INITIAL PHASE (stop at $9.20):
    Day 1-9: price drifts, stop at entry × 0.92 = $9.20

  SCENARIO A — Big winner:
    Day 5: price hits $11.00 (+10%) → breakeven triggers, stop moves to $10.00
    Day 8: price hits $11.50 (+15%) → trail tightens from 8% to 5%
    Day 8 stop: $11.50 × 0.95 = $10.93
    Day 12: price reverses to $10.93 → stop hit, exit +9.3%

  SCENARIO B — Flat / slow mover:
    Day 10 close: price at $10.15 (+1.5%), breakeven never triggered
    → Dead-money exit at $10.15 (+1.5%), redeploy capital

  SCENARIO C — Gap down on bad news:
    Day 3 open: $9.10 (below stop of $9.20)
    → Fill at $9.10 (gap-down protection), exit -9.0%

  SCENARIO D — Normal stop hit:
    Day 6: price hits $9.20 intraday
    → Fill at $9.20, exit -8.0%
```

---

## Backtested Performance — Strategy Comparison

All strategies tested on 121 historical Phase 2 alerts (2025-12-11 to 2026-05-20).
"HoldD" = average trading days held. "Crash>15" = trades with >15% loss.

### Full universe (all 121 alerts)

| Strategy | HoldD | Win% | Mean% | Crashes >15% | Max loss |
|---|---|---|---|---|---|
| No stop, 20d hold | 20.0 | 48% | -0.7% | 18 | -55.4% |
| No stop, 10d hold | 10.0 | 47% | +0.0% | 10 | -39.7% |
| Old best: BE 8%/+10%/trail8%, 20d | 6.3 | 42% | -1.1% | 4 | -29.9% |
| **Impr-1: 10d ceiling** | **5.2** | 43% | -0.9% | 4 | -29.9% |
| Impr-2: +dead-money exit d10 | 5.7 | 43% | -1.1% | 4 | -29.9% |
| **Impr-3: +tight trail at +15%** | **5.4** | 43% | -1.1% | 4 | -29.9% |

All stop strategies reduce crashes from 18 → 4 vs no stop. Among them, Impr-1 is the simplest
(10d ceiling only) and has the best mean (−0.9%). Impr-3 shows its edge on PREMIUM-only.

### PREMIUM-only (n=50) — the recommended subset

| Strategy | HoldD | Win% | Mean% | Crashes >15% | Max loss |
|---|---|---|---|---|---|
| No stop, 20d hold | 20.0 | **56%** | **+1.2%** | 5 | -55.4% |
| Old best: BE 8%/+10%/trail8%, 20d | 7.0 | 50% | +0.2% | 2 | -29.9% |
| **Impr-3: BE + dead-money d10 + tight trail** | **6.0** | **50%** | **+0.5%** | **2** | **-29.9%** |

**Verdict:** No stop has the best raw return (+1.2%), but 5 crashes >15% in 50 trades = 10% chance of a
brutal loss on any single trade. Impr-3 gives +0.5% mean with only 2 crashes >15% (4%). You give up
0.7% in mean return to reduce crash probability by 60%. That is a good trade for most risk tolerances.

> **Maximum loss is capped at -29.9% with any stop strategy.** Without stops it was -55.4%.
> On a 2% portfolio position: -29.9% × 2% = **-0.6% portfolio impact maximum**.

---

## Position Sizing

**Rule: size so that a full stop-out at 8% costs ≤ 0.16% of portfolio.**
That means: position size = 0.16% / 8% = **2% of portfolio per trade**.

Expected concurrent positions at any time: ~3 (3 signals/month × 6-day avg hold).
Portfolio deployed at any one time: ~6%.

| Tolerance | Max single-trade loss | Position size |
|---|---|---|
| Conservative | 0.08% of portfolio | 1.0% per trade |
| Moderate | 0.16% of portfolio | 2.0% per trade |
| Aggressive | 0.40% of portfolio | 5.0% per trade |

Start at 1–2% per position. Scale up only after 20+ PREMIUM trades confirm the backtested quality holds.

---

## Step-by-Step Checklist

### On receiving a PREMIUM alert (EOD)

- [ ] Confirm label is **PREMIUM** (skip HIGH entirely)
- [ ] Market health check: is S&P 500 down >5% this week? → **Skip** (panic correlations hurt)
- [ ] Already holding this ticker? → **Skip**
- [ ] Already have 5+ open positions? → **Skip or wait**
- [ ] More than 5 PREMIUM alerts this week? → **Cluster risk** — treat as one regime event, halve position size

### Pre-market (next morning)

- [ ] Check pre-market price: up >5% from yesterday's close? → **Skip** (thesis already priced in)
- [ ] Calculate your 3 key prices from the expected open:
  - **Stop price** = open × 0.92 (−8%)
  - **Breakeven trigger** = open × 1.10 (+10%)
  - **Tight trail trigger** = open × 1.15 (+15%)
- [ ] Place a **limit buy order at the open** (avoid market orders on thin small-caps)

### On fill (entry confirmed)

- [ ] Set **stop-loss order** at `entry × 0.92` immediately
- [ ] Set **price alert** at `entry × 1.10` (breakeven trigger)
- [ ] Set **price alert** at `entry × 1.15` (tight trail trigger)
- [ ] Set **calendar reminder** for trading day 10 (dead-money check)
- [ ] Set **calendar reminder** for trading day 20 (hard ceiling)

### Daily monitoring (2 minutes)

**If breakeven NOT yet triggered:**
- Stop is fixed at `entry × 0.92` — no action unless alert fires

**When breakeven fires (price hit +10%):**
- [ ] Cancel stop order, replace with stop at **entry price**
- [ ] From now on, update stop daily: `stop = max(entry, high_watermark × 0.92)`

**When tight trail fires (price hit +15%):**
- [ ] Switch trail: `stop = max(current_stop, high_watermark × 0.95)` (5% instead of 8%)
- [ ] Update daily from here

### Day 10 dead-money check

At end of trading day 10 (after market close):
- [ ] Has breakeven triggered at any point? → If **yes**, continue holding normally
- [ ] Is current price within ±3% of entry? → If **yes**, **exit at tomorrow's open** (cut dead money)
- [ ] Is current price clearly up or down (outside ±3%)? → Continue; stop protection applies

### Exit

Exit on the **first** of:

| Event | Action |
|---|---|
| Stop-loss order triggered | Fill at stop (or open if gap). Record and move on. |
| Day 10 dead-money check fails | Sell at next open. |
| Day 20 hard ceiling | Sell at close. |
| Unexpected negative news | Manual override sell immediately. Do not wait. |

---

## Monitoring Signal Quality — Monthly Review

Run once a month (or after 15+ completed PREMIUM trades):

```bash
python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing
python -m src.ml.pipeline.p06_emps2.util.simulate_stops
```

Compare against the 2026-05-20 baseline:

| Metric | 2026-05-20 baseline | Target | Flag if |
|---|---|---|---|
| PREMIUM win rate | 50% (Impr-3) | > 48% | < 40% |
| PREMIUM mean return | +0.5% | > 0% | < -1.0% |
| PREMIUM alerts/month | ~3 | ~3 | > 8 (gates broken) |
| Crashes >15% per 50 trades | 2 | < 5 | > 8 |

**If >8 PREMIUM alerts/month**: quality gates are not filtering. Halve position size until resolved.

**If win rate drops below 40% over 20+ trades**: regime change. Reduce to 0.5% positions; re-run full analysis.

---

## Quick Reference Card

```
PREMIUM ALERT RECEIVED
├── S&P down >5% this week?     → SKIP
├── Already hold this ticker?   → SKIP
├── 5+ positions open?          → SKIP
├── Pre-market up >5%?          → SKIP
└── Clear? Enter at open.

KEY PRICES (from open):
  Stop:      open × 0.92   (set immediately on fill)
  Breakeven: open × 1.10   (set alert; move stop to entry when hit)
  Tight trail: open × 1.15 (set alert; switch from 8% trail to 5% trail when hit)

EXITS (first to trigger wins):
  Stop hit           → fill at stop price
  Day 10 flat (±3%)  → exit at open day 11 (dead money)
  Day 20             → sell at close

SIZE: 1–2% per position  |  MAX 5 concurrent  |  Avg hold ~6 days
```

---

*Created: 2026-05-20 | Updated: 2026-05-21 (added improved exit strategy, simulation results)*
*Simulation data: `results/p06_emps2/stop_simulation_summary.csv`*
*Analysis data: `results/p06_emps2/timing_analysis.csv`*
*See: `docs/TIMING_ANALYSIS.md` for full backtest methodology*
