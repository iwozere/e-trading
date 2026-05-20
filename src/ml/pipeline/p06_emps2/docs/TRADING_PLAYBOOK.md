# EMPS2 PREMIUM Alert — Trading Playbook

*Based on backtested data: 2025-12-11 to 2026-05-20 (n=121 alerts, 50 PREMIUM)*
*Post quality-gate expectation: ~3 PREMIUM alerts/month, ~3 concurrent positions*

---

## What is a PREMIUM alert?

A PREMIUM alert means a ticker:
1. Appeared in the high-volatility universe **5+ times in 14 days** (quiet institutional accumulation)
2. Has a **vol Z-score ≥ 3.0 today** AND vol is **accelerating** vs its own average (×1.3)
3. **Price pulled back** during the accumulation window — it is cheaper now than when first detected
4. The signal is **fresh** (≤ 7 days since first detection — not stale)

Backtested win rate: **56%, mean return +1.2% over 20 days** (vs 48%/-0.7% for ALL alerts).

HIGH alerts (drift ≥ 0%) have a 42% win rate and −2.0% mean. Ignore them.

---

## Position Sizing

**Expected cadence (post quality gates):**
- ~3 PREMIUM alerts per month
- ~3 positions open simultaneously (20-day hold × 3/month)
- Old rate (pre-gates) was ~8 PREMIUM/month — if you see this, something is wrong with the filters

**Risk budget per trade:**
- Decide your max acceptable single-trade loss as a fraction of portfolio first.
- With an 8% stop, that means: **position size = risk budget / 0.08**

| If max single-trade loss is... | Position size |
|---|---|
| 0.08% of portfolio | 1.0% per trade |
| 0.16% of portfolio | 2.0% per trade |
| 0.40% of portfolio | 5.0% per trade |

**Recommended starting point: 1–2% per position.**
- At 2% per position, 3 concurrent = 6% of portfolio deployed at once.
- Worst case (stop hit on all 3): −0.48% total portfolio impact.
- This universe is explicitly high-volatility — do not size aggressively.

> If you are getting 8+ PREMIUM alerts/month, the gates are not working. Reduce position size to
> 0.5% until you re-run `analyze_alert_timing` and verify the quality is still there.

---

## Step-by-Step Checklist

### Step 1 — Alert received (EOD, after pipeline runs)

- [ ] Confirm it is labelled **PREMIUM** (not HIGH — skip HIGH entirely)
- [ ] Note: `ticker`, `alert_date`, `pre_alert_drift_pct`, `lag_days`
- [ ] Check overall market: if S&P 500 is in a >5% single-week drawdown, **skip this alert** — high-vol stocks get hit hardest in panics
- [ ] Check if you already hold this ticker or have a recent stop-out on it (within 10 days) — if so, skip
- [ ] Count your open PREMIUM positions. If you already have 5+ open, skip or wait for one to close

---

### Step 2 — Pre-market (next morning, before open)

- [ ] Look up yesterday's close and today's pre-market price
- [ ] **Do not chase**: if the stock is already up >5% pre-market from yesterday's close, **skip** — the move may already be live
- [ ] Place a **limit order at the open price** (or set it to fill at open). Do not use a market order on illiquid small-caps
- [ ] Calculate your stop price: `stop = open_price × 0.92` (8% below entry)
- [ ] Calculate your breakeven trigger price: `breakeven = open_price × 1.10` (10% above entry)
- [ ] Write these three numbers down: **entry, stop, breakeven trigger**

---

### Step 3 — Order is filled (entry confirmed)

- [ ] Immediately place a **stop-loss order** at `entry × 0.92`
  - Use a stop-limit or stop-market depending on your broker
  - Stop-limit: limit = stop − 0.5% (protects against gap fills at extreme prices)
- [ ] Set a **price alert** at `entry × 1.10` to trigger the breakeven adjustment
- [ ] Set a **calendar reminder** 20 trading days from today for the time-exit

---

### Step 4 — While holding (daily check, 2 minutes)

Every day until exit, check:

**If price has NOT yet hit breakeven trigger (`entry × 1.10`):**
- [ ] Stop is still at `entry × 0.92` — no action needed
- [ ] If price gaps down below stop at open → you are filled at open (gap protection). Accept the loss.

**If price has hit breakeven trigger (you were notified):**
- [ ] Move stop up to `entry × 1.00` (break even — can no longer lose money on this trade)
- [ ] From this point: trail stop at `high_watermark × 0.92` (8% below the highest price since entry)
- [ ] Update stop each day: `new_stop = max(current_stop, today_high × 0.92)`
- [ ] Cancel and replace the stop-loss order with the new level

---

### Step 5 — Exit

Exit happens on the **first** of these events:

| Event | Action |
|---|---|
| Stop-loss order triggered | Fill at stop (or open if gap-down). Record loss. |
| 20 trading days reached | Sell at close. Record result. |
| Unexpected negative news (FDA rejection, fraud allegation, etc.) | Manual override sell. Do not wait for stop. |

- [ ] Record the trade: entry date, exit date, entry price, exit price, return %, reason for exit
- [ ] Update your running log (see Monitoring section below)

---

## What the Stops Actually Do

```
Entry at $10.00

  Phase 1 (initial stop):
  Stop at $9.20 (-8%)
  Max loss if stopped: -8% of position

  If price rises to $11.00 (+10%):
  → Breakeven triggers
  → Move stop to $10.00 (entry)
  → Max loss is now $0

  If price continues to $12.00:
  → Trail stop = $12.00 × 0.92 = $11.04
  → Max loss is now a GAIN of +$1.04 per share (+10.4%)

  If price then falls from $12 to $11.04:
  → Stop triggered. Exit at ~$11.04
  → Locked in +10.4% gain
```

**The breakeven stop acts as insurance**: you pay a small premium (slightly lower win rate vs no stop)
in exchange for eliminating ruin risk on the position.

---

## Monitoring Results — Monthly Review

Run once a month (or after 10+ completed trades):

```bash
python -m src.ml.pipeline.p06_emps2.util.analyze_alert_timing
```

Compare against the 2026-05-20 baseline:

| Metric | Baseline (pre-gate) | Post-gate target | Flag if... |
|---|---|---|---|
| PREMIUM win rate (20d) | 56% | > 55% | Drops below 45% |
| PREMIUM mean return (20d) | +1.2% | > +0.5% | Goes negative |
| PREMIUM alerts/month | ~8 (old) | ~3 | Goes above 8 |
| % of alerts labelled PREMIUM | 41% | > 40% | Drops below 25% |

**If win rate drops below 45% over 20+ PREMIUM trades**: the market regime may have changed.
Reduce position size to 0.5% until 10 more trades confirm the signal quality is back.

**If you see 8+ PREMIUM alerts in a single month**: likely a market panic creating many "pullback"
signals simultaneously. These tend to cluster in volatile periods and underperform individually.
Treat a cluster of 5+ in one week as a single regime event, not 5 independent setups.

---

## Quick Reference Card

```
RECEIVED PREMIUM ALERT
├── Market in panic (S&P -5% this week)?  → SKIP
├── Already hold this ticker?             → SKIP
├── 5+ open positions?                   → SKIP
├── Pre-market up >5%?                   → SKIP
└── All clear?
    ├── Entry:    limit at open
    ├── Stop:     entry × 0.92   (set immediately on fill)
    ├── Trigger:  entry × 1.10   (set price alert)
    └── After trigger fires:
        ├── Move stop → entry (floor)
        └── Trail stop → high × 0.92 (update daily)

EXIT: stop hit  OR  day 20 (sell at close)  OR  bad news (manual)
SIZE: 1–2% per position (max 5 concurrent positions = 5–10% portfolio)
```

---

*Created: 2026-05-20 | Based on TIMING_ANALYSIS.md backtested findings*
*See simulate_stops.py for full stop-strategy simulation data*
