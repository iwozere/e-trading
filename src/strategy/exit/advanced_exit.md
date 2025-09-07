# ATR-Based Exit Strategy — Technical Specification (Advanced)

## 1) Purpose & Scope

Design a **robust, volatility-adaptive trailing stop** exit for long/short positions, centered on ATR. The stop:

* trails only in the favorable direction,
* adapts to volatility (multi-ATR, multi-timeframe),
* supports break-even, phase switching, structural (swing) ratchets, time-based tightening, and partial take-profits,
* is suitable for both **Backtesting** and **Live** trading (e.g., Backtrader integration).

---

## 2) Definitions

* **True Range (TR)** for bar *t*:
  `TR_t = max(High_t − Low_t, |High_t − Close_{t-1}|, |Low_t − Close_{t-1}|)`
* **ATR(p, mode)**:
  Smoothing modes: `RMA/Wilder (default)`, `EMA`, or `SMA`.
  `ATR_fast = ATR(p_fast)`, `ATR_slow = ATR(p_slow)`, optional `ATR_htf = ATR on higher timeframe`.
* **Effective ATR (ATR\_eff)** (aggregator):
  `ATR_eff_t = max( α_f * ATR_fast_t, α_s * ATR_slow_t, α_h * ATR_htf_t, ATR_floor )`
  where `α_f, α_s, α_h ≥ 0`, `ATR_floor` prevents over-tight stops in low vol.
* **Risk unit (R0)**: initial risk distance at entry, e.g. `R0 = EntryPrice − InitialStop` (long).

---

## 3) Inputs & Parameters

### Core

* `side`: long | short
* `anchor`: `high` | `close` | `mid` — price reference to trail against (default: `high` for longs, `low` for shorts).
* `update_on`: `bar_close` | `intrabar` | `high_low` (default: `bar_close`)
* `k_init` (float): initial ATR multiplier (default: 2.5; range 1.0–6.0)
* `k_run` (float): running ATR multiplier after arming/trailing (default: 2.0; 1.0–5.0)
* `p_fast` (int): ATR fast period (default: 7; 5–21)
* `p_slow` (int): ATR slow period (default: 21; 14–100)
* `use_htf_atr` (bool): include higher-TF ATR (default: true)
* `htf` (str): higher timeframe (e.g., `4h` when trading `15m`)
* `α_f, α_s, α_h` (floats): weights for ATR aggregator (defaults: 1.0, 1.0, 1.0)
* `ATR_floor` (float): absolute min ATR in ticks/price units (default: 0)

### Break-even & Phases

* `arm_at_R` (float): move to break-even after profit ≥ `arm_at_R * R0` (default: 1.0; 0.5–2.0)
* `breakeven_offset_atr` (float): offset above entry after BE arm, in ATR\_eff (default: 0.0; −0.5–0.5)
* `phase2_at_R` (float): switch to tighter trailing after profit ≥ `phase2_at_R * R0` (default: 2.0; 1.0–4.0)
* `k_phase2` (float): ATR multiplier in phase 2 (default: 1.5; 0.8–3.0)

### Structural Ratchet (Swing-based)

* `use_swing_ratchet` (bool): enable (default: true)
* `swing_lookback` (int): lookback bars for last swing high/low since entry (default: 10)
* `struct_buffer_atr` (float): place stop beyond swing by this ATR (default: 0.25; 0–1.0)

### Time-based Tightening & Stagnation

* `tighten_if_stagnant_bars` (int): if no new HH/LL for N bars, tighten (default: 20; 10–60)
* `tighten_k_factor` (float): multiply current k by this factor (default: 0.8; 0.6–0.95)
* `min_bars_between_tighten` (int): cooldown between tighten events (default: 5)

### Noise & Step Filters

* `min_stop_step` (price units): minimum increment to move the stop (default: 0)
* `noise_filter_atr` (float): skip trailing updates if current bar range < this \* ATR\_eff (default: 0; 0–0.5)
* `max_trail_freq` (bars): at most one trail move per this many bars (default: 1)

### Partial Take-Profit (optional but recommended)

* `pt_levels_R` (list): e.g., `[1.0, 2.0]`
* `pt_sizes` (list): e.g., `[0.33, 0.33]`
* `retune_after_pt`: adjust k after each PT (e.g., reduce to `k_phase2`) (default: true)

### Execution & Slippage

* `fill_policy`: `stop_mkt` | `stop_limit` (default: `stop_mkt`)
* `slippage_model`: backtest fill assumptions (gap through stop → filled at next bar open or stop, configurable)
* `tick_size`, `lot_step`: rounding rules

### Optimization (Optuna)

Suggest optimizing:
`k_init, k_run, p_fast, p_slow, α_f, α_s, α_h, arm_at_R, breakeven_offset_atr, phase2_at_R, k_phase2, struct_buffer_atr, tighten_if_stagnant_bars, tighten_k_factor, noise_filter_atr`
With conditional parameters: `use_htf_atr`, `use_swing_ratchet`.

---

## 4) State Machine

**States:** `INIT → ARMED → PHASE1 → PHASE2 → LOCKED → EXIT`

* **INIT**: on entry, set `InitialStop = Entry −/+ k_init * ATR_eff_entry` (long/short).
* **ARMED (Break-even ready)**: when price ≥ `Entry + arm_at_R * R0` (long), set
  `Stop = max(Stop, Entry + breakeven_offset_atr * ATR_eff)` (for short: symmetric). Then transition to PHASE1.
* **PHASE1 (Running trail)**:
  Trail using `k_run * ATR_eff` with anchor. If profit ≥ `phase2_at_R * R0` ⇒ PHASE2.
* **PHASE2 (Tighter trail)**:
  Trail using `k_phase2 * ATR_eff`. Optional move to **LOCKED** once partials are taken or new equity HH.
* **LOCKED**: parabolic-style tightening (e.g., apply `tighten_k_factor` on stagnation; structural ratchet dominates).
* **EXIT**: stop hit → close position, log metrics.

---

## 5) Trailing Logic (per bar)

For **long** (mirror for short):

1. Compute `ATR_fast, ATR_slow, ATR_htf (if enabled) → ATR_eff_t`.
2. Choose **distance** `D_t = k_current * ATR_eff_t`.
3. Choose **anchor price**:

   * `anchor=high`: `candidate = HighestHighSinceEntry − D_t`
   * `anchor=close`: `candidate = Close_t − D_t`
   * `anchor=mid`: `(High_t + Low_t)/2 − D_t`
4. **Noise filter**: if `BarRange_t < noise_filter_atr * ATR_eff_t` → skip update.
5. **Max frequency**: if last update < `max_trail_freq` bars → skip.
6. **Structural ratchet (optional)**:

   * `lastSwingLow = LLV_since_entry( Low, swing_lookback )`
   * `struct_stop = lastSwingLow − struct_buffer_atr * ATR_eff_t`
   * `candidate = max(candidate, struct_stop)`
7. **Break-even/phase constraints**: if in ARMED and BE condition met → set BE stop as floor.
8. **Time-based tightening**: if no new HH for `tighten_if_stagnant_bars` and cooldown passed → `k_current *= tighten_k_factor` (bounded by `k_min`, e.g., 0.8).
9. **Ratcheting**: `Stop_t = max(Stop_{t−1}, candidate)` (never loosens for long).
10. **Min step**: if `(Stop_t − Stop_{t−1}) < min_stop_step` → keep `Stop_{t} = Stop_{t−1}`.
11. If `Low_t ≤ Stop_{t−1}` (long) → **EXIT** at fill per `slippage_model`.

**Short**: replace highs with lows, maxima with minima, greater/less comparisons inverted, and use `min(Stop_{t−1}, candidate)` for ratcheting.

---

## 6) Partial Take-Profit (PT)

* On reaching each level `Entry + R_i * R0` (long), exit `pt_sizes[i]` of position.
* After PT, optionally:

  * switch to `k_phase2`,
  * reduce `phase2_at_R` thresholds,
  * tighten structural buffer (`struct_buffer_atr *= 0.8`).

---

## 7) Multi-Timeframe ATR

* Compute ATR on trade timeframe (`TTF`) and higher timeframe (`HTF`, e.g., 4× TTF).
* Synchronize `HTF` bars to `TTF` using last known `HTF` ATR.
* Aggregator: `ATR_eff = max(α_f*ATR_fast_TTF, α_s*ATR_slow_TTF, α_h*ATR_HTF, ATR_floor)`
* Rationale: avoids over-tight stops in micro-noise while respecting regime-level volatility.

---

## 8) Edge Cases & Rules

* **Warm-up**: no trailing until all ATR windows are ready (require `max(p_fast, p_slow)` bars and at least 1 HTF bar).
* **Gaps**: if price gaps beyond stop:

  * `stop_mkt`: exit at next bar open (backtest) or best available (live).
  * `stop_limit`: may not fill; log slippage and missed exit risk.
* **Decimals**: round stops to `tick_size`.
* **Fees & slippage**: incorporate in performance metrics.
* **Position sizing**: R0 drives reporting; sizing is external to this module.

---

## 9) Outputs & Logging

On every update and on exit, log:

* timestamp, symbol, side, entry price/time
* ATR\_fast, ATR\_slow, ATR\_htF, ATR\_eff
* k\_current, state, D\_t, anchor, candidate, final Stop
* events: BE armed, Phase switch, PT fills, Tighten events
* fill details on exit (price, slippage, reason: stop hit/gap/pt)

Persist as CSV/Parquet; add optional JSON event stream for UI.

---

## 10) Backtesting vs Live Consistency

* **Update cadence**: default `bar_close`. For live, allow intrabar updates only if broker supports server-side stops; otherwise simulate at bar close to avoid look-ahead bias.
* **Order model**: unify stop type (`stop_mkt` recommended for reliability).
* **Time sync**: ensure ATR\_htf alignment to avoid repaint.
* **Determinism**: seed any stochastic components (none expected here).

---

## 11) Configuration Schema (example)

```yaml
exit:
  anchor: high
  update_on: bar_close
  atr:
    p_fast: 7
    p_slow: 21
    use_htf_atr: true
    htf: 4h
    alpha_fast: 1.0
    alpha_slow: 1.0
    alpha_htf: 1.0
    floor: 0.0
  k:
    init: 2.5
    run: 2.0
    phase2: 1.5
  breakeven:
    arm_at_R: 1.0
    offset_atr: 0.0
  phases:
    phase2_at_R: 2.0
  structure:
    use_swing_ratchet: true
    swing_lookback: 10
    struct_buffer_atr: 0.25
  time_tighten:
    tighten_if_stagnant_bars: 20
    tighten_k_factor: 0.8
    min_bars_between_tighten: 5
  filters:
    noise_filter_atr: 0.0
    min_stop_step: 0.0
    max_trail_freq: 1
  partial_tp:
    levels_R: [1.0, 2.0]
    sizes: [0.33, 0.33]
    retune_after_pt: true
  execution:
    fill_policy: stop_mkt
    slippage_model: gap_to_open
    tick_size: 0.01
    lot_step: 1
```

---

## 12) Optuna Optimization Plan

**Objective(s):**

* Primary: `Sharpe` or `Calmar` (stable),
* Constraints: `MaxDD ≤ X%`, `Loss streak ≤ Y`, `Trades ≥ N`.

**Search space (suggested):**

* `k_init: Uniform(1.0, 6.0)`
* `k_run: Uniform(1.0, 5.0)`
* `p_fast: Int(5, 21)`
* `p_slow: Int(14, 100)`
* `use_htf_atr: Categorical([True, False])`
* `α_f, α_s, α_h: LogUniform(0.25, 2.0)`
* `arm_at_R: Uniform(0.5, 2.0)`
* `breakeven_offset_atr: Uniform(-0.25, 0.25)`
* `phase2_at_R: Uniform(1.0, 4.0)`
* `k_phase2: Uniform(0.8, 3.0)`
* `use_swing_ratchet: Categorical([True, False])`
* `swing_lookback: Int(5, 30)`
* `struct_buffer_atr: Uniform(0.0, 0.6)`
* `tighten_if_stagnant_bars: Int(10, 60)`
* `tighten_k_factor: Uniform(0.6, 0.95)`
* `noise_filter_atr: Uniform(0.0, 0.4)`

**Procedure:**

1. **Stage A (core):** tune `k_init, k_run, p_fast, p_slow` (+ `use_htf_atr`).
2. **Stage B (risk):** add BE, phase2, structural ratchet.
3. **Stage C (polish):** time-tighten & noise filters.
4. **Walk-Forward:** rolling windows (train/validate), preserve only parameter sets that generalize across folds.
5. **Stress tests:** higher fees/slippage, random bar drops, volatility regime shifts.

---

## 13) Pseudo-Algorithm (long; mirror for short)

```
on_entry:
  compute ATR_eff_entry
  Stop = Entry - k_init * ATR_eff_entry
  State = INIT

on_each_bar:
  compute ATR_fast, ATR_slow, (ATR_htf if enabled) → ATR_eff
  D = k_current * ATR_eff
  candidate = anchor_price_since_entry - D                         # high|close|mid variant

  if noise_filter and bar_range < noise_filter_atr * ATR_eff: skip update
  if not cooldown_passed(max_trail_freq): skip update

  if use_swing_ratchet:
      lastSwingLow = LLV_since_entry(Low, swing_lookback)
      struct_stop  = lastSwingLow - struct_buffer_atr * ATR_eff
      candidate = max(candidate, struct_stop)

  if State == INIT and Profit ≥ arm_at_R * R0:
      Stop = max(Stop, Entry + breakeven_offset_atr * ATR_eff)
      State = PHASE1

  if State == PHASE1 and Profit ≥ phase2_at_R * R0:
      k_current = k_phase2
      State = PHASE2

  if stagnation_detected and cooldown_ok:
      k_current = max(k_min, k_current * tighten_k_factor)

  newStop = max(Stop, candidate)                                   # ratchet
  if (newStop - Stop) ≥ min_stop_step:
      Stop = newStop

  if Low ≤ Stop:
      exit at policy_fill_price()
      State = EXIT
```

---

## 14) Validation Metrics & Reporting

* Trade-level: R multiple, MAE/MFE before exit, exit reason.
* Portfolio-level: CAGR, Sharpe/Sortino/Calmar, MaxDD, MAR, hit-rate, avgR, expectancy, tail loss stats (P95 loss).
* Robustness: Parameter heatmaps; sensitivity around best trial; WFA performance dispersion.

---

## 15) Notes & Recommendations

* Prefer **RMA/Wilder** ATR for stability; consider EMA for faster reaction in scalping.
* Use **HTF ATR** when trading low TF crypto to avoid micro-noise over-tightening.
* Default to **stop market** for reliability (stop-limit can miss fills).
* Ensure **no look-ahead**: all computations must use data available at the decision point.
* Keep **ratcheting monotonic** and symmetric for shorts.

---

Develop config json in config/optimizer/exit/ folder for this mixin.