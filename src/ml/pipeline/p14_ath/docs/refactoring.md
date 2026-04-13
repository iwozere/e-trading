# p14_ath Refactoring Plan: Local Swings + ATR × k

This document describes the agreed direction for refactoring the Sequential ATH & Drawdown pipeline so it uses **local** swing highs and lows (treating the first bar as the reference regime, not “global ATH”), with **ATR × k** as the primary scale for what counts as a reversal—not fixed percentage cuts.

---

## 1. Motivation

**Current behavior (see `ath_pipeline.py`):**

- Tracks a **running maximum** over the whole series. A “cycle” ends only when price **exceeds the prior global high**.
- Filters cycles with a **fixed −1%** minimum drawdown to reduce clutter.

**Limitations:**

- Peaks that are only **local** maxima (never a new all-time high) never start a new cycle.
- A **fixed %** threshold does not scale across tickers or volatility regimes the same way **ATR-based** rules do.

**Target behavior:**

- Treat the **start of the lookback window** as the initial reference (conceptually: “we begin at the first close”; subsequent structure is **local** swings).
- Detect **swing highs** and **swing lows** using a **minimum move** expressed as **k × ATR** (not `%`), so the same `k` is more comparable across symbols and time.
- Keep the same **output shape** where possible (one row per “peak → subsequent trough” leg, or an explicitly documented variant—see §4).

---

## 2. Look-ahead bias (accepted)

Many swing / zigzag definitions **confirm** a pivot only after price has moved against the prior extreme by the threshold (e.g. “this was a swing high once we have dropped by k × ATR”). For **historical charts and research-style equity curves**, using the **final** swing labels over the full window is acceptable for this exercise.

**Explicit stance:** Look-ahead in the sense of “knowing the full series when labeling pivots” is **not a problem** here. If a stricter causal mode is needed later, it can be a separate mode or pipeline.

---

## 3. ATR × k: definitions

**ATR (Average True Range):**

- Use a standard ATR on the OHLC series (Wilder-style or SMA of True Range—pick one implementation and document it in code).
- **Period** `atr_period` (e.g. 14) is a config parameter.
- For bar `t`, denote `ATR[t]` after sufficient warmup; rows before warmup can be excluded from swing detection or ATR can be forward-filled from first valid—decide in implementation and document.

**Threshold:**

- A move of size **k × ATR** (evaluated at a defined bar—see algorithm) is required to **confirm** a reversal and thus the previous extreme as a **swing pivot**.
- `k` is a **dimensionless multiplier** (e.g. 1.0–3.0); tune via config, optionally per run.

**Why not % for the core rule:**

- Percent drawdown from a peak mixes **price level** and **volatility** in a way that differs across assets. **ATR × k** ties the hurdle to **recent volatility**, aligning better with “meaningful local move” across names.

*Optional later:* keep a **secondary** `%` or **minimum calendar gap** filter for display only (noise reduction), but the **primary** gate should be **ATR × k**.

---

## 4. Swing detection (high-level algorithm)

The exact state machine should be implemented in a dedicated function (e.g. `detect_swings_atr(df, atr_period, k) -> pivots`) with unit tests. One standard pattern:

1. **Initialize** from the first bar: track a **candidate** swing high (or low) while price trends.
2. After a **candidate high**, require a decline of at least **k × ATR** (ATR at the confirmation bar or at the candidate bar—**must be fixed in code**) to **confirm** that high and start seeking a **swing low**, and vice versa.
3. Record **confirmed** pivot timestamps and prices for highs and lows in chronological order.
4. Pair **each swing high** with the **next swing low** (or the documented pairing rule) to produce rows analogous to today’s `ATH_Date` / `Max_Drop_Date` windows.

**Edge cases to specify in implementation:**

- Flat stretches, gaps, missing bars.
- Last segment: open swing without a confirming opposite pivot by series end—either drop, or emit a partial row (document choice).

---

## 5. Schema and naming

Today’s CSV columns are ATH-centric:

| Current column   | Intended meaning after refactor                                      |
| ---------------- | ------------------------------------------------------------------- |
| `ATH_Date`       | Rename conceptually to **swing high date** (e.g. `Swing_High_Date`) |
| `ATH_Price`      | **Swing high price** (`Swing_High_Price`)                           |
| `Max_Drop_Date`  | **Swing low date** after that high (`Swing_Low_Date`)               |
| `Max_Drop_Price` | **Swing low price** (`Swing_Low_Price`)                             |
| `Drop_Percent`   | Replace or supplement with **drop in ATR units** or **k-equivalent** metric, e.g. `Drop_ATR` = \((P_{low} - P_{high}) / ATR_{ref}\), plus optional retained `%` for readability |

**Backward compatibility:** Either keep old column names as aliases for one release (deprecated), or bump output filename / pipeline version and update consumers. Prefer **clear new names** in CSV + plot labels.

---

## 6. Configuration (`config.py`)

Add or replace parameters, for example:

| Parameter        | Role |
| ---------------- | ---- |
| `atr_period`     | Bars for ATR (e.g. 14). |
| `swing_k`        | Multiplier k in **k × ATR** for confirming reversals. |
| `atr_column_ref` | Which ATR series to use if multiple (single canonical implementation). |

Remove or demote **`-1%` hard filter** once ATR × k is the primary gate; if a small `%` filter remains, make it **optional** and off by default.

Keep existing **visualization** settings (`log_scale`, `initial_equity_usd`, etc.); update plot titles from “ATH” to “swing” wording.

---

## 7. Visualization and equity simulation

- **Price panel:** Mark **swing highs** / **swing lows** (same marker style as today unless cluttered).
- **Equity panel:** Reuse the same mechanical strategy—**sell at swing high price, buy at next swing low price**—driven off the new result table. Rename the legend from “ATH / trough” to something like **“Swing high / swing low strategy”**.
- **`_simulate_ath_dd_equity`:** Rename to a neutral name (e.g. `_simulate_swing_equity`) and parameterize column names once schema is stable.

---

## 8. Tests

- **Unit tests** for ATR computation (known small OHLC series).
- **Unit tests** for swing detection on synthetic series with hand-counted pivots for a given `k`.
- Update or replace `test_ath_logic.py` which currently encodes **global ATH** semantics and row counts that no longer apply after the refactor.

---

## 9. Documentation updates

- **`docs/pipeline-specification.md`:** Revise §3–§6 to describe **local swings + ATR × k**, not global ATH + greedy peak-trough only.
- **`README` or package docstring:** If present, one-line summary of the new behavior.

---

## 10. Implementation checklist

1. [ ] Add ATR helper (single implementation, documented).
2. [ ] Implement `detect_swings_atr` (or equivalent) with configurable `atr_period` and `swing_k`.
3. [ ] Replace core loop in `analyze_ticker` with swing-based rows; align CSV schema (new column names or versioned output).
4. [ ] Update `_plot_results` labels and titles; adjust equity simulation naming and column references.
5. [ ] Extend `ATHPipelineConfig` (or rename pipeline class if desired—optional).
6. [ ] Refresh tests; remove reliance on global ATH row counts.
7. [ ] Update `pipeline-specification.md` to match shipped behavior.

---

## 11. Out of scope (for this refactor doc)

- Live trading without look-ahead; causal pivot detection modes.
- Multi-timeframe swing alignment.
- Transaction costs, slippage, and dividend reinvestment in the equity line.

These can be separate follow-ups if needed.
