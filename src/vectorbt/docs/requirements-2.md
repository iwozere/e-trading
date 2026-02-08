Alright, putting on the **grumpy senior Python architect / quant infra hat** 🎩
Overall: this is a **strong, coherent design**. You’re already thinking like someone who’s been burned by research/production drift and leverage fantasy PnL. What follows is not a rewrite — it’s a **hardening + scalability + correctness pass**.

I’ll go layer by layer, then give a **revised architecture checklist** you can actually implement.

---

# 1. High-Level Verdict

**What’s good (keep it):**

* Clear **research vs production** boundary
* SQLite + Optuna WAL is the right choice
* Vectorbt for research, Backtrader for live — *acceptable if parity is enforced*
* Futures + leverage modeled explicitly (rarely done correctly)
* Cross-margin awareness 👏

**Main risks I see:**

1. **SQLite + `n_jobs=-1` + WAL ≠ free lunch**
2. **Vectorbt liquidation ≠ Binance liquidation**
3. **Multi-asset cross-margin math is underspecified**
4. **Objective function still allows pathological strategies**
5. **Promotion pipeline lacks governance & reproducibility**
6. **Funding-rate modeling is too hand-wavy right now**

Let’s fix those.

---

# 2. Folder Structure Improvements

### Current

```
e-trading/
└── src/vectorbt/
    ├── db/
    ├── reports/
```

### Proposed (minimal but critical change)

```
e-trading/
├── src/
│   ├── research/
│   │   ├── vectorbt/
│   │   │   ├── signals/
│   │   │   ├── objective/
│   │   │   ├── wfo/
│   │   │   ├── db/
│   │   │   └── reports/
│   ├── shared/
│   │   ├── indicators/      # TA-Lib wrappers
│   │   ├── schemas/         # JSON / Pydantic
│   │   └── risk/
│   └── production/
│       ├── backtrader/
│       └── promote_strategy.py
```

**Why this matters**

* `shared/` becomes your **single source of truth**
* Indicators, risk math, schemas live *outside* Vectorbt/Backtrader
* Promotion script never imports research-only code

This structure saves you from the “just one import” disaster later.

---

# 3. SQLite + Optuna Concurrency Reality Check

### What you wrote

> Use WAL mode to handle high-concurrency writes (`n_jobs=-1`)

### Reality

* WAL helps
* SQLite **still serializes writers**
* Optuna with many short trials → write amplification
* On Raspberry Pi, this *will* bite you

### Recommendations

#### 3.1 Explicit SQLite tuning

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA busy_timeout=5000;
```

Apply this **once** on DB creation.

#### 3.2 Cap parallelism explicitly

Instead of `n_jobs=-1`:

```python
n_jobs = min(os.cpu_count() - 1, 6)
```

SQLite scales poorly beyond ~6 writers.
Your speedup curve will flatten anyway.

#### 3.3 Separate trial metadata from heavy metrics

* Store **scalar metrics** in Optuna DB
* Dump large objects (trades, equity curves) as:

  ```
  reports/{study_id}/{trial_id}.json
  ```

This keeps SQLite lean and fast.

---

# 4. Signals & Indicator Factory (Important Refinement)

### Current idea

> `Signals` class using `IndicatorFactory.from_talib()`

Good — but incomplete.

### Stronger pattern

Create **thin TA-Lib adapters**:

```python
class RSI:
    @staticmethod
    def compute(close, window):
        return talib.RSI(close, timeperiod=window)
```

Then:

```python
RSI_IF = vbt.IndicatorFactory(
    input_names=['close'],
    param_names=['window'],
    output_names=['rsi']
).from_apply_func(RSI.compute)
```

**Why this matters**

* Same function can be imported by:

  * Vectorbt
  * Backtrader
  * Unit tests
* No Vectorbt-only magic hidden inside indicators

---

# 5. Multi-Asset Data Model (Subtle Bug Prevention)

### Requirement

> Multi-index DataFrame with all 4 tickers

Correct — but define this **strictly**:

```text
index: DatetimeIndex
columns: MultiIndex
  level 0 → symbol (BTCUSDT, ETHUSDT…)
  level 1 → field (open, high, low, close, volume)
```

**Rule:**
👉 *Never mix assets on rows.*

Vectorbt broadcasting assumes column-wise independence unless told otherwise.

---

# 6. Walk-Forward Optimization (WFO) – Missing Guardrails

You use:

* `rolling_split`
* IS / OOS

Good, but add these **three constraints**:

### 6.1 Minimum trades filter

Discard trials with:

```python
pf.trades.count() < N_MIN_TRADES
```

Otherwise Calmar gets gamed by low-frequency luck.

---

### 6.2 Stability score (huge upgrade)

Instead of only best OOS score:

```text
FinalScore =
    median(OOS_scores)
    - std(OOS_scores)
```

This kills fragile parameter sets.

---

### 6.3 IS–OOS degradation penalty

```text
penalty = max(0, IS_score - OOS_score)
```

This prevents overfit monsters.

---

# 7. Futures & Liquidation — Critical Corrections

### 7.1 Vectorbt liquidation ≠ Binance liquidation

Vectorbt liquidation is **simplified equity ≤ 0**.

Binance:

* Maintenance margin tiers
* Per-symbol requirements
* Cross-margin interactions

### Recommendation

Implement a **custom liquidation proxy**:

```python
if pf.drawdowns.max() > MAX_ALLOWED_DD_BY_LEVERAGE[leverage]:
    return FAIL_SCORE
```

Use conservative thresholds (e.g. 60–70% DD at 10x).

This is more realistic than trusting `pf.is_liquidated()` alone.

---

### 7.2 Funding rates (must be explicit)

Don’t hide funding in “custom fee magic”.

Do this instead:

* Funding rate series per symbol
* Apply periodic cash adjustment:

```python
cash_flows = position_value * funding_rate
```

Vectorbt supports `cash_flow` arrays — use them.

This keeps math auditable.

---

# 8. Objective Function – Stronger Version

Your formula:
[
Score = \frac{Return}{DD \times Leverage}
]

Better, but still exploitable.

### Proposed production-grade score

```text
Score =
    CAGR
    × Calmar
    × WinRate
    / (Leverage²)
```

Why:

* Quadratic leverage penalty kills YOLO configs
* WinRate stabilizes trade quality
* CAGR keeps focus on compounding, not spikes

---

# 9. Cross vs Isolated Margin Simulation

Good call using `cash_sharing=True`.

**But add explicit controls:**

* `max_position_size`
* `max_concurrent_positions`
* `per-symbol leverage caps`

Otherwise cross-margin lets BTC save XRP stupidity.

---

# 10. Promotion Pipeline (Underrated Risk)

### Current

> Promote best trial to PostgreSQL

### Missing:

* Reproducibility
* Auditability
* Rollback

### Required fields in `active_configs`

Add:

* `study_id`
* `trial_id`
* `data_hash`
* `vectorbt_version`
* `talib_version`
* `promotion_timestamp`

This is **non-negotiable** if real money is involved.

---

# 11. JSON & Report Export (One Improvement)

Along with trades:

* Export **equity curve**
* Export **margin usage**
* Export **max adverse excursion (MAE)**

These become your human sanity checks before deployment.

---

# 12. Final Architect’s Checklist

If you implement **only these 10 things**, you’re ahead of 95% of retail quant stacks:

1. Shared TA-Lib indicator layer
2. SQLite write limits + tuning
3. Conservative liquidation proxy
4. Explicit funding cash flows
5. Stability-aware WFO scoring
6. Minimum trades filter
7. Leverage² penalty
8. Cross-margin caps
9. Promotion metadata & hashes
10. Version-locked research environment

---

## Bottom line

This is **not a toy system**.
With these changes, it becomes:

* Research-safe
* Capital-preserving
* Auditable
* Scalable to more symbols and regimes
