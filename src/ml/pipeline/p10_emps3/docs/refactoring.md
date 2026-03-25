# Pipeline p10_emps3 — Architectural Review & Refactoring Plan

_Generated: 2026-03-25 | Senior Software Architect Review_

---

## 1. Overview

`p10_emps3` is the **Accumulation Phase Detection** pipeline (EMPS3). It focuses on detecting the **Coiled Spring** pattern — stocks with **high volume hidden by low price volatility** — serving as a precursor signal before the explosive move identified by EMPS2.

Pipeline stages:
1. Universe download (reuses EMPS2's `NasdaqUniverseDownloader`)
2. Fundamental filtering (reuses EMPS2's `FundamentalFilter`)
3. TRF dark-pool volume correction
4. Accumulation Analysis (`AccumulationAnalyzer`) — scores stocks using Vol Z-Score, Absorption Ratio, Bollinger Band squeeze, 52w proximity
5. Sentiment filtering (reuses EMPS2's `SentimentFilter`)
6. Roll Memory analysis (Phase 1.5 early warning detection)
7. Alert dispatch

---

## 2. Architecture Strengths

- ✅ **Focused scope** — EMPS3 is much smaller than EMPS2, with a single, well-defined "Coiled Spring" signal.
- ✅ **AccumulationAnalyzer scoring** — The 3-signal composite score (Absorption Ratio, BB squeeze, resistance proximity) is well-structured.
- ✅ **Diagnostic output** — `08_absorption_diagnostics.csv` provides clear pass/fail reasons for every ticker examined.

---

## 3. Issues & Recommendations

---

### 🔴 Issue 1: p10 is Tightly Coupled to p06 via Direct Imports

**Problem**: EMPS3 imports directly from the p06 module:

```python
# emps3_pipeline.py
from src.ml.pipeline.shared.universe_downloader import NasdaqUniverseDownloader
from src.ml.pipeline.shared.fundamental_filter import FundamentalFilter
from src.ml.pipeline.shared.sentiment_filter import SentimentFilter

# accumulation_analyzer.py
from src.ml.pipeline.p06_emps2.trf_downloader import get_trf_correction_factor

# config.py
from src.ml.pipeline.p06_emps2.config import EMPS2UniverseConfig, SentimentFilterConfig
```

p10 **cannot run** without p06. If p06 is refactored or moved, p10 silently breaks. These are effectively shared components pretending to be private to one pipeline.

**Fix**: Extract shared components (`NasdaqUniverseDownloader`, `FundamentalFilter`, `SentimentFilter`, `TRFDownloader`) into a common module (e.g., `src/ml/pipeline/shared/`). Both p06 and p10 import from there.

**Effort**: 🟡 Half-day (move files and update imports)

---

### 🔴 Issue 2: `accumulation_analyzer.py` Hardcodes Its Own Results Path

**Problem**: `AccumulationAnalyzer.__init__` sets its own `self._results_dir` internally, unconditionally.

```python
self._results_dir = Path("results") / "p10_emps3" / target_date
self._results_dir.mkdir(parents=True, exist_ok=True)
```

This makes testing impossible without creating real filesystem directories, and creates a tight coupling between the component and the pipeline's output path.

**Fix**: Accept `results_dir: Path` as a constructor argument (like `FundamentalFilter` does).

**Effort**: 🟢 30 minutes

---

### 🔴 Issue 3: `_save_results` in `AccumulationAnalyzer` Silently Loses Data

**Problem**: `_save_results()` filters the passed data to only rows with `prebreakout_score > 70` before writing to disk. This means all tickers that passed the accumulation filters but scored below 70 are **silently dropped** from `07_prebreakout_watchlist.csv`. Meanwhile, the calling code in `emps3_pipeline.py` reads this CSV back and treats it as the final list.

```python
def _save_results(self, results_data: list):
    ...
    filtered_df = df[df.get('prebreakout_score', 0) > 70]  # Silently drops rows
    filtered_df.to_csv(out, index=False)                     # Full df never saved
```

**Fix**: Save the full results (all passed tickers) to the CSV. Apply score-based filtering in the pipeline orchestrator where it can be logged explicitly.

**Effort**: 🟢 30 minutes

---

### 🔴 Issue 4: `sentiment_filter._results_dir` Mutated After Construction

**Problem**: After initializing `self.sentiment_filter`, the pipeline immediately replaces the output directory by directly mutating a private attribute:

```python
self.sentiment_filter = SentimentFilter(self.config.sentiment_config, target_date=target_date)
self.sentiment_filter._results_dir = self._results_dir  # ❌ Breaks encapsulation
```

**Fix**: Add a `results_dir: Optional[Path]` argument to `SentimentFilter.__init__`. Pass it at construction time.

**Effort**: 🟢 30 minutes

---

### 🔴 Issue 5: `asyncio.run()` Called Twice in `EMPS3AlertSender.send_phase1_5_alert`

**Problem**: `asyncio.run()` is called once for Telegram and once for Email in two sequential calls. If either is already inside an async context, this will crash. Additionally, the `NotificationServiceClient` is closed inside the `finally` block of `_send_async_notification`, meaning the **second call fails** because the client is already closed.

```python
asyncio.run(self._send_async_notification(title, message, attachments, ['telegram']))
asyncio.run(self._send_async_notification(title, message, attachments, ['email']))
# Second call: client already closed after first call's finally block runs
```

**Fix**: Merge into a single async call passing `channels=['telegram', 'email']`. Remove `await self.client.close()` from the `finally` block, and handle client lifecycle at the sender level.

**Effort**: 🟢 30 minutes

---

### 🟡 Issue 6: `detect_phase1_5_candidates` Does Not Validate Trend Direction

**Problem**: The trend validation in `detect_phase1_5_candidates` only uses linear regression slope direction (`atr_slope < 0 and vol_slope > 0`). It doesn't check:
- The **magnitude** of change (a slope of -0.000001 is "negative" but meaningless)
- The **statistical significance** of the trend

This means borderline/noisy tickers will appear as Phase 1.5 candidates.

**Fix**: Add a minimum slope magnitude check (e.g., `abs(atr_slope) > 0.001`) or an R² filter. Document the chosen threshold in config.

**Effort**: 🟡 1–2 hours

---

### 🟡 Issue 7: `TRF Surge Logic` is Mocked / Incomplete

**Problem**: In `AccumulationAnalyzer._check_accumulation`:

```python
trf_surge = False  # Needs actual rolling 3-day window of trf factor check.
# For now, we will flag it if ar is very high.
```

`trf_surge` is computed but **never used** in the scoring logic or pass/fail conditions. It is a dead variable.

**Fix**: Either implement the intent (rolling 3-day TRF comparison) or remove the variable entirely to avoid confusion. Document as a known incomplete feature in the docstring.

**Effort**: 🟡 1–3 hours (depends on if you want to implement or just remove)

---

### 🟡 Issue 8: `EMPS3RollingMemoryConfig` Has Duplicate Fields with `RollingMemoryConfig` (p06)

**Problem**: `EMPS3RollingMemoryConfig` (p10) redeclares many of the same fields as `RollingMemoryConfig` (p06):
- `enabled`, `lookback_days`, `send_alerts`, `save_rolling_candidates`

There is no shared base class, so threshold logic is maintained in two places.

**Fix**: Create a `BaseRollingMemoryConfig` dataclass in the shared module. Both p06 and p10 configs inherit from it.

**Effort**: 🟡 1 hour

---

## 4. Summary & Execution Order

| Priority | # | Issue | Effort |
|---|---|---|---|
| 🔴 | 5 | `asyncio.run()` double-call closes client on first call | 🟢 30 min |
| 🔴 | 2 | `AccumulationAnalyzer` hardcodes own results path | 🟢 30 min |
| 🔴 | 3 | `_save_results` silently drops tickers with score ≤ 70 | 🟢 30 min |
| 🔴 | 4 | `sentiment_filter._results_dir` mutated after construction | 🟢 30 min |
| 🟡 | 7 | `trf_surge` is dead code | 🟡 30 min–3h |
| 🟡 | 6 | Trend slope lacks magnitude filtering | 🟡 1–2h |
| 🟡 | 8 | Duplicate rolling memory config fields | 🟡 1h |
| 🔴 | 1 | Hard cross-pipeline coupling via direct p06 imports | 🟡 half-day |

**Recommended start**: 5, 2, 3, 4 — all under 30 min, zero logic risk. Tackle Issue 1 as a dedicated refactoring sprint after quick wins are done.

---

## 5. p06 ↔ p10 Cross-Pipeline Dependency Map

```
p10_emps3/
├── emps3_pipeline.py  → imports from p06_emps2: NasdaqUniverseDownloader, FundamentalFilter, SentimentFilter
├── config.py          → imports from p06_emps2: EMPS2UniverseConfig, SentimentFilterConfig
└── accumulation_analyzer.py → imports from p06_emps2: get_trf_correction_factor
```

All of these should live in `src/ml/pipeline/shared/`.
