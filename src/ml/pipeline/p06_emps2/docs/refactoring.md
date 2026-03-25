# Pipeline p06_emps2 — Architectural Review & Refactoring Plan

_Generated: 2026-03-25 | Senior Software Architect Review_

---

## 1. Overview

`p06_emps2` is the **Enhanced Explosive Move Pre-Screener v2**. It is a large, multi-stage pipeline that screens ~8,000 NASDAQ tickers down to a high-conviction watchlist using:

1. Universe download (NASDAQ FTP)
2. Fundamental filtering (cap, float, volume)
3. TRF dark-pool volume correction
4. Volatility filtering (ATR, Z-score, accumulation signals)
5. Rolling memory analysis (phase 1/2 detection across 14 days)
6. UOA analysis
7. Sentiment data collection
8. Alert dispatch (Telegram/Email)

---

## 2. Architecture Strengths

- ✅ **Typed config hierarchy** — `EMPS2PipelineConfig` with named sub-configs is clean and extensible.
- ✅ **Per-scan logging** — `RotatingFileHandler` attached to the root logger on each run is a solid observability pattern.
- ✅ **Checkpoint/Resume** — `FundamentalFilter` has crash recovery via CSV checkpointing.
- ✅ **Factory config presets** — `create_default()`, `create_aggressive()`, `create_conservative()` reduce caller confusion.

---

## 3. Issues & Recommendations

---

### 🔴 Issue 1: Duplicate `DataManager` Instantiation in `__init__`

**Problem**: `DataManager()` is called twice in `EMPS2Pipeline.__init__` (lines 89 and 98). The second silently shadows the first.

```python
self.data_manager = DataManager()  # line 89
self.fundamental_filter = FundamentalFilter(self.data_manager, ...)

self.data_manager = DataManager()  # line 98 — duplicate!
self.volatility_filter = VolatilityFilter(self.data_manager, ...)
```

**Fix**: Remove the second instantiation. Both filters should share the same object.

**Effort**: 🟢 5 minutes

---

### 🔴 Issue 2: Duplicate import of `FinnhubDataDownloader`

**Problem**: `FinnhubDataDownloader` is imported twice on lines 22–23 in `emps2_pipeline.py` and is never used in the file body. The actual downloading is done through `DataManager`.

```python
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader  # line 22
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader  # line 23 — duplicate
```

**Fix**: Remove both orphaned imports.

**Effort**: 🟢 2 minutes

---

### 🔴 Issue 3: `FundamentalFilter` Receives `EMPS2FilterConfig` — Tied to p06

**Problem**: `FundamentalFilter.__init__` is typed to accept `EMPS2FilterConfig` (p06's config type). However, in p10, it is called with `EMPS3FilterConfig`, relying on duck typing (see `emps3_pipeline.py`, line 61 comment: *"Will rely on duck typing"*).

This means if `EMPS3FilterConfig` is refactored to rename a field, `FundamentalFilter` will break silently.

**Fix**: Extract a `FundamentalFilterConfig` base dataclass with only the fields needed by `FundamentalFilter` (`min_price`, `min_avg_volume`, `min_market_cap`, `max_market_cap`, `max_float`). Both `EMPS2FilterConfig` and `EMPS3FilterConfig` should inherit from it.

**Effort**: 🟡 1–2 hours

---

### 🔴 Issue 4: `self._output_files` Accessed Before Assignment in `_stage8_send_alerts`

**Problem**: `_stage8_send_alerts` accesses `self._output_files` (line 547), but this attribute is only set inside `_stage4_rolling_memory_analysis` IF historical data is found. If Stage 4 returns early, `self._output_files` doesn't exist and Stage 8 will throw `AttributeError`.

**Fix**: Initialize `self._output_files = {}` in `__init__`.

**Effort**: 🟢 5 minutes

---

### 🟡 Issue 5: `_stage3_volatility_filter` Reads Its Own Output From Disk

**Problem**: `VolatilityFilter.apply_filters()` saves results to a CSV. Then `_stage3_volatility_filter` immediately re-reads the same CSV to get the `DataFrame`. This is a broken pipeline contract — the filter should return the DataFrame directly.

```python
filtered_tickers = self.volatility_filter.apply_filters(tickers)  # Saves to disk...
volatility_csv = self._results_dir / "05_volatility_filtered.csv"
filtered_df = pd.read_csv(volatility_csv)                          # ...then reads it back
```

**Fix**: Update `VolatilityFilter.apply_filters()` to return the full `DataFrame` instead of `List[str]`. Match the contract of `FundamentalFilter`.

**Effort**: 🟡 1–2 hours (needs review of `volatility_filter.py`)

---

### 🟡 Issue 6: Stages Are Numbered Out of Order in Source Code

**Problem**: The stage methods are defined in this order in the source file:
`_stage1`, `_stage2`, `_stage2b`, `_stage3`, `_stage4`, **`_stage7`**, **`_stage8`**, **`_stage6`**, `_stage5`, `_generate_summary`

Stage 6 is defined after Stage 7, and Stage 5 after Stage 8. This makes the file hard to navigate.

**Fix**: Reorder method definitions to match the logical execution order.

**Effort**: 🟢 30 minutes (pure refactoring, no logic changes)

---

### 🟡 Issue 7: Root Logger Pollution via `RotatingFileHandler`

**Problem**: `_setup_pipeline_logging` attaches a `RotatingFileHandler` to the **root logger**, not just the pipeline logger. This means every other module's logs (trading, API, etc.) will also land in `pipeline.log` while the pipeline is running. In a multi-threaded environment, this is a shared mutable state problem.

**Fix**: Instead of attaching to the root logger, use a module-specific logger hierarchy (e.g., `logging.getLogger("emps2")`). Pass this logger through to sub-components.

**Effort**: 🟡 2–3 hours

---

### 🟢 Issue 8: Stage 7 Creates a `SentimentFilter` with a Local Config Instead of Using `self.sentiment_filter`

**Problem**: `_stage7_sentiment_data_collection` creates a **new** `SentimentFilter` with permissive settings, completely ignoring the already-initialized `self.sentiment_filter`. This wastes resources and breaks the DI pattern.

```python
# Ignores self.sentiment_filter entirely
sentiment_filter = SentimentFilter(config)  # Newly created object
sentiment_df = sentiment_filter.apply_filters(tickers)
```

**Fix**: Call `self.sentiment_filter.apply_filters()` with `apply_filters(tickers, config_override)` or redesign `apply_filters()` to accept optional threshold overrides.

**Effort**: 🟡 1–2 hours

---

## 4. Summary & Execution Order

| Priority | # | Issue | Effort |
|---|---|---|---|
| 🔴 | 2 | Duplicate import of `FinnhubDataDownloader` | 🟢 2 min |
| 🔴 | 4 | `_output_files` uninitialized in `__init__` | 🟢 5 min |
| 🔴 | 1 | Duplicate `DataManager` instantiation | 🟢 5 min |
| 🟡 | 6 | Stage methods out of order | 🟢 30 min |
| 🔴 | 3 | `FundamentalFilter` tied to `EMPS2FilterConfig` (duck typing in p10) | 🟡 1–2h |
| 🟡 | 5 | Volatility filter reads own disk output | 🟡 1–2h |
| 🟢 | 8 | Stage 7 ignores `self.sentiment_filter` | 🟡 1–2h |
| 🟡 | 7 | Root logger pollution | 🟡 2–3h |

**Recommended start**: 2 → 4 → 1 (all under 10 min, zero risk).
