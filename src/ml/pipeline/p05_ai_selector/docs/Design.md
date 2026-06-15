# Design — P05 AI Selector

## Purpose

Produce a daily ranked shortlist of the top 5 equities and crypto instruments with complete, actionable position management guides. Unlike static scoring pipelines, the LLM layer synthesises heterogeneous signals into coherent narratives with specific price levels rather than generic recommendations.

## Architecture — 4-Stage Funnel

```
Input: ~3,020 symbols (Russell 3000 + top-20 crypto)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — LIQUIDITY & MOMENTUM PRE-FILTER               │
│  3,020 → ~200 candidates                                 │
│  Hard filters: price > $2, avg daily vol > $5M (equity)  │
│  Soft score: SMA crossover, RSI, volume trend            │
│  Cache: DATA_CACHE_DIR/p05/stage1/{date}.csv.gz          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — DETERMINISTIC SIGNAL SCORING                  │
│  ~200 → top-25 candidates                                │
│  Fundamentals: P/E, margins, debt, growth, dividend      │
│  Institutional flow: P18 output CSVs (score boosts)      │
│  Contextual: earnings calendar proximity flag            │
│  Cache: DATA_CACHE_DIR/p05/stage2/{date}.csv.gz          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — LLM SYNTHESIS (Claude Sonnet 4.6)             │
│  25 → top-5 picks                                        │
│  Single API call, all 25 data packets in one prompt      │
│  Structured output via tool_use (forced)                 │
│  Per-pick: rank, confidence, bias, thesis, exit advice   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — OUTPUT GENERATION                             │
│  results/p05_ai_selector/{date}/  (always written)       │
│  Telegram (top 3 + condensed exit advice)                │
│  Email (top 5 + full exit advice)                        │
│  Notifications: P18 signals ≥ 1 OR confidence ≥ 9       │
└──────────────────────────────────────────────────────────┘
```

## Component Design

### `Russell3000Downloader` (`src/data/downloader/`)
Shared infrastructure used by both P05 and P15 weekly bundle. FMP → static CSV fallback. 90-day TTL. Writes to `DATA_CACHE_DIR/universe/russell3000.csv.gz`.

### `UniverseLoader` (`stages/universe_loader.py`)
Combines Russell 3000 + 20 crypto tickers into a deduplicated sorted list. No filtering at this stage.

### `P18Reader` (`signals/p18_reader.py`)
Reads P18 output CSVs with date-fallback (searches for most recent run ≤ as_of_date). Returns sets of tickers for each P18 signal type.

### `EarningsCalendar` (`signals/earnings_calendar.py`)
FMP earnings calendar with 24-hour monthly cache. Returns earnings-within-7-days dict. Fails silently (returns `{}`) on any API error.

### `Stage1Prefilter` (`stages/stage1_prefilter.py`)
Idempotent (caches daily output). DataManager handles OHLCV caching and provider failover. Crypto uses raw volume threshold instead of USD volume.

### `Stage2Scorer` (`stages/stage2_scorer.py`)
Fetches fundamentals via DataManager (multi-provider, cached). Computes sector medians across the candidate pool for relative PE comparison. Missing fundamentals → 0 pts, ticker stays in funnel.

### `Stage3LLMSynthesizer` (`stages/stage3_llm_synthesizer.py`)
Single Anthropic API call with forced tool_use. Validates response structure; raises on malformed output (caller handles). Detects `confidence >= 9` to override notification logic.

### `Stage4Output` (`stages/stage4_output.py`)
Pure output writer + formatter. `should_notify()` implements the dual OR-trigger. Telegram messages are hard-capped at 3800 chars. No side effects on construction.

## Data Flow

```
Russell3000Downloader → UniverseLoader
                                      ↓
              P18Reader → Stage2Scorer ← EarningsCalendar
DataManager → Stage1Prefilter ↗
                              ↓
                       Stage2Scorer
                              ↓
                  Stage3LLMSynthesizer (Anthropic API)
                              ↓
                       Stage4Output → results/ dir
                              ↓
                         P05Pipeline → scheduler result dict
```

## Design Decisions

1. **Single LLM call**: All 25 candidates in one prompt. Avoids per-ticker API latency; costs ~$0.05/run. Forced `tool_use` guarantees structured output.

2. **Deterministic scoring first, LLM narrative second**: Separation of concerns — quantitative ranking is reproducible and auditable; LLM adds qualitative synthesis and exit strategy.

3. **Dual-trigger notifications**: OR logic between P18 signals and LLM confidence threshold. P18 trigger ensures we notify when institutional flow is active even if LLM confidence is modest.

4. **Idempotent stage caches**: Both Stage 1 and Stage 2 write daily `.csv.gz` files and reload on re-run. The `force_refresh` flag bypasses caches for debugging/backfill.

5. **Silent fundamental failures**: Missing fundamentals score 0 pts but the ticker remains in the funnel. The LLM is told `"available": false` and should discount the fundamental section for that ticker.

6. **P18 path without `_tracker`**: The actual P18 output directory is `results/p18_institutional_flow/` (not `_tracker`). `P18Reader` uses this path.
