# P05 AI Selector — Pipeline Specification

## 1. Overview

**Purpose:** Daily AI-powered screener that combines quantitative signals with Claude LLM synthesis to produce a ranked shortlist of the top 5 equities and crypto instruments worth watching or trading — each with a complete position management guide (entry conditions, hold conditions, thesis-breakers, profit targets).

**Key differentiator:** Unlike static scoring pipelines, the LLM reasoning layer synthesises multiple heterogeneous signals (technical, fundamental, institutional flow) into coherent, actionable narratives. The exit advice section makes each pick immediately tradeable — providing specific price levels, catalyst checkpoints, and thesis-breakers rather than a generic price target.

**Cadence:** Runs weekdays at 10:00 UTC (after P18 at 07:00 UTC, after P15 daily at 13:00 UTC the prior day — all signal inputs are available).

**Notification trigger (dual OR logic):**
- Primary: P18 reported `high_score_count >= 1` today
- Override: LLM assigned `confidence >= 9` (out of 10) for any pick, regardless of P18

---

## 2. Universe

| Source | ~Count | Provider |
|--------|--------|----------|
| Russell 3000 equities | ~3,000 | `Russell3000Downloader` (shared, see §3.1) |
| Top-20 crypto | 20 | Fixed list (BTC, ETH, BNB, SOL, ADA, AVAX, DOT, LINK, MATIC, UNI, XRP, LTC, ATOM, NEAR, ICP, FIL, APT, ARB, OP, DOGE) |
| **Total** | **~3,020** | — |

**Universe maintenance:**
- `Russell3000Downloader` caches to `DATA_CACHE_DIR/universe/russell3000.csv.gz`; TTL 90 days (quarterly).
- Crypto list is static in `config.py`; no cache needed.
- Penny stocks (price < $2), OTC / Pink Sheet issues, and delisted symbols excluded at Stage 1 via FMP profile flags.

---

## 3. Shared Infrastructure: Russell 3000 Downloader

### 3.1 Location & Purpose

**File:** `src/data/downloader/russell3000_downloader.py`

A standalone shared downloader following the same pattern as `fred_downloader.py` and `edgar_downloader.py`. Both P05 and the P15 weekly bundle use it — neither embeds its own copy of the universe list.

### 3.2 Cache Layout

```
DATA_CACHE_DIR/
  universe/
    russell3000.csv.gz    ← cached constituent list; TTL 90 days
                             columns: ticker, name, sector, industry, exchange
    russell3000_meta.json ← last_updated, source_used, row_count
```

### 3.3 Class Interface

```python
class Russell3000Downloader(BaseDataDownloader):
    """
    Downloads and caches the Russell 3000 index constituents.

    Sources tried in order:
      1. FMP /v3/russell_constituent (free tier — available if key present)
      2. Bundled static CSV at src/data/downloader/data/russell3000_static.csv
         (committed to repo; update manually each quarter from Slickcharts or
          FTSE Russell quarterly rebalance announcement)

    Cache: DATA_CACHE_DIR/universe/russell3000.csv.gz
    TTL:   90 days; force=True bypasses the TTL check.
    """

    def load(self, force: bool = False) -> pd.DataFrame:
        """Return cached constituent list, refreshing from source if stale."""

    def is_stale(self) -> bool:
        """Return True when cache is absent or older than 90 days."""

    def _fetch_from_fmp(self) -> Optional[pd.DataFrame]:
        """Call FMP /v3/russell_constituent; return None on any error."""

    def _load_static_fallback(self) -> pd.DataFrame:
        """Load bundled static CSV; raises FileNotFoundError if missing."""
```

### 3.4 Bundled Static Fallback

`src/data/downloader/data/russell3000_static.csv` — committed to the repo. Columns: `ticker, name, sector, industry, exchange`. Updated manually every quarter (March / June / September / December FTSE Russell rebalance).

### 3.5 P15 Weekly Integration

Add one job to `p15_weekly.py` (after fred_quarterly, before fred_combined):

```python
results["russell3000_refresh"] = _run_job(
    "russell3000_refresh",
    lambda: _job_russell3000_refresh()
)
```

```python
def _job_russell3000_refresh() -> Optional[Dict[str, Any]]:
    dl = Russell3000Downloader()
    if not dl.is_stale():
        _logger.info("russell3000: cache fresh — skipping")
        return {"skipped": True}
    df = dl.load(force=True)
    return {"rows": len(df), "source": dl.last_source_used}
```

This runs every Friday but only actually re-downloads when the cache is > 90 days old. Harmless on non-rebalance weeks.

---

## 4. Architecture: 4-Stage Funnel

```
Input: ~3,020 symbols (Russell 3000 + top-20 crypto)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — LIQUIDITY & MOMENTUM PRE-FILTER               │
│  3,020 → ~200 candidates                                 │
│  60-day OHLCV via DataManager (cached 1d bars)           │
│  Hard filters: price > $2, avg daily vol > $5M (equity)  │
│  Soft score: SMA crossover, RSI, volume trend            │
│                                                          │
│  NOTE — cold start: first ever run fetches OHLCV for all │
│  ~3,020 symbols via DataManager; estimated ~20–40 min.   │
│  Subsequent daily runs are fast (gap detection fills     │
│  only the latest 1–2 days from cache).                   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — DETERMINISTIC SIGNAL SCORING                  │
│  ~200 → top-25 candidates                                │
│  Fundamentals: P/E, margins, debt (FundamentalsCache /   │
│                FMP / Yahoo fallback; missing → 0 pts)    │
│  Crypto: fundamental section skipped (0 pts)             │
│  Institutional flow: P18 output CSVs (today's date)      │
│  Contextual: earnings calendar proximity flag            │
│  Composite deterministic score → rank → top 25           │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — LLM SYNTHESIS (Claude Sonnet 4.6)             │
│  25 → top-5 picks                                        │
│  Single API call, all 25 data packets in one prompt      │
│  Structured output via tool_use                          │
│  Per-pick: rank, confidence (1–10), bias, thesis,        │
│            risk factors, time horizon                    │
│  Per-pick exit advice: entry conditions, hold            │
│            conditions, thesis-breakers, profit targets   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — OUTPUT GENERATION                             │
│  results/p05_ai_selector/{date}/  (always written)       │
│  Telegram (top 3 + condensed exit advice)                │
│  Email (top 5 + full exit advice)                        │
│  Notifications fire when: p18_signals_count >= 1         │
│                        OR notification_override = 1      │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Infrastructure Reuse

| Capability | File path | How P05 uses it |
|-----------|-----------|----------------|
| Russell 3000 universe | `src/data/downloader/russell3000_downloader.py` | Stage 1: constituent list; shared with P15 weekly |
| OHLCV data + caching | `src/data/data_manager.py` | Stage 1: 60-day 1d bars for all symbols |
| Fundamentals | `src/data/cache/fundamentals_cache.py` | Stage 2: P/E, margins, debt; missing → 0 pts |
| Fundamentals (primary) | `src/data/downloader/fmp_data_downloader.py` | Stage 2: earnings calendar + gap-fill |
| Fundamentals (fallback) | `src/data/downloader/yahoo_data_downloader.py` | Stage 2: small-cap gap-fill |
| P18 signals | `results/p18_institutional_flow_tracker/{date}/` | Stage 2: score boost for flagged tickers |
| Notifications | `src/notification/` | Stage 4: email + Telegram delivery |
| Scheduler | `src/scheduler/scheduler_service.py` | Daily 10:00 UTC DATA_PROCESSING job |
| Logging | `src/notification/logger.py` | All modules via `setup_logger(__name__)` |
| Claude API | `anthropic` SDK | Stage 3: LLM synthesis + exit advice |

---

## 6. Signal Catalog

### 6.1 Technical Signals (Stage 1 & 2)

| Signal | Computation | Points |
|--------|-------------|--------|
| SMA crossover bullish | price > SMA20 > SMA50 | 15 |
| SMA crossover bearish | price < SMA20 < SMA50 (short candidate) | 10 |
| RSI oversold | RSI14 < 30 | 12 |
| RSI overbought | RSI14 > 70 | 8 |
| Volume surge | today_vol > 1.5× 20-day avg_vol | 15 |
| Price momentum | 5d return, top quartile vs. full universe | 10 |
| ATR compression | 5d ATR < 0.7× 20d ATR (consolidation setup) | 5 |
| 52-week high proximity | price within 5% of 52w high | 8 |
| 52-week low proximity | price within 5% of 52w low (reversal) | 8 |

### 6.2 Fundamental Signals (Stage 2, equities only)

Crypto assets skip this section (score 0). Tickers with no fundamentals data silently score 0 and remain in the funnel.

| Signal | Computation | Points |
|--------|-------------|--------|
| Value | P/E < sector median | 10 |
| Quality | net profit margin > 15% | 10 |
| Safety | debt-to-equity < 1.5 | 5 |
| Growth | revenue YoY > 10% | 10 |
| Dividend | dividend yield > 0 | 3 |

### 6.3 Institutional Flow Signals (Stage 2, from P18 output CSVs)

| Signal | P18 source | Points |
|--------|-----------|--------|
| P18 composite score ≥ 60 | today's P18 top_picks | 40 |
| Consensus exit (3+ institutions) | P18 consensus CSV | 25 |
| Form 4 insider buy | P18 form4 CSV `transaction_type = B` | 15 |
| 13D/G new stake accumulation | P18 13dg CSV | 15 |

### 6.4 Contextual Signals (Stage 2)

| Signal | Source | Effect |
|--------|--------|--------|
| Earnings within 7 days | FMP earnings calendar | Flag only — passed to LLM as risk context; no points |

Options flow: deferred to Phase 5 (no Tradier API access currently).

**Maximum deterministic score:** ~176 points  
**Stage 2 → Stage 3 cutoff:** top 25 by deterministic score; ties broken by volume surge ratio.

---

## 7. LLM Integration (Claude Sonnet 4.6)

### 7.1 Input Data Packet (per candidate)

All 25 candidates are sent in **one API call**.

```json
{
  "ticker": "AAPL",
  "name": "Apple Inc.",
  "sector": "Technology",
  "asset_type": "equity",
  "price": 195.20,
  "market_cap_b": 3.0,
  "technicals": {
    "rsi_14": 42.3,
    "sma20_above_sma50": true,
    "volume_surge_ratio": 2.1,
    "momentum_5d_pct": 3.4,
    "atr_compression": false,
    "pct_from_52w_high": -8.2,
    "pct_from_52w_low": 34.1
  },
  "fundamentals": {
    "pe_ratio": 28.5,
    "profit_margin_pct": 26.0,
    "debt_to_equity": 1.2,
    "revenue_yoy_pct": 8.1,
    "dividend_yield_pct": 0.5,
    "available": true
  },
  "institutional_flow": {
    "p18_score": 75,
    "signals_active": ["consensus_exit", "volume_spike"],
    "institution_count": 4,
    "form4_insider_buy": false
  },
  "contextual": {
    "earnings_in_days": 5,
    "earnings_date": "2026-06-19"
  },
  "deterministic_score": 138
}
```

### 7.2 System Prompt

```
You are a quantitative equity and crypto analyst. You receive a ranked list of stock
and crypto candidates with their quantitative signal summaries. Your tasks are:

1. Select the top 5 most actionable picks for the next 1–12 months (time horizon
   varies by setup quality), ranked by conviction.

2. For each pick, write a concise thesis and a complete position management guide:
   - Specific price levels for adding to the position
   - Catalyst-based hold conditions (what must remain true to stay in)
   - Thesis-breakers: concrete, falsifiable events that trigger an immediate exit
   - Profit-taking levels with partial trim percentages and price targets
   - A brief time-horizon note explaining why patience is the edge for this setup

3. Write a brief market context paragraph.

Guidelines:
- Favour setups with signal confluence; be sceptical of single-signal stories.
- For crypto: ignore traditional fundamentals; focus on momentum, volume, and
  institutional signals only.
- Earnings within 7 days = binary risk event; note it explicitly in risk_factors.
- Price levels must be specific (e.g. "$44–46") not vague (e.g. "near support").
- Profit-taking levels should reflect realistic upside from the current price.
- thesis_breakers must be concrete falsifiable events, not market platitudes.

Return ONLY the structured JSON defined in the tool schema.
```

### 7.3 Structured Output (tool_use schema)

```json
{
  "picks": [
    {
      "rank": 1,
      "ticker": "AAPL",
      "confidence": 8,
      "bias": "long",
      "thesis": "Concise 2–3 sentence explanation of why this setup is compelling.",
      "risk_factors": [
        "Earnings in 5 days — binary risk event",
        "High P/E vs. sector median"
      ],
      "time_horizon": "3–6 months",
      "exit_strategy": {
        "add_conditions": [
          "Post-earnings dip toward $185–188 if results are in-line or better",
          "Broad market pullback of 3%+ provides a re-entry window",
          "A break above $200 on volume > 1.5× average confirms momentum"
        ],
        "hold_conditions": [
          "Services revenue growth remains above 12% YoY",
          "Gross margin holds above 43%",
          "Management reaffirms buyback programme"
        ],
        "thesis_breakers": [
          "Q3 revenue misses guidance by more than 3%",
          "Services segment growth drops below 8% YoY",
          "Major regulatory action limits App Store economics",
          "CEO departure or sudden management restructuring"
        ],
        "profit_targets": [
          {
            "price_level": 230,
            "action": "Trim 25% of position to lock in gains",
            "note": "~18% from current price; covers downside risk on remaining position"
          },
          {
            "price_level": 270,
            "action": "Trim another 25%; hold remainder toward 12-month target",
            "note": "Half the original position now de-risked"
          }
        ],
        "time_horizon_note": "This is not a quick trade — the Services re-rating thesis plays out over 2–3 quarters. Selling at 10–15% would mean abandoning a position where the structural tailwind is intact. The patience is the edge here."
      }
    }
  ],
  "market_context": "One-paragraph assessment of broad conditions relevant to these picks.",
  "notification_override": false
}
```

`notification_override` is set to `true` by the LLM when any pick has `confidence >= 9`, enabling the dual-trigger notification logic without re-scanning the picks array in the pipeline.

### 7.4 Model & Cost

| Parameter | Value |
|-----------|-------|
| Model | `claude-sonnet-4-6` |
| Estimated input tokens / run | ~10,000 |
| Estimated output tokens / run | ~4,000 (5 picks × full exit_strategy) |
| Cost per run | ~$0.05 |
| Monthly cost (22 trading days) | ~$1.10 |

---

## 8. Output & Notifications

### 8.1 Results Directory

```
results/p05_ai_selector/
  {YYYY-MM-DD}/
    top_picks.csv        ← top 5: rank, ticker, confidence, bias, thesis, time_horizon
    full_ranking.csv     ← top 25: all deterministic signal scores + total
    report.md            ← narrative report with full exit strategies
    metadata.json        ← run metadata: timing, counts, trigger reason, tokens used
```

### 8.2 Telegram Message (top 3, condensed)

Sent when `p18_signals_count >= 1` OR `notification_override = true`:

```
P05 AI Selector — {YYYY-MM-DD}
Trigger: {trigger_reason}

1. ${TICKER} [{bias}] conf {N}/10 | {time_horizon}
   "{thesis}"
   Exit if: {breaker_1}; {breaker_2}
   First target: ${price_level} → {action_short}

2. ${TICKER} [{bias}] conf {N}/10 | {time_horizon}
   "{thesis}"
   Exit if: {breaker_1}; {breaker_2}
   First target: ${price_level} → {action_short}

3. ${TICKER} [{bias}] conf {N}/10 | {time_horizon}
   "{thesis}"
   Exit if: {breaker_1}; {breaker_2}
   First target: ${price_level} → {action_short}

Full report (incl. #4–5 and complete exit guides):
results/p05_ai_selector/{date}/report.md
```

### 8.3 Email Report (HTML, top 5)

Full HTML email:

1. **Header:** date, trigger reason, market context paragraph
2. **Summary table:** rank, ticker, bias, confidence, thesis, time horizon, earnings flag
3. **Per-pick sections (×5):**
   - Thesis
   - Signal breakdown (scored signals that fired, P18 flags)
   - Risk factors
   - **Position management guide:**
     - *Entry conditions* — when/where to add to the position
     - *Hold conditions* — what must remain true
     - *Thesis-breakers* — exit immediately if any of these occur
     - *Profit targets* — trim schedule with price levels and % of position
     - *Time horizon note* — why patience is the edge for this specific setup
4. **Disclaimer** footer

---

## 9. Scheduler Registration

```sql
-- Weekdays at 10:00 UTC (after P18 at 07:00 UTC)
INSERT INTO job_schedules (user_id, name, job_type, cron, task_params, enabled)
VALUES (
  1,
  'P05 AI Selector Daily',
  'data_processing',
  '0 10 * * 1-5',
  '{
    "script_path": "src/ml/pipeline/p05_ai_selector/run_p05_scan.py",
    "script_args": [],
    "notification_rules": {
      "condition_mode": "any",
      "conditions": [
        {
          "check_field": "p18_signals_count",
          "operator": ">=",
          "threshold": 1,
          "channels": ["telegram", "email"]
        },
        {
          "check_field": "notification_override",
          "operator": "==",
          "threshold": 1,
          "channels": ["telegram", "email"]
        }
      ]
    }
  }',
  true
);
```

**Result dict keys** (`__SCHEDULER_RESULT__:<json>`):

| Key | Type | Description |
|-----|------|-------------|
| `success` | bool | Run completed without error |
| `pick_count` | int | Picks written (normally 5) |
| `p18_signals_count` | int | P18 `high_score_count` from today (0 if P18 not run) |
| `notification_override` | int | 1 if any pick had confidence ≥ 9 |
| `trigger_reason` | str | Human-readable trigger description |
| `top_ticker` | str | Rank-1 ticker |
| `top_confidence` | int | LLM confidence for rank-1 pick (1–10) |
| `stage1_out` | int | Symbols passing Stage 1 |
| `stage2_out` | int | Symbols reaching Stage 3 (normally 25) |
| `llm_tokens_used` | int | Total tokens consumed in Stage 3 |
| `results_dir` | str | Absolute path to output directory |
| `timestamp` | str | ISO 8601 run timestamp |
| `user_id` | int | Injected by scheduler |

---

## 10. File Structure

```
src/data/downloader/
  russell3000_downloader.py          ← NEW: shared constituent downloader (used by P05 + P15)
  data/
    russell3000_static.csv           ← NEW: bundled quarterly fallback; update each quarter

src/ml/pipeline/p05_ai_selector/
  run_p05_scan.py                    ← scheduler entry point
  pipeline.py                        ← 4-stage orchestrator
  config.py                          ← thresholds, weights, model ID, output paths
  stages/
    universe_loader.py               ← calls Russell3000Downloader; adds crypto list
    stage1_prefilter.py              ← liquidity + momentum filter (3,020 → ~200)
    stage2_scorer.py                 ← deterministic composite score (200 → 25)
    stage3_llm_synthesizer.py        ← Claude API call: prompt + tool_use output parser
    stage4_output.py                 ← CSV/MD/JSON writer + email + Telegram formatter
  signals/
    technical.py                     ← SMA, RSI, volume surge, ATR, momentum helpers
    fundamental.py                   ← P/E, margins, debt, growth scoring (0 pts if missing)
    p18_reader.py                    ← loads P18 output CSVs for today's run date
    earnings_calendar.py             ← FMP earnings calendar: fetch, cache, proximity flag
    options_flow.py                  ← STUB — Phase 5 placeholder (no-op returns empty)
  tests/
    test_russell3000_downloader.py   ← in src/data/downloader/tests/ (shared infra)
    test_universe_loader.py
    test_stage1_prefilter.py
    test_stage2_scorer.py
    test_stage3_llm.py
    test_stage4_output.py
    test_technical_signals.py
    test_fundamental_signals.py
    test_p18_reader.py
    test_earnings_calendar.py
  docs/
    pipeline-specification.md
    Requirements.md
    Design.md
    Tasks.md
  README.md
```

---

## 11. Cache Layout

```
DATA_CACHE_DIR/
  universe/
    russell3000.csv.gz          ← Russell 3000 constituents; TTL 90 days
                                   columns: ticker, name, sector, industry, exchange
    russell3000_meta.json       ← last_updated, source_used ("fmp" or "static"), row_count
  p05/
    stage1/{YYYY-MM-DD}.csv.gz  ← Stage 1 output (~200 symbols with momentum scores)
    stage2/{YYYY-MM-DD}.csv.gz  ← Stage 2 output (top-25 with full signal breakdown)
    earnings/{YYYY-MM}.csv.gz   ← FMP earnings calendar for the month; TTL 24h
```

---

## 12. Build Phases

```
Phase 0 — Shared Infrastructure (prerequisite, before P05 Phase 1)
  ├── src/data/downloader/russell3000_downloader.py
  │     Russell3000Downloader class, FMP → static CSV fallback, 90-day TTL
  ├── src/data/downloader/data/russell3000_static.csv
  │     Download from Slickcharts; commit to repo
  ├── src/data/downloader/tests/test_russell3000_downloader.py
  └── Add russell3000_refresh job to p15_weekly.py

Phase 1 — Universe & Signal Data (Week 1)
  ├── stages/universe_loader.py: call Russell3000Downloader + append crypto list
  ├── signals/p18_reader.py: locate today's P18 output CSVs, load scored tickers
  ├── signals/earnings_calendar.py: FMP earnings calendar fetch + monthly cache
  └── Tests for all three

Phase 2 — Signal Computing (Week 2)
  ├── signals/technical.py: SMA, RSI, volume surge, ATR, momentum, 52w helpers
  ├── signals/fundamental.py: scoring from FundamentalsCache / FMP / Yahoo; missing → 0
  ├── stages/stage1_prefilter.py: hard filters + momentum score (uses DataManager)
  ├── stages/stage2_scorer.py: composite deterministic score + P18 boost + ranking
  └── Tests for all four

Phase 3 — LLM Integration & Output (Week 3)
  ├── stages/stage3_llm_synthesizer.py: Claude API call + tool_use output parser
  ├── stages/stage4_output.py: CSV/MD/JSON + email HTML + Telegram formatter
  ├── pipeline.py: 4-stage orchestrator + dual-trigger notification logic
  └── run_p05_scan.py: scheduler entry point

Phase 4 — Integration & Docs (Week 4)
  ├── Scheduler SQL registration
  ├── End-to-end test run against a historical date (dry-run, no notifications)
  ├── Email HTML template review
  └── README.md, Requirements.md, Design.md, Tasks.md

Phase 5 — Enhancements (Deferred)
  ├── signals/options_flow.py: Tradier integration (if API access secured)
  ├── Simple ML pre-filter: e.g. XGBoost or logistic regression on 2-year history;
  │   label = binary top-quartile 5-day forward return; replaces or supplements
  │   the deterministic Stage 2 score for equities only
  ├── Sector-relative scoring: compare vs. sector peers, not absolute thresholds
  ├── Short-candidate mode: explicit bearish pipeline variant
  └── Backtesting: track P05 top-5 vs. forward returns for ongoing calibration
```

---

## 13. Open Questions

### Resolved

| # | Question | Decision |
|---|----------|----------|
| 1 | Russell 3000 source | Shared `Russell3000Downloader`: FMP free tier → bundled static CSV fallback; added to P15 weekly bundle |
| 2 | Crypto exact list | Confirmed (20 coins listed in §2) |
| 3 | P18 trigger coupling | Dual OR trigger: `p18_signals_count >= 1` OR `notification_override = 1` (confidence ≥ 9) |
| 4 | Options flow / Tradier | Not available; `options_flow.py` is a Phase 5 no-op stub |
| 5 | Missing fundamentals | Score 0 pts silently; ticker stays in funnel |
| 6 | Crypto fundamentals | Fundamental section skipped for crypto; technicals + P18 flow only |
| 7 | Stage 1 cold-start | P15 daily covers only 57 macro tickers — does not pre-warm Russell 3000 cache. First P05 run: ~20–40 min one-time warm-up via DataManager. Subsequent runs: fast (gap fill only). No separate warm-up job needed. |
| 8 | Weekdays only | `0 10 * * 1-5` — yes, weekdays only |
| 9 | ML pre-filter | Deferred to Phase 5. V1 uses deterministic scoring only. If added later: XGBoost or logistic regression, label = top-quartile 5-day forward return (binary). |
| 10 | OR condition in scheduler | `condition_mode: "any"` — confirm implementation supports this before SQL registration |
