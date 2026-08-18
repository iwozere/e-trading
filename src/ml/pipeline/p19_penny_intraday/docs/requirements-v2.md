# Requirements — P19 v2 (Structural Integrity rework)

Supersedes `Requirements.md` for the fields that change under
[`pipeline-specification-v2.md`](pipeline-specification-v2.md). Everything not
listed here (IBKR feed, Finnhub/Polygon cross-check, sentiment adapters,
notification delivery) is unchanged — see the v1 `Requirements.md`.

Scope of this document: **Phase 1.5** (schema v2 + Layer 0 core subset + outcome
labels — spec §16) and, as of 2026-08-18, **Phase 3** (N5/N6/N15/N16/P8/P9/P11,
intraday filings poll, sentiment attach — spec §16, design-v2.md §9). Phase 2/4
requirements (Disposition Engine, alerting, Optuna calibration) will be written
when those phases are scheduled; see `tasks-v2.md` §Roadmap for their
placeholder scope.

---

## Python Dependencies

No new packages. Everything Phase 1.5 needs is already a project dependency:

- `pandas`, `requests` *(existing)*
- `yfinance` *(existing — used by `yahoo_data_downloader.py`, `market_agent.py`,
  `finra_*_downloader.py`; Layer 0 adds a direct `.splits` read, same call
  pattern as `market_agent.py`, no new downloader class)*
- `sqlite3` *(stdlib — `ShadowStore` stays SQLite; see design-v2.md §Storage)*

**Explicitly not added:** `duckdb`. The v1 spec (§12) proposed it citing a
"planned P15 DuckDB layer" that was never built (P15's GDELT/DuckDB design was
superseded before implementation — no DuckDB usage exists anywhere in this
codebase). Introducing a second storage engine for a dataset that currently has
**zero accumulated rows** is not justified; revisit only if calibration query
performance on the (extended) SQLite store actually proves limiting.

---

## External Dependencies (cross-module)

| Dependency | Used for | Status |
|---|---|---|
| `src.data.downloader.edgar_downloader.EdgarDownloader` | `load_company_facts` (XBRL), `download_submissions`/`get_recent_filings` (8-K, S-3, 424B*, S-1), `resolve_tickers_to_ciks` | **existing, reused as-is** |
| `src.data.downloader.edgar_downloader.EdgarDownloader` | Form 4 transactions, all codes (P/S/A/M/F) | **existing daily cache widened**, no new network calls — `download_form4_filings` already runs universe-wide daily (design-v2.md §3.1); P19 reads the cached `.csv.gz` directly |
| `src.data.downloader.edgar_downloader.EdgarDownloader` | CIK-scoped EFTS phrase search (N5/N6/N16) | **built (Phase 3)**: `efts_text_search` — design-v2.md §9.1 |
| `src.data.downloader.edgar_downloader.EdgarDownloader` | Multi-CIK EFTS filing lookup (intraday filings poll) | **built (Phase 3, new)**: `efts_filings_search` — design-v2.md §9.1 |
| `src.data.downloader.edgar_downloader.EdgarDownloader` | EX-23.1 auditor-consent extraction (N16) | **built (Phase 3, new)**: `get_auditor_name` — design-v2.md §9.2 |
| `src.data.downloader.edgar_downloader.EdgarDownloader` | 13D/G filings, all form types (P8) | **existing daily cache reused**, no new network calls — `download_13dg_filings` already runs daily via P18's scan; P19 reads the cached `.csv.gz` (design-v2.md §9.5) |
| `yfinance` (`yf.Ticker(t).splits`) | reverse-split history (N1/N2) | existing call |
| `yfinance` (`yf.Ticker(t).info`) | short interest / days-to-cover (P11) | **new call (Phase 3)**, same direct-call pattern as splits |
| `src.common.sentiments.collect_sentiment_async.collect_sentiment_batch_sync` | mention counts / sentiment score, context only (spec §10) | **new call (Phase 3)**, throttled by `sentiment_cache.py` — design-v2.md §9.8 |
| `src.ml.pipeline.p17_penny_stocks.agents.dilution_agent.DilutionAgent` | superseded reference implementation for shelf/ATM/reverse-split keyword detection — informs the new grading logic, not called directly | reference only |
| `src.ml.pipeline.p19_penny_intraday.shadow_store.ShadowStore` | schema v2 additive columns (`is_fpi` added in Phase 3) | **extended**, not replaced |
| `src.ml.pipeline.p19_penny_intraday.metrics` | momentum_score / momentum_tier (log-only, no alerting) | **extended** — see design-v2.md §4 |
| `src.notification.logger` | logging | existing |

**New intra-module code** (`p19_penny_intraday/structural/`, spec §18):
`profiler.py`, `xbrl_facts.py`, `grading.py`, `cache.py`; new models
`models/structural_profile.py`; new top-level modules `label_backfill.py`,
`filings_poll.py` (Phase 3), `sentiment_cache.py` (Phase 3).

---

## External Services

- **SEC EDGAR** (no key, fair-use `User-Agent` header, ≤10 req/s — already
  enforced by `EdgarDownloader`). Confirmed live and warm on the Pi
  (`R:\data-cache\edgar\`: 716 `companyfacts`, 436 `submissions`, an 80-day
  8-K index, a daily universe-wide Form 4 cache running since 2026-06-12).
  Phase 1.5 adds:
  - `companyfacts` reads — **must pass `force=True` on Layer 0's own weekly
    cadence**; the shared cache has no TTL, so file-exists is not
    freshness (design-v2.md §0.1).
  - `submissions` reads for offering-form detection (same caveat)
  - Form 4 — **no new fetch**, reads the already-scheduled daily cache with
    its parser widened to all codes (design-v2.md §3.1)
  - EFTS phrase search scoped to CIK (N5/N6/N16) — **built, Phase 3**
  - EFTS multi-CIK filing lookup, no phrase (intraday filings poll, spec §9)
    — **built, Phase 3**, real per-CIK-scoped calls during market hours, not
    a shared cache
  - 13D/G — **no new fetch**, reads P18's already-scheduled daily cache
    (Phase 3)
- **Yahoo Finance** (`yfinance`, unauthenticated) — split history + short
  interest / days-to-cover (P11, Phase 3).
- **Sentiment providers** (Reddit/StockTwits/etc. via
  `src.common.sentiments`, Phase 3) — same providers every other pipeline
  already uses, throttled to ~hourly regardless of the 15-minute poll
  cadence (design-v2.md §9.8); degrades to empty per-provider on missing
  credentials, does not block a poll.

No new API keys required.

---

## System Requirements

- Structural profiling runs **pre-market only** (spec §13: 06:00–08:00 ET),
  outside the intraday IBKR rate budget — a separate scheduler job, not part
  of the `run-once` shadow poll.
- Per-ticker structural profile cache, keyed by ticker + CIK, persisted
  alongside the shadow store (`results/p19_penny_intraday/structural_cache/`).
  Survives across days; weekly full-refresh + daily delta check (spec §4.0).
- `ShadowStore` schema v2 must be **additive and backward-compatible**:
  confirmed 1,741 existing rows across 3 trading days / 60 tickers on the Pi
  (`R:\results\p19_penny_intraday\shadow.sqlite`) must read back with `NULL`
  in every new column, not require a table rebuild.
- (Phase 3) Two new small on-disk files alongside the shadow store, both
  under `results/p19_penny_intraday/`: `filings_events.sqlite` (its own
  table, kept separate from the shared universe-wide 8-K index — see
  design-v2.md §9.4 for why) and `sentiment_cache.json` (single-file batch
  cache, not per-ticker).

## Performance / Rate Requirements

- Layer 0 at N≤100 watchlist names, weekly full refresh: ~3 EDGAR calls/ticker
  (companyfacts + submissions + one EFTS query) ≈ 300 requests/week against the
  10 req/s SEC limit — negligible (spec §13).
- **Phase 3 addition**: N5/N6/N16's EFTS calls add up to ~5 more requests/ticker
  on the same weekly cadence (one per (date, form) window — never a comma-list
  of forms, since EFTS doesn't OR those, design-v2.md §9.1) — still negligible
  against the 10 req/s limit.
- **Intraday filings poll (Phase 3, spec §9)**: 4 EFTS queries per run (one per
  watched form type), each scoped to the **whole watchlist's CIKs at once**
  (`efts_filings_search`, chunked at 100), not one query per ticker. At the
  scheduled `*/30 13-21 * * 1-5` cadence that's ~4 requests every 30 minutes
  during market hours — negligible.
- **Sentiment (Phase 3, spec §10)**: throttled to at most once per
  `SentimentCache` TTL window (default 60 min) regardless of the 15-minute
  shadow-poll cadence — unthrottled per-poll calls would make P19 the
  heaviest consumer of these provider rate limits in the codebase for no
  signal benefit (every other pipeline calls this at most daily).
- Daily delta check (new-filing detection) re-profiles only names with a new
  submissions entry since the last cached refresh — O(1) submissions read per
  name, not a full re-profile.
- Coverage must be tracked **per signal**, not assumed — every `StructuralProfile`
  field is nullable and `coverage` is the fraction resolved (spec §4.0,
  StructuralSignals.md §1 rule 3).

## Correctness Requirements (non-negotiable, StructuralSignals.md §7)

These gate Phase 1.5 sign-off — get them wrong and the signals invert:

1. **Share-count series must be split-adjusted before CAGR (N3/N4/P3).**
   Unadjusted, a reverse split makes the most-diluted names score cleanest.
2. **Quarterly cash-flow figures must be de-cumulated from XBRL's YTD reporting
   (N10/N11)** before any quarter-over-quarter logic, or Q4 looks like a spike
   on every filer.
3. **Unknown must grade C, never A** (N17) — a signal that could not be
   resolved is not evidence of cleanliness.
4. **Form 4 codes A/M/F/G excluded from insider-buy counts (P1/P2)** — only
   code `P`, non-derivative, plan-checkbox excluded.
5. **`insider_conviction` renormalises over resolved signals only** — absence
   of Form 4 data (FPIs) must score null, not a low positive value.
6. **(Phase 3) Text-extraction heuristics must return unresolved, not a
   guess, on ambiguous input** — N16's auditor-name extraction returns `None`
   when no signature line is found, rather than scanning the whole document
   and risking a false match on unrelated boilerplate. Caught by a test
   during review; see tasks-v2.md's Phase 3 section for the bug it fixed.
7. **(Phase 3) EFTS `forms` is never a comma-list** — unlike `ciks`, which
   does OR a comma-list, `forms` is an exact match and a comma-list
   paradoxically returns only amendments. Every N5/N6/N16/filings-poll caller
   queries one form type per request (design-v2.md §9.1).

## Security

- Unchanged from v1: keys only from `config/donotshare/.env`, never logged.
- EFTS phrase-search results (filing excerpts) may be surfaced in alert copy —
  no PII beyond what SEC already discloses publicly; no new handling concern.
- (Phase 3) Extracted auditor names, 13D/G filer names, and sentiment mention
  counts are all already-public SEC filings / public social-media aggregate
  counts — same no-PII posture as the rest of Layer 0.

## Testing Requirements

- Unit tests for every N/P signal detector against synthetic XBRL/Form4/EFTS
  fixtures (the split-adjustment and de-cumulation fixes get dedicated
  assertions per StructuralSignals.md §7 items 1–2 — required, not optional).
- `ShadowStore` v2 migration test: append a v1-shaped row, confirm new columns
  read back as `NULL`, confirm a v2-shaped row round-trips.
- Coverage regression test: a synthetic FPI-shaped profile (no Form 4, no 8-K)
  must resolve to grade C via N17, never A/B.
- (Phase 3) `filings_poll.py` tests must confirm it never calls
  `EdgarDownloader.download_8k_filings` (the shared-cache collision risk,
  design-v2.md §9.4) and that a re-poll of the same filing does not
  duplicate it. `SentimentCache` tests must confirm the TTL actually
  suppresses a second fetch within the window. P11's conditional branch
  (grade A/B feeds `insider_conviction`, C/D bumps `dilution_urgency`
  instead) needs its own regression test — the two code paths are easy to
  accidentally conflate.
