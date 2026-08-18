# Design — P19 v2 (Structural Integrity rework)

> The authoritative *thesis* is [`pipeline-specification-v2.md`](pipeline-specification-v2.md)
> and [`StructuralSignals.md`](StructuralSignals.md). This file is the
> **implementation design**: what gets built, in what modules, calling what
> existing code, and why — for **Phase 1.5** (spec §16) and, as of
> 2026-08-18, **Phase 3** (§9 below). Phase 2 (Disposition Engine + alerting)
> and Phase 4 (Optuna calibration) get a lighter roadmap sketch at the end;
> they are deliberately not designed in detail until Phase 1.5/3 shadow data
> exists to validate the thesis and fit thresholds (spec §15 Q1 is the gate).

## Decisions carried into this design (resolved 2026-08-18)

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Detail level this round | **Phase 1.5 fully designed; Phases 2–4 roadmap-only** | Matches spec §0.1's own urgency framing — the thing that's time-critical is schema v2 + Layer 0 + labels, not the alerting engine, which can't be tuned until this data exists anyway. |
| 2 | Shadow store backend | **Keep SQLite, extend `ShadowStore`** — do **not** migrate to DuckDB | Spec §12 justified DuckDB by pointing at "the planned P15 layer" — that layer was never built (P15's DuckDB design is superseded, per project memory). It would be new infra, not reuse. `ShadowStore` already works, is tested, and the dataset is currently empty. Revisit only if calibration queries prove slow. |
| 3 | Legacy shadow data | **Confirmed present, small.** `R:\results\p19_penny_intraday\shadow.sqlite` has **1,741 rows across 3 trading days** (2026-08-13, 2026-08-14, 2026-08-18 — not a continuous run since 06-28; the shadow-poll cron only started actually landing rows very recently, plausibly unblocked by the 2026-08-16 numba/Python-3.14 scheduler crash-loop fix, commit `30e13ee`), **60 distinct tickers**, `trigger_reason` empty on every row (Trigger Engine genuinely unbuilt, not just quiet). `watchlist.json` exists for every weekday back to 06-28, so the *watchlist* build has been running all along — only the *poll* step was the gap. The additive `ALTER TABLE` migration (§6) applies cleanly to this; no elaborate backfill needed at this volume. | Grounded in the actual file, not inferred from spec text. Small enough that the migration path can stay simple. |
| 4 | New Layer 0 network code | **Extend `EdgarDownloader`**, and prefer widening an **already-scheduled shared cache** over adding new per-ticker calls where one already exists (see §3 below — Form 4 is exactly this case) | Matches "reuse existing code" directive. `R:\data-cache\edgar\` is already warm: 716 cached `companyfacts`, 436 `submissions`, an 80-day 8-K index, and a **daily universe-wide Form 4 bulk cache running since 2026-06-12** (feeds P15/P18/P20) that P19 can read directly instead of issuing new EDGAR calls. |

---

## 0.1 What's already warm on the Pi (checked directly, 2026-08-18)

`R:\data-cache\edgar\` (= `DATA_CACHE_DIR/edgar/`, the shared cache
`EdgarDownloader` reads/writes) is not a cold start:

| Cache | Volume | Feeds |
|---|---|---|
| `companyfacts/*.json` | 716 CIKs | P17 `DilutionAgent`, P18, P20 |
| `submissions/CIK*.json` | 436 CIKs | P17, P18, P20 |
| `8k/index/{date}.csv.gz` | 80 daily files | P17 `CatalystAgent` |
| `13f/form4/{date}.csv.gz` | daily, 2026-06-12 → today | P18 `form4_monitor.py`, P20 `filings_ingest.py`, P05 (via P18) |
| `13f/13dg/{date}.csv.gz` | daily, same range | P20 `filings_ingest.py` |

**Caveat that matters for Layer 0's weekly-refresh requirement:**
`EdgarDownloader.download_company_facts`/`download_submissions` skip the
fetch **whenever the destination file exists**, with no TTL — freshness is
entirely up to whichever caller last passed `force=True`. Layer 0 cannot
assume a cached file is current just because it exists; `structural/cache.py`
owns its own weekly-refresh decision and must call the download methods with
an explicit `force=True` on that cadence, independent of what other pipelines
happen to have cached.

**Current watchlist already contains the spec's own worked example**: `GRSD`
(the Jersey-domiciled 6-K filer StructuralSignals.md §2 cites verbatim as the
FPI-coverage-gap case) is in `shadow_log`'s ticker list right now — a live,
not hypothetical, first test case for the coverage/grade-C-via-N17 behavior.

**Dormant bug found while checking this, worth flagging separately from
Phase 1.5 scope**: `EdgarDownloader._parse_form4_xml` hard-filters to sale
codes `{"S", "S-"}` only. P20 Kestrel's `filings_ingest.py::_process_form4`
already reads that same cache expecting **buy** codes `{"P", "A"}`
(`_FORM4_BUY_CODES`) and silently finds nothing to match, every day, since
the cache never contains them — `insider_buy_value_90d` signals have likely
never fired. Not a P19 v2 problem to fix under this doc, but §3.1 below fixes
the root cause as a byproduct of building P1/P2, which is worth calling out
to whoever owns P20.

---

## 1. Architecture — Phase 1.5 slice

```
  (pre-market, 06:00–08:00 ET — outside the intraday IBKR budget)
                                                                                             
  watchlist.json (from existing WatchlistBuilder, unchanged)                                
        │                                                                                    
        ▼                                                                                    
  ┌─────────────────────────────────────────────────────────────────┐                       
  │ structural/profiler.py — StructuralProfiler.refresh(watchlist)  │                       
  │                                                                  │                       
  │  for each ticker:                                                │                       
  │   1. cache.py: cached & fresh (weekly, or no new filing since    │                       
  │      last refresh)? → reuse cached StructuralProfile             │                       
  │   2. else fetch:                                                  │                       
  │      EdgarDownloader.load_company_facts(cik)      (existing)     │                       
  │      EdgarDownloader.get_recent_filings(cik, ...)  (existing)     │                       
  │      read cached edgar/13f/form4/{date}.csv.gz     (existing cache,│                       
  │        parser widened to all codes — §3.1, no new network call)   │                       
  │      yf.Ticker(ticker).splits                       (new call)    │
  │      [Tier 3 / Phase 3, NOT in this slice: EFTS phrase search      │
  │       for N5/N6, §3.2]                                             │                       
  │   3. xbrl_facts.py: extract + de-cumulate + split-adjust series   │                       
  │   4. grading.py: evaluate N1–N17 / P1–P11 (Tier 1+2 subset) →    │                       
  │      StructuralProfile(grade, dilution_urgency,                  │                       
  │        insider_conviction, coverage, disqualifiers, ...)         │                       
  │   5. cache.py: persist                                            │                       
  └───────────────────────────┬─────────────────────────────────────┘                       
                              ▼                                                              
              structural_cache/{ticker}.json  (per-ticker, survives across days)             
                              │
                              ▼ (joined into watchlist.json entries at build time)
                                                                                              
  (intraday, existing shadow poll cron — unchanged cadence)                                  
        │                                                                                    
        ▼                                                                                    
  ┌─────────────────────────────────────────────────────────────────┐                       
  │ shadow_loop.py (extended, not rewritten)                        │                       
  │  compute_signal() [metrics.py, extended]:                        │                       
  │    existing: pct_from_open, rvol_so_far, dollar_volume_so_far    │                       
  │    NEW: momentum_score, momentum_tier (T0–T3, log-only —          │                       
  │         no Disposition Engine, no alert, spec §4.3 gate logic     │                       
  │         reused as a pure classifier)                              │                       
  │    NEW: denormalise the cached StructuralProfile fields onto      │                       
  │         the IntradaySignal (point-in-time snapshot, spec §12.1)   │                       
  └───────────────────────────┬─────────────────────────────────────┘                       
                              ▼                                                              
              ShadowStore (SQLite, schema v2 — additive columns)                             
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                            ▼
  eod-backfill (extended)                    label_backfill.py (NEW, T+10 cron)
  existing O/H/L/C fill +                     ret_t1/t3/t5/t10, dilution_event_within_5d,
  NEW: high_time, close_retention,            reverse_split_within_180d
  mae_from_alert, mfe_from_alert
```

---

## 2. New model: `models/structural_profile.py`

Dataclass, field-for-field per spec §4.0 (`StructuralProfile`) — reproduced there
verbatim, not re-derived here. One addition not in the spec text: a
`disqualifier_severities: dict[str, str]` map (signal id → `"C"`/`"D"`) alongside
the existing `disqualifiers: list[str]` human-readable strings, because the
grading formula (§7.5) needs to distinguish "any D fired" from "any C fired"
and re-deriving that from free-text strings at grading time is fragile.

`to_dict()` mirrors `IntradaySignal.to_dict()`'s pattern (flat, JSON/SQLite-safe,
NaN-guarded per the existing `_safe_round`/`_safe_int` helpers in
`models/watchlist_entry.py` — reused, not reinvented).

---

## 3. `EdgarDownloader` extensions

Changes to the existing class in `src/data/downloader/edgar_downloader.py`,
following its existing rate-limit/cache/error-handling conventions (`_get`,
`_fetch`, `_efts_search`, `_write_json`). Not a new class. Only §3.2
(`efts_text_search`) is actually needed for **Phase 1.5** — it exists here
because it's the natural extension point, but its callers (N5/N6) are Tier 3
and out of scope until Phase 3 (§4). §3.1 is a parser fix with no new public
surface, needed now.

### 3.1 Form 4 — widen the existing daily bulk cache, no new per-ticker calls

**Revised after checking the live cache (§0.1): P19 needs zero new EDGAR calls
for Form 4.** `download_form4_filings` already runs **daily, universe-wide**
(driven by P15's daily bundle) and caches every Form 4 filed each day to
`edgar/13f/form4/{date}.csv.gz` — P19's watchlist tickers are already in that
population on every trading day the job runs. The only gap is that
`_parse_form4_xml` hard-filters to sale codes `{"S", "S-"}` before the row
ever reaches the cache.

**Change**: generalise `_parse_form4_xml` to return *all* transaction codes
plus `acquired_disposed_code`, `is_10b5_1_plan` (plan-adoption checkbox), and
`is_derivative` — additive columns, existing consumers reading named columns
are unaffected. Two call sites read the result:

- `download_form4_filings` (public signature, cache path, and **default
  behaviour unchanged** — still returns/caches sale-code rows) filters the
  generalised parse down to `{"S", "S-"}` before returning, so **P18's
  `form4_monitor.py` (which already defensively re-filters to `_SALE_CODES`
  itself) and P15's daily-bundle caller are unaffected.**
- **P20's `filings_ingest.py::_process_form4` starts working** — it already
  filters the same cache file for `{"P", "A"}` (`_FORM4_BUY_CODES`) expecting
  buy rows that were never actually present; this is a side-effect fix, not
  new P19 scope, but call it out to whoever owns P20 before shipping.

**P19's `grading.py`** reads `edgar/13f/form4/{date}.csv.gz` directly for the
trailing 90/30-day windows P1/P2/N14 need, filtered to the watchlist ticker +
`transaction_code == "P"` (P1/P2, excluding grants/exercises per
StructuralSignals.md's code-A/M/F exclusion — note this is *stricter* than
P20's `{"P", "A"}`, deliberately, per StructuralSignals.md P1's "code `P`
only" requirement) or `"S"` excluding `F`/`M`-paired and 10b5-1 rows (N14).
No `EdgarDownloader` method addition needed here beyond the parser widening —
this whole signal group becomes a **read of an already-cached file**, not a
network call.

**One thing this doesn't solve**: the cache is keyed by *filing date*, not
issuer, so reading a 90-day window means reading ~90 small `.csv.gz` files
per profiler run. Fine at N≤100 tickers/weekly cadence (cheap, local disk);
if it ever isn't, add a `structural/cache.py`-level rollup rather than
touching `EdgarDownloader` again.

### 3.2 `efts_text_search(cik, phrases, forms) -> list[dict]`

CIK-scoped EFTS full-text phrase search, for N5 (floating converts) / N6
(going-concern). Extends `_efts_search`'s params with `ciks` (comma-separated
10-digit) and `q` (phrase, quoted) — **verify against the live EFTS endpoint
before relying on it**; add the finding to
`src/data/downloader/tests/test_edgar_efts_schema.py` (which already exists
for exactly this kind of "confirm the real field/param shape" work) rather
than assuming the parameter contract.

Scoped to **latest annual + latest interim filing only** per
StructuralSignals.md's N5 failure-mode note — unscoped matching produces false
D grades on names that already retired a convert.

### 3.3 Reverse splits — no `EdgarDownloader` change

`yf.Ticker(ticker).splits` called directly from `structural/profiler.py`,
same pattern P17's `market_agent.py` already uses for direct yfinance calls.
No new downloader wrapper — it's a single cheap call, cached at the P19
profile-cache layer (weekly), not worth its own class.

---

## 4. `p19_penny_intraday/structural/` package

| Module | Responsibility |
|---|---|
| `profiler.py` | Orchestrates one ticker's refresh: cache check → fetch (via `EdgarDownloader` + yfinance) → delegate to `xbrl_facts` + `grading` → cache write. `StructuralProfiler.refresh_watchlist(entries) -> dict[ticker, StructuralProfile]` is the entry point called from `run_p19.py profile-structural`. |
| `xbrl_facts.py` | Pure functions over a `companyfacts` JSON blob: `shares_outstanding_series`, `split_adjust(series, splits)`, `decumulate_quarterly(series)`, `cash_and_burn`, `proceeds_from_issuance`. No network calls — takes already-loaded JSON, testable with static fixtures. **This is where the two StructuralSignals.md §7 traps (#1 split-adjustment, #2 de-cumulation) get fixed once, centrally**, rather than per-signal. |
| `grading.py` | Tier 1 + Tier 2 signal evaluators (N1,N2,N3,N4,N7,N8,N10,N11,N13,N14 / P1,P2,P3,P4,P5,P6,P7) + §5 baby-shelf arithmetic for N9 + `dilution_urgency` + `insider_conviction` + `grade` assignment (spec §7.5) + `coverage` accounting. Tier 3 signals (N5,N6,N9-above-$75M,N12,N15,N16,P8,P9,P10) are **out of scope for Phase 1.5** — evaluators return `None` (unresolved), which correctly depresses `coverage` rather than silently omitting the field. |
| `cache.py` | Per-ticker JSON cache (`structural_cache/{ticker}.json`): `is_fresh(ticker) -> bool` (weekly TTL, or a new filing landed since last refresh per the issuer's `submissions.json` — the "daily delta check", spec §4.0), `load`, `save`. |

**Deliberately not built in Phase 1.5** (per spec §16 item 2's own
"deterministic, high-coverage first" framing): N5, N6 (text-parse precision
unmeasured — open question 6), N9 above $75M float, N12/N16 (need external
datasets), P8/P9/P10. Their `StructuralProfile` fields stay `None` and count
against `coverage` — never silently assumed clean.

**FPI grade "U" (StructuralSignals.md §2)**: not built this round. FPIs
correctly fall to grade C via N17 (`coverage < 0.4`) under the existing
formula; the distinct `"U"` grade is deferred pending measuring actual FPI
share of the watchlist (open question 5 in the spec) — tracked in
`tasks-v2.md` roadmap. Shadow rows still carry enough to reconstruct it later
(coverage + disqualifier list), so this is a grading-formula change, not a
data-loss risk.

---

## 5. Momentum tier — pulled forward from Phase 2, log-only

Spec §16 item 5 requires storing "the simulated trigger point" on every poll,
even in shadow mode, so Phase 2's first live alerts have a historical
analogue. That needs `momentum_tier` (T0–T3) computed now — but **not** the
Disposition Engine, Alert Manager, dedup state, or Telegram delivery, all of
which stay Phase 2.

- `metrics.py` gets a new `classify_momentum_tier(signal, trigger_cfg) -> str`
  using the *existing* `P19TriggerConfig` thresholds (`move_trigger_pct`,
  `rvol_trigger`, `dollar_volume_floor`) — these already exist as config
  placeholders (spec: calibrate later, don't hand-tune now).
- Pure function, no state, no side effects — same shape as the existing
  `compute_signal()`. Called from `shadow_loop.py`'s existing per-poll loop.
- `IntradaySignal` gets `momentum_score: float` and `momentum_tier: str`
  fields (spec §11), replacing the unused `severity` field it currently
  carries (v1's flat severity — the v2 spec explicitly removes it, §11 note).

No Disposition column is written in Phase 1.5 — `structural_grade` and
`momentum_tier` are stored side by side in the same row (spec §12.1's
denormalisation requirement) so the Phase 2 matrix can be computed
**retroactively** in analysis, without needing the engine to exist yet.

---

## 6. `ShadowStore` schema v2 — additive, same class

Extend `_COLUMNS` in `shadow_store.py` (not a new store class). New columns,
all nullable:

```
momentum_score, momentum_tier,                       # replaces `severity`
fresh_dilution_filing,
structural_grade, dilution_urgency, insider_conviction,
runway_quarters, shelf_capacity_pct_mcap, share_count_cagr_8q,
days_since_last_offering, insider_buys_90d, distinct_insider_buyers_90d,
reverse_splits_24m, floating_convert_flag, going_concern_flag,
structural_coverage, disqualifiers,                   # JSON-encoded list
-- outcome labels (§12.2), filled by eod-backfill / label_backfill:
high_time, close_retention, mae_from_alert, mfe_from_alert,
ret_t1, ret_t3, ret_t5, ret_t10,
dilution_event_within_5d, reverse_split_within_180d
```

`severity` stays in the table (existing rows may have it) but is no longer
written; `momentum_score`/`momentum_tier` are its replacement per spec §11.

**Migration behaviour**: `_ensure_schema()` already uses `CREATE TABLE IF NOT
EXISTS` — for an *existing* table missing the new columns, add an `ALTER
TABLE ... ADD COLUMN` pass (SQLite supports additive `ALTER TABLE`) guarded
per-column (`PRAGMA table_info` check, skip if present) so it's idempotent and
safe to run against a table that already has some v2 columns from a partial
prior run. This only matters if Pi data actually exists — see decision #3;
write the migration regardless since it's cheap and makes the "unconfirmed"
answer safe either way.

### 6.1 Outcome labels — `eod-backfill` extension + new `label_backfill.py`

- `eod-backfill` (existing `ShadowLoop.eod_backfill`, extended): adds
  `high_time`, `close_retention`, `mae_from_alert`, `mfe_from_alert` —
  computable same-day from the OHLC fetch already happening, plus the stored
  `momentum_tier` crossing point as the "alert price" proxy (spec §12.2 note:
  use the would-have-alerted price in shadow mode).
- `label_backfill.py` (new, `run_p19.py label-backfill` mode, T+10 cron): fills
  `ret_t1/t3/t5/t10` (forward closes, via existing `DataManager`) and the two
  structural-decay labels via `EdgarDownloader.get_recent_filings` (offering
  forms in the 5-session window) and `yf.Ticker().splits` (180-day window) —
  both already-established calls, no new fetch logic.

---

## 7. CLI / scheduling

`run_p19.py` gains two subcommands, following the existing `common` parser
pattern (`--user-id`, `--date`):

```
python run_p19.py profile-structural [--date YYYY-MM-DD] [--force]
python run_p19.py label-backfill [--date YYYY-MM-DD]
```

New scheduler entries (pattern-matched to `insert_p19_schedules.sql`'s
existing three jobs — same idempotent `ON CONFLICT DO NOTHING` insert style):

| Job | Cadence | Depends on |
|---|---|---|
| P19 Structural Profile | `0 11 * * 1-5` (pre-market, before watchlist build) | none |
| P19 Label Backfill | `0 12 * * 1-5`, but only acts on dates ≥10 sessions old | none |

Structural profiling runs **before** watchlist build in spec §4's diagram, but
since Phase 1.5 caches per-ticker (not per-watchlist-day), running it *after*
watchlist build and profiling only that day's entries is equally correct and
avoids profiling names that never made the cut — this implementation runs it
second, diagram intent (structural data available before intraday polling
starts) is preserved either way.

---

## 8. Error handling (mirrors existing conventions)

- Structural profiling failure for one ticker (EDGAR 404, foreign-issuer XBRL
  gap, EFTS timeout) → that ticker gets `coverage` penalised, not a pipeline
  abort — same "never abort the loop" principle as `WatchlistBuilder`'s
  gappers-source degradation.
- Cache corruption/missing → treated as cache-miss, re-fetch, never crashes.
- `label_backfill` failures per-ticker are logged and skipped (a label that
  never fills is a `NULL` in calibration, not a crash) — same tolerance model
  as `eod_backfill`'s existing per-ticker fetch loop.

---

## 9. Phase 3 design (built 2026-08-18)

Scope per spec §16: N5, N6, N15, N16, P8, P9, P11, the intraday EFTS filings
poll (§9), and sentiment context attach (§10). Sequenced **before** Phase 2
(2026-08-18 decision, recorded in pipeline-specification-v2.md §19) — the
§8.2 escalation rule needs the intraday filings poll this phase builds, and
disposition thresholds are meant to be fit from shadow data that doesn't
exist until this phase's signals start logging.

### 9.1 EFTS extensions (`EdgarDownloader`)

Three new public methods, all built on the existing `_efts_search`:

- **`efts_text_search(cik, phrases, forms, start_dt, end_dt)`** — the method
  §3.2 originally speced for N5/N6, now also used by N16 (see 9.2). Verified
  live (2026-08-18) that EFTS `q` does an Elasticsearch `match_phrase` and
  `ciks` filters by CIK, but the response carries **no highlighted snippet** —
  only match metadata. That ruled out the originally-imagined "search returns
  the surrounding text" approach for N16 and drove the EX-23.1-exhibit design
  instead (9.2).
- **`efts_filings_search(ciks, forms, start_dt, end_dt)`** — no-phrase,
  multi-CIK filing lookup for the intraday poll (9.4). Verified live that
  `ciks` accepts a comma-list as an OR filter (Elasticsearch `terms`), unlike
  `forms` which is an **exact match, not an OR** — `"10-K,10-K/A"` paradoxically
  returns only the amendments (same quirk already documented on the existing
  8-K catalyst path). Every new caller here queries one form type at a time
  and unions results, never a comma-list.
- **`get_auditor_name(cik, start_dt, end_dt, forms)`** — N16, see 9.2.

### 9.2 N16 — auditor quality via the EX-23.1 consent exhibit

The originally-imagined approach (search the 10-K body for the audit opinion
paragraph) turned out to be the wrong shape once verified live: EFTS indexes
each filing's individual **exhibit documents** separately, and every 10-K
carries an **EX-23.1 "Consent of Independent Registered Public Accounting
Firm"** exhibit — a few sentences, filed by the auditor itself, boilerplate
enough to parse reliably. Verified against a real filing (DeltaSoft Corp, CIK
0002020919, accession 0001683168-26-005450): the exhibit's `/S/ <signer>`
signature line is immediately followed by the firm's name on its own line
(`BOLADALE LAWAL & CO`) and often the PCAOB registration number too.

`_extract_auditor_name` (edgar_downloader.py) splits the cleaned document text
on block boundaries, finds the `/S/` line, and returns the **first following
line** that carries a firm-designation marker (`LLP`, `LLC`, `P.C.`, `CPA(s)`,
`&`, `+`, `Chartered Accountants`) — matching the whole line rather than a
bounded regex capture, so unusual separators (`WithumSmith+Brown`) don't
truncate the name the way a lazy character-class capture would. Returns
`None` (unresolved, not a guess) if no consent exhibit is found or no line in
the post-signature window matches.

**Whitelist, not live PCAOB integration** (design decision, not spec text):
`P19StructuralConfig.auditor_whitelist` is a static, substring-matched list of
Big 4 + common PCAOB-registered microcap auditors. StructuralSignals.md's own
N16 section suggests exactly this ("a maintained whitelist ... needs periodic
review") as the practical alternative to standing up a PCAOB Form AP scraper —
a new external-data integration that wasn't judged worth building before any
data exists to show N16 matters. Revisit if calibration says otherwise.

### 9.3 N5 / N6 — text-scoped phrase search

Per StructuralSignals.md's recency-scoping requirement (unscoped matching
produces false D grades on names that retired a convert), both are scoped to
`efts_text_search`'s exact filing date, not a rolling window:

- **N5** (floating-rate/toxic convertible): the ticker's **latest annual AND
  latest interim** filing (two single-day EFTS queries, form type taken from
  whichever filing was actually found — not re-guessed across every candidate
  type). Fires at **grade C**, not D — `P19StructuralConfig.n5_severity`
  defaults to `"C"` per the spec's own explicit fallback (open question 6:
  precision unmeasured, needs a hand-labelled sample of ~50 filings before
  trusting D, which is a hard suppression).
- **N6** (going-concern): **latest annual only**. Fires at grade D
  (near-perfect precision per StructuralSignals.md).

### 9.4 Intraday filings poll (spec §9) — `filings_poll.py`

New module, **not** part of `structural/` (it's an intraday concern, Layer 0
is pre-market-only by decision #5). `FilingsPoll.run()`: resolve the day's
watchlist tickers to CIKs, then one `efts_filings_search` call per watched
form type (`424B5`, `S-1`, `S-3`, `8-K`) scoped to **all** watchlist CIKs at
once — not one call per ticker. 8-K hits are filtered client-side to items
3.01/3.02 (EFTS returns the full `items` list per hit already).

**Deliberately does not call `EdgarDownloader.download_8k_filings`** even
though that method already does almost this — it writes to the **shared,
universe-wide** daily 8-K index cache (`edgar/8k/index/{date}.csv.gz`) that
P17's CatalystAgent reads once daily expecting a complete end-of-day
snapshot. Calling it intraday with `force=True` would overwrite that shared
file with a partial same-day snapshot and, because the file would then
already exist, suppress P17's own next-day refresh — silently starving P17 of
anything filed after P19's last intraday poll. `filings_poll.py` keeps its
own separate SQLite table (`filings_events.sqlite`, dedup'd on
`(date, ticker, accession, item)`) instead.

**Log-only** — hits are recorded for awareness and future calibration; there
is no Alert Manager yet (Phase 2) to escalate into. `events_for_date()` is the
reader `shadow_report.py` and, eventually, the Phase 2 escalation rule will use.

### 9.5 P8 — 13D/G accumulation (presence proxy, not magnitude)

Reuses `EdgarDownloader.download_13dg_filings`, already run daily by P18's
scheduled scan (`edgar/13f/13dg/{date}.csv.gz`) — same "read an
already-warm shared cache" pattern as Form 4 in Phase 1.5, no new fetch
surface. **What's built**: "was any 13D/G filed against this CIK in the
trailing 2 quarters" (`inst_13dg_activity_2q`). **What's not**: the spec's
full P8 (new/increased positions specifically) needs the filing's Item 3/11
percent-of-class, which means fetching and parsing each 13D/G document — a
real 13F-style reverse-lookup job, deferred (matches the spec's own "Full 13F
reverse-lookup is a heavier quarterly job; defer to Phase 3" note — the
presence proxy here is the lighter thing that *was* judged worth building
this round).

### 9.6 P9 — no near-term debt maturity

`xbrl_facts.has_near_term_debt_maturity`: `LongTermDebtCurrent`, falling back
to `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths` (both
verified live against Apple's companyfacts to confirm the instant-fact shape
before trusting the tag names). A **12-month proxy for the spec's 24-month
window** — documented approximation, not the literal test. Unresolved (not
"no maturity") when neither tag is present at all, since absence could mean
"no debt" or "not tagged" and those aren't distinguishable without the tag.

### 9.7 P11 — short interest, conditional on grade

`StructuralProfiler` pulls `shortPercentOfFloat`/`shortRatio` directly from
`yfinance.Ticker(t).info` (same direct-call pattern as splits, §3.3).
`grading.py`'s `_eval_short_interest_conditional` resolves the disqualifier
set (has_d/has_c/coverage) **before** computing P11, so it can branch: at
grade A/B, feeds `insider_conviction` like any other resolved P-signal; at
C/D, marked `resolved=False` (excluded from the renormalisation — a real
exclusion, not missing data) and instead adds a config-sized bump
(`p11_dilution_urgency_bump`) directly to `dilution_urgency`. This mirrors
StructuralSignals.md §4's framing exactly: "identical raw number, opposite
meaning."

### 9.8 Sentiment attach (spec §10) — throttled, not truly per-poll

The spec text says "captured per poll", but the shadow loop polls every 15
minutes during market hours (~32×/day) while every other pipeline calling
`collect_sentiment_batch_sync` (P04's daily deep scan) does so once a day —
and mention counts don't meaningfully change on a 15-minute cadence. Unthrottled
per-poll calls would make P19 by far the heaviest consumer of these provider
rate limits in the codebase for no signal benefit. `SentimentCache` (new,
`sentiment_cache.py`) throttles the batch fetch to once per TTL window
(default 60 min) regardless of poll cadence; `ShadowLoop` reuses the existing
(previously unpopulated) `IntradaySignal.sentiment: Dict[str, float]` scaffold
field — no shadow-store schema change needed, `mentions_24h` /
`sentiment_score_24h` / `mentions_growth_7d` serialise into the same
`"k=v;k2=v2"` text column v1 already had.

### 9.9 New shadow-row field: `is_fpi`

The one Phase-3 addition to the per-poll denormalisation set (§6). Everything
else new folds into the existing `structural_grade`/`disqualifiers` columns,
but StructuralSignals.md §2 is explicit that the FPI population must be
tracked *separately* in calibration or the grade-vs-`close_retention` test
(spec §15) confounds two very different populations that both land in grade
C. `shadow_report.py` surfaces FPI share + a flag once it exceeds 20% of the
day's names, matching the coverage-flag pattern already in place.

### 9.10 Explicitly still deferred (documented, not silently dropped)

| Signal | Why deferred |
|---|---|
| **N12** (warrant overhang) | No XBRL tag exists, and — unlike N9 — no safe arithmetic shortcut (the baby-shelf rule) exists either. Real warrant-table text extraction is a materially harder problem than the boilerplate EX-23.1/phrase-match signals built this round; shipping a low-confidence guess for a signal that still writes a `disqualifiers` entry was judged worse than staying unresolved. |
| **N9 above $75M float** | Spec's own explicit fallback (§5) is to stay boolean above the baby-shelf threshold until the estimate proves to be the limiting factor against `dilution_event_within_5d` — untested, so not worth building the prospectus parser yet. |
| **P10** (insider ownership stability) | Needs Form 3 (initial ownership) ingestion, which nothing in the codebase currently downloads — approximating from Form 4 deltas alone has no baseline to delta from. |

All three fields stay `None` on `StructuralProfile` and correctly depress
`coverage` — never silently assumed resolved.

---

## Roadmap (Phases 2 and 4, not designed yet)

Kept intentionally thin — full design happens once Phase 1.5/3 data can
answer spec §15 Q1 (does `structural_grade` separate `close_retention`?),
which is the gate for whether the two-axis thesis is worth building alerting
around, and Phase 4's calibration is what's supposed to set Phase 2's
thresholds in the first place (decision #3).

- **Phase 2** — Disposition Engine (spec §8, the matrix already fully
  specified — just not implemented), dedup/escalation `State Store`, Alert
  Manager with per-disposition caps, Telegram delivery via existing
  `NotificationService`. The §8.2 escalation rule (intraday 424B5/S-1/8-K
  3.02 → forced `FADE_CANDIDATE`) can now be wired directly to §9.4's
  `filings_poll.py` event log.
- **Phase 4** — Optuna calibration of both axes (reuses the existing P17
  `strategy_sim.py` + Optuna harness pattern), LULD halt detection, optional
  LLM alert summarizer.
