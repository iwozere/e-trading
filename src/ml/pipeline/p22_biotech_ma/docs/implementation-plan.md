# P22 Biotech M&A — Implementation Plan

**Date:** 2026-08-30
**Based on:** `pipeline-specification.md` v0.6
**Status:** M1 (Ingest + storage) done. M2 (Entity resolution) functionally done — remaining items
(size/asset-floor filters, deal cross-reference) are blocked on other milestones, not on more M2 work
(§4.2, §4.3 below; historical exchange resolution was investigated and found to need new scraping
infrastructure — §4.3). M3 (Feature store) in progress — scaffolding, Block C, data-quality checks,
and the lookahead-audit logic are built (§4.4 below); Blocks A/B/D/E/F are blocked on data/config this
session flagged rather than fabricated — see `docs/Tasks.md` "Decisions needed". M4–M10 not started.

---

## 1. Spec-vs-repo deviations, resolved here

The spec was written stack-agnostic (§7.1). This repo already has house conventions that
supersede the spec's stack table where they conflict. Recording the resolution once, here,
rather than re-litigating it in every module docstring.

| Spec says (§7.1) | This repo does instead | Resolution |
|---|---|---|
| Orchestration: Prefect 2 or Dagster | A custom `job_schedules` DB table (cron string + dotted `target` module + `task_params`), executed as subprocesses by the existing scheduler service | **Follow repo convention.** P22 jobs are `run_*.py` scripts under `jobs/`, registered via `jobs/register_jobs.py`, identical in shape to P20 Kestrel's. No Prefect/Dagster dependency added. |
| Storage — warehouse: Postgres 15+ or DuckDB | Postgres via SQLAlchemy 2.0 + Alembic, one shared `Base`/`MetaData`, a `Repo`/`Service.uow()` layering | **Follow repo convention.** Postgres only; DuckDB not considered — the repo has no DuckDB infra and introducing a second warehouse engine for one pipeline isn't worth the operational cost. |
| Transform: dbt-core | No dbt anywhere in this repo | **Not adopted.** Normalization/entity-resolution logic lives in plain Python ingest/repo code, matching every other pipeline module here. Revisit only if a future pipeline independently justifies bringing in dbt. |
| Validation: Great Expectations or pandera | `pandera` is already a repo dependency (used elsewhere) | **Use `pandera`**, not Great Expectations, when §8.2 data-quality checks are implemented (M3). |
| API: FastAPI (§7.1, M10) | Not yet needed for M1–M9 | Deferred to M10 as specced; no API scaffolding added yet. |

Everything else in §7.1 (Python 3.11+, httpx, scikit-learn/LightGBM/SHAP) is followed as specced.

---

## 2. What already exists — reuse map

### 2.0 Universe construction (spec §2.0, added v0.5)

The spec is explicit that the universe must be built from EDGAR (point-in-time, survivorship-free
via the SEC DERA Financial Statement Data Sets), never from a market-data vendor's current roster —
building it from a vendor snapshot silently drops every acquired/delisted company, which is exactly
the set carrying positive labels. This reshapes §2.4's "vendor gap" materially:

| Spec need (§2.0.5) | Existing component | Notes |
|---|---|---|
| Point-in-time universe roster, survivorship-free | **Net new** — SEC DERA Financial Statement Data Sets (`sub.txt`, quarterly ZIP) | Not the same as anything currently in `src/data/downloader/`; landing page URL must be resolved dynamically, not hardcoded (spec §2.0.1 warns the path has moved before) |
| Fundamentals: cash, burn, debt, shares, as-filed | `EdgarDownloader.download_company_facts` | Already covered, §2.1 above |
| Historical ticker/exchange resolution (delisted names) | **Net new** — parse `dei:TradingSymbol` / `dei:SecurityExchangeName` off 10-K/10-Q cover pages | Current snapshot endpoints (`company_tickers.json`) don't resolve historical/delisted tickers; cover-page XBRL is point-in-time by construction (§2.0.2) |
| Daily adjusted prices + corporate actions, current names | `src.data.downloader.ibkr_downloader` (already integrated, already paid for) | Per spec's own capability matrix (§2.0.5), IBKR is the assigned source — **not** a new vendor. Mind IBKR's aggressive historical-data pacing limits over a 700-ticker loop. |
| Options IV, short interest / borrow | IBKR | Same as above — sole source per the matrix |
| **Historical prices for *delisted* tickers** | **The one real gap** (§2.0.6) | Needed for `E[return \| deal]` (§5.3) labels — positive-label companies are exactly the ones that stopped trading. Spec's own recommendation: **FMP Starter (~$15/mo)** is the cheapest path; validate delisted-ticker coverage against ~20 known acquisitions before committing. This is a much narrower, cheaper ask than "pick a market-data vendor" — revisit the earlier deferred vendor decision with this framing before M6/M7. |
| yfinance | Already integrated (`yahoo`/yfinance downloaders exist) | Spec: **prototyping only** — no delisted coverage, no redistribution grant, unofficial scraper. Not used in any P22 pipeline path, only acceptable in a scratch notebook. |
| FMP free tier | Already integrated | Spec: **cannot serve this system** at free-tier limits (250 calls/day, EOD-only, no 13F). Confirms the free tier is not the answer even for current-name data; FMP only becomes useful at a paid tier and specifically for the delisted-price gap above. |

**Consequence for M3 (Block A/C features, `dilution_gate`):** for *currently listed* names, `enterprise_value`/`cash_runway_months`/market cap can likely be built entirely from EDGAR (shares outstanding, as-filed) + IBKR (daily price) — no new vendor needed. The deferred-vendor decision (§2.4 below) narrows almost entirely to the backtest's delisted-name price history, not live scoring. This should be re-confirmed once M3 starts, not assumed.

Acquirer universe (§2.0.4) is a small (~25 company), hand-curated `config/acquirers.yaml` — not screened, not sourced from any of the above. Tracked as an M1-adjacent config task in `docs/Tasks.md` since Block A (M3) depends on it existing.

#### 2.0.7 Price archive and corporate actions (added v0.6)

New in v0.6: raw-price storage with read-time adjustment is now mandatory, not merely a good idea. The
spec's own rationale (§2.0.7) — storing adjusted prices corrupts point-in-time market cap (computed
against as-filed, unadjusted share counts) and turns a retro-adjusted price level into a lookahead
leak — applies with full force to this pipeline's microcap biotech universe, where reverse splits are
common and frequently coincide with a ticker change.

This is schema/design work, buildable now independent of the deferred vendor decision (§2.4 below):

| Component | Status |
|---|---|
| `p22_price_daily` (raw OHLCV, PK `company_id, trade_date, vendor`) | **Built** — `model_p22_biotech_ma.py`, migration `004_p22_price_archive` |
| `p22_corporate_action` (splits/dividends/spinoffs/ticker changes) | **Built** — same migration |
| Read-time adjustment (`adjusted_close`, spec's own pseudocode) | **Built** — pure math in `ingest/price_archive.py` (unit-testable without a DB), combined with a live lookup in `P22Repo.get_adjusted_close` |
| Raw-row immutability ("never rewritten") | **Enforced at the DB layer** — `P22Repo.upsert_price_daily` uses `ON CONFLICT DO NOTHING` on `(company_id, trade_date, vendor)`, not `DO UPDATE`; a second write for the same key is silently a no-op rather than overwriting |
| Actual price/corporate-action ingest job | **Not built** — blocked on the same deferred vendor decision as §2.4; the schema exists so M6/M7 doesn't also need a schema-design pass once a vendor is picked |

**A likely finding that would change the §2.0.5 capability-matrix assignment — flagged, not yet
live-verified against a real IBKR session (no TWS/Gateway connection available in this pass):** the
existing `src.data.downloader.ibkr_downloader` calls `reqHistoricalData(whatToShow="TRADES", useRTH=True)`.
IBKR's documented historical-data behavior is that TRADES bars are **automatically split-adjusted
server-side** for stocks (dividends are not adjusted) — there is reportedly no request parameter to get
genuinely raw, as-traded prints. If that holds, re-downloading the same historical window after a split
would silently change the older bars — exactly the "retroactive rewrite" spec §2.0.7 says raw storage
exists to prevent, meaning **IBKR would not be usable as the `p22_price_daily` source for split-adjustment
purposes as currently integrated**, despite §2.0.5 assigning it "daily adjusted prices, current names."
**Action item before M6/M7 build against this:** confirm this against a live IBKR session (this session's
established practice of live-verifying every external-data assumption — see §4.1 — applies here too, it
just couldn't be done this pass). Two ways to reconcile if confirmed, neither implemented yet:
1. Store IBKR bars as vendor=`'ibkr'` rows anyway, accept they are pre-adjusted-as-of-fetch-time (not
   truly raw), and rely on the SEC-filing corporate-action reconciliation job (§2.0.7's precedence order,
   item 1) to detect and flag the resulting discontinuities rather than silently trusting them.
2. Source raw prints from whichever vendor is eventually picked for the delisted-ticker gap (§2.0.6) —
   if that vendor's API can return unadjusted data (FMP's historical endpoints generally can, with an
   explicit parameter), use it for *all* `p22_price_daily` writes, not just delisted names, and drop IBKR
   from the price-archive role entirely (keep it for options IV / short interest / borrow, where this
   problem doesn't apply).

Recorded here rather than resolved, since it interacts with the still-open vendor decision. See
`docs/Tasks.md` Known Issues.

### 2.1 SEC EDGAR

| Spec need (§2.1, §2.5, §2.6) | Existing component | Notes |
|---|---|---|
| CIK resolution, submissions, filing history | `src.data.downloader.edgar_downloader.EdgarDownloader` | `resolve_tickers_to_ciks`, `download_submissions`/`load_submissions`, `get_recent_filings(form_type=...)` |
| XBRL company facts (cash/burn/debt/shares) | `EdgarDownloader.download_company_facts` / `load_company_facts` | Already implements the exact tag-fallback list §2.1 specifies |
| Full-text search (EFTS) — 8-K item/phrase search | `EdgarDownloader.efts_filings_search`, `efts_text_search` | Directly reusable for §2.6.1 strategic-alternatives phrase detection once M5 starts |
| 13F-HR quarterly holdings | `EdgarDownloader.download_13f_index` / `download_13f_infotable` / `parse_13f_infotable` | Reuse for Block F (M3+); already used by P18 |
| Form 4 insider transactions (all codes) | `EdgarDownloader.download_form4_filings` + `_parse_form4_xml` | Reuse for Block F |
| Schedule 13D/13G | `EdgarDownloader.download_13dg_filings` | Uses quarterly `form.idx` full-index (EFTS does not index 13D/G) — same technique needed for §2.6.2 in M5 |
| Form 10/10-12B (spin-offs) | `EdgarDownloader.download_form10_filings` | Not needed by P22, noted for completeness |
| Default User-Agent | `EdgarDownloader(user_agent="e-trading-research akossyrev@gmail.com")` | Already satisfies §2.1's `"{AppName} {contact-email}"` requirement — reused as-is, no new config needed |
| **SC 14D9 / DEFM14A / S-4** (§2.5 deal labels) | **Not present** — net new | EFTS indexes these as ordinary filing types (unlike 13D/G), so this is an incremental `EdgarDownloader` method using the *existing* `efts_filings_search` machinery, not a new fetch layer. **Scoped to M6**, not M1. |

**P18 (`p18_institutional_flow_tracker`)** already calls `EdgarDownloader` for 13F/13D-G but persists only to filesystem CSV caches — it has no DB tables. P22 cannot reuse P18's *storage*, only the *downloader calls*; P22's own bitemporal tables are the persistence layer (§3.2).

### 2.2 Database layer

| Spec need | Existing component | Notes |
|---|---|---|
| Declarative ORM base | `src.data.db.core.base.Base` (shared `MetaData`, one `create_all()`) | P22 models declared on this same `Base` |
| JSON/JSONB column type | `src.data.db.core.json_types.JsonType` | Used for `score.subscores`, `score.contributions`, `review_item.payload` |
| Migrations | Alembic at `src/data/db/migrations/versions/`; latest is `002_kestrel` (`down_revision="032f9959e8cf"`) | P22 adds `003_p22_biotech_ma_schema.py`, `down_revision="002_kestrel"` |
| Session / UoW | `src.data.db.core.database.session_scope()`, `src.data.db.services.database_service.DatabaseService.uow()` → `ReposBundle` | P22 adds a `p22: P22Repo` field to `ReposBundle`, mirroring `kestrel: KestrelRepo` |
| Repo pattern | `src.data.db.repos.repo_kestrel.KestrelRepo` | Session-in-constructor, `flush()` not `commit()`, `pg_insert(...).on_conflict_do_update(...)` for upserts |
| Table naming | `k20_*` for Kestrel | **P22 uses `p22_*`** — same convention, new prefix per pipeline |

**No bitemporal (`valid_from`/`valid_to`/`known_from`) pattern exists anywhere else in this codebase.** This is genuinely new design work for P22, not a reuse case. Two things follow from that:

1. The restatement rule in §2.4 ("never update in place; close `valid_to`, insert a new row") is implemented once, generically, as `repo_p22_biotech_ma.py:upsert_financial_fact_bitemporal()`, and every future bitemporal write path reuses that one function rather than hand-rolling the close-and-insert logic per call site.
2. The §8.3 lookahead-audit test is the only thing in this repo that will ever exercise this invariant end-to-end — it is written in M1 alongside the schema (see §5 below), not deferred to M3 as the milestone table implies, because a schema whose core property has never been tested is not "live."

The real-Postgres restatement test lives in its own `tests/db/` subdirectory, not `tests/` directly: the repo-layer conftest's `_apply_migrations` fixture is session-scoped `autouse=True`, so any test file that imports it forces every *other* test collected in the same directory to also connect to Postgres. Isolating it to a subdirectory keeps P22's mocked-HTTP/no-DB unit tests running without a database configured at all.

### 2.3 Rate limiting

`src.data.utils.rate_limiting.RateLimiter` (thread-safe, per-second + per-minute, configurable backoff) already exists and is reused for the CT.gov and openFDA clients — one instance per host, per §7.2's "single shared token-bucket limiter per host." `EdgarDownloader` has its own internal rate limiting/backoff already wired for SEC's 10 rps cap; not touched.

### 2.4 Market/fundamentals vendor (§2.4, narrowed by §2.0.5–2.0.6)

**No existing integration provides point-in-time, survivorship-bias-free historical prices for *delisted* tickers** — that is now the precisely-scoped gap, not "point-in-time market data" broadly (see §2.0 above: EDGAR + already-integrated IBKR cover live names). FMP, Polygon, Tiingo, EODHD etc. are integrated for current-snapshot data only, and per §2.0.6 the spec's own recommendation is FMP Starter (~$15/mo) as the cheapest path once validated. Per explicit decision (2026-08-30): **defer vendor selection** regardless — `ingest/vendor_market_data.py` defines the adapter `Protocol` the spec's bitemporal contract requires (`known_from` = vendor first-publication date, restatements as new rows, `config/vendor_lag.yaml` fallback per §2.4) plus a `NullMarketDataProvider` that raises `NotImplementedError` on every call. This is now understood to block specifically `E[return | deal]` labeling (M6/M7), not Block A/C live scoring — tracked in `docs/Tasks.md`.

### 2.5 ClinicalTrials.gov / openFDA / Orange Book / Purple Book

Nothing exists in this repo for any of these. All four clients are net new, built in M1 (§2.2–2.3 of the spec).

---

## 3. Net-new build — M1 scope

M1's definition of done (spec §9): *"SEC, CT.gov, openFDA, Orange Book landing in raw zone daily; bitemporal schema live; rate limits respected."* This plan additionally lands Purple Book alongside Orange Book (§2.3 names both as M1-adjacent sources sharing the same quarterly-ZIP ingest shape), lands the SEC DERA Financial Statement Data Sets that §2.0 (added v0.5) makes the sole basis for universe construction — without it there is no company list for any other M1 source's per-company pulls to target, so it belongs in M1 rather than M2 — and includes the §8.3 lookahead-audit scaffold per §2.2 above.

**Note:** M1 here lands the raw DERA quarterly archives only (`sec_universe_ingest.py`, mirroring the SEC-raw-zone pattern). Turning that into an actual populated `p22_company` roster — applying the §2.0.3 eligibility filters, resolving historical tickers off 10-K/10-Q cover pages, and cross-referencing roster disappearances against `deal` — is entity-resolution work and stays in M2, consistent with the milestone table (§9: "M2 Entity resolution... review queue tooling functional").

Explicitly **not** in M1: entity resolution (M2), feature computation (M3), scoring (M5.1+), deal labels (M6). The DB schema for those milestones' tables (`asset`, `trial`, `patent_expiry`, `deal`, `corporate_process_event`, `activist_position`, `partnership_structure`, `score_run`, `score`) is created now (schema-live requirement) but left unpopulated — no ingest job writes to them yet.

### 3.1 Module structure

```
src/ml/pipeline/p22_biotech_ma/
├── __init__.py
├── config.py                        # paths, source URLs, feature flags, EDGAR UA reuse
├── ingest/
│   ├── __init__.py
│   ├── raw_zone.py                  # partitioned, content-addressed, gzip JSON raw-zone writer (§7.3)
│   ├── http_retry.py                # shared GET-with-retry: retries only 429/5xx, never other 4xx (§7.2)
│   ├── rate_limits.py               # shared RateLimiter instances, one per host (§7.2)
│   ├── sec_universe_ingest.py       # SEC DERA Financial Statement Data Sets -> raw zone (§2.0, universe basis)
│   ├── sec_raw_ingest.py            # EdgarDownloader submissions/companyfacts -> raw zone, known_from stamping
│   ├── clinicaltrials_client.py     # CT.gov API v2 + version-history pull (§2.2)
│   ├── openfda_client.py            # Drugs@FDA (§2.3)
│   ├── orange_book_client.py        # Orange Book quarterly ZIP (products/patent/exclusivity)
│   ├── purple_book_client.py        # Purple Book CSV
│   ├── vendor_market_data.py        # Protocol + NullMarketDataProvider stub (§2.4, deferred vendor)
│   ├── price_archive.py             # pure split-adjustment math (§2.0.7, added v0.6)
│   ├── entity_resolution.py         # M2: DERA rows -> UniverseCandidate, eligibility filters (§2.0.2-3);
│   │                                 #     build_universe_history() for per-quarter point-in-time re-computation
│   ├── alias_matching.py            # M2: sponsor-name -> company_id, deterministic + fuzzy (§3.3)
│   ├── review_queue.py              # M2: confirm_item/reject_item/queue_depth_report (§3.4)
│   ├── financial_facts.py           # M3: SEC XBRL companyfacts -> p22_financial_fact (§2.1, §3.1); tag-fallback merge + quarter-delta derivation
│   ├── trial_normalization.py       # M3: CT.gov studies -> p22_trial (§2.2, §3.2); asset_id linked for single-intervention trials only
│   ├── asset_normalization.py       # M3: single-intervention trial -> p22_asset, deduped per (company_id, name)
│   ├── therapeutic_area_classifier.py  # M3: best-effort keyword classifier, CT.gov conditions -> therapeutic_area.yaml
│   ├── acquirer_config.py           # M3: p22_acquirers.yaml -> p22_company identity rows (§2.0.4)
│   ├── patent_expiry_normalization.py  # M3: Orange Book patent.txt -> p22_patent_expiry (§2.3, §4.1)
│   ├── fmp_client.py                # M3: FMP /stable client (historical price, name search) — live-verified endpoints
│   ├── fmp_universe.py              # M3: known-ticker vs. needs-name-search universe split
│   ├── fmp_backfill.py              # M3: bulk historical-price backfill orchestration, resumable
│   ├── yfinance_client.py           # M3: narrow trailing-window daily bars — never wide/historical, see docstring
│   └── price_ingest.py              # M3: yfinance bars -> p22_price_daily/p22_corporate_action
├── features/                         # M3: feature store (spec §4)
│   ├── __init__.py
│   ├── context.py                   # FeatureContext — the one lookahead-safe read path every feature uses
│   ├── registry.py                  # register_feature/get_feature/list_features
│   ├── block_c.py                   # Block C — Financial Screen (§4.3), all 6 features
│   ├── quality.py                   # pandera schemas + assert_every_company_has_a_verified_alias (§8.2)
│   └── lookahead_audit.py           # stratified_sample + assert_lookahead_safe (§8.3) — logic only, not wired to real data yet
├── jobs/
│   ├── __init__.py
│   ├── run_common.py                # logging setup, mirrors p20_kestrel/jobs/run_common.py
│   ├── run_sec_ingest.py
│   ├── run_financial_facts_normalization.py  # M3: sec_company_facts -> p22_financial_fact
│   ├── run_entity_resolution.py     # M2: writes p22_company from the latest DERA snapshot
│   ├── run_clinicaltrials_ingest.py
│   ├── run_openfda_ingest.py
│   ├── run_alias_matching.py        # M2: landed CT.gov/openFDA sponsor names -> p22_company_alias
│   ├── run_trial_normalization.py   # M3: landed CT.gov studies -> p22_trial
│   ├── run_acquirer_load.py         # M3: p22_acquirers.yaml -> p22_company identity rows
│   ├── run_orange_book_ingest.py
│   ├── run_patent_expiry_normalization.py  # M3: landed Orange Book -> p22_patent_expiry
│   ├── run_purple_book_ingest.py
│   ├── run_price_ingest.py          # M3: DAILY (unlike fmp_backfill's one-time run) — yfinance current prices
│   └── register_jobs.py             # idempotent job_schedules upserts
├── cli/                              # human-run interactive tools, NOT scheduler jobs (§3.4)
│   ├── __init__.py
│   ├── review_queue_cli.py          # M2: argparse status/list/show/confirm/reject over p22_review_item
│   └── fmp_backfill_cli.py          # M3: test-search / backfill --dry-run / backfill (one-time, run during a Premium month)
├── tests/
│   ├── __init__.py
│   ├── test_db_models.py            # table-shape assertions, no live DB (mirrors P20's)
│   ├── test_raw_zone.py             # dedup-by-hash, partitioning, immutability, latest-partition (+manifest) read
│   ├── test_http_retry.py
│   ├── test_clinicaltrials_client.py
│   ├── test_openfda_client.py
│   ├── test_orange_book_client.py
│   ├── test_sec_universe_ingest.py
│   ├── test_sec_raw_ingest.py
│   ├── test_universe_snapshot.py    # incl. all_landed_quarters spanning multiple ingest dates
│   ├── test_vendor_market_data.py
│   ├── test_price_archive.py        # pure adjustment math, no DB
│   ├── test_entity_resolution.py    # incl. build_universe_history
│   ├── test_alias_matching.py       # incl. field-extraction, known_from threading
│   ├── test_review_queue.py         # confirm/reject dispatch, queue-depth report
│   ├── test_financial_facts.py      # incl. comparative-column dedup, total_debt tag-migration merge, quarter-delta derivation
│   ├── test_trial_normalization.py  # incl. NA-allocation, partial-date (YYYY-MM), and asset-linking edge cases
│   ├── test_asset_normalization.py  # incl. dedup-by-(company_id,name), multi-intervention-stays-unlinked
│   ├── test_therapeutic_area_classifier.py  # incl. heme-vs-solid-oncology ordering, never-guessed categories
│   ├── test_acquirer_config.py      # incl. round-trip against the real repo config file
│   ├── test_patent_expiry_normalization.py  # incl. unmatched-product and blank-date drop cases
│   ├── test_fmp_client.py           # incl. 402/404/unexpected-shape cases
│   ├── test_fmp_universe.py         # incl. dedup-by-CIK-keeping-latest-name
│   ├── test_fmp_backfill.py         # incl. the live-caught multi-exact-match tie-break, resumability
│   ├── test_yfinance_client.py      # incl. the narrow-trailing-window request assertion
│   ├── test_price_ingest.py         # incl. forward/reverse split ratio handling
│   ├── test_feature_context.py      # incl. get_trailing_average
│   ├── test_feature_registry.py
│   ├── test_block_c.py              # every feature's real-computation path AND null path (§8.1)
│   ├── test_quality.py              # pandera schema accept/reject cases
│   ├── test_lookahead_audit.py      # stratified sampling + both assertion functions
│   └── db/                          # isolated: real-Postgres tests only
│       ├── __init__.py
│       ├── conftest.py              # re-imports src/data/db/tests/repos/conftest.py fixtures
│       └── test_repo_p22_bitemporal.py  # restatement/price-archive/review-item/alias-coverage/trial/acquirer-merge/patent-expiry/asset round trips; opt-in (ETRADING_TEST_DB_URL)
└── docs/
    ├── pipeline-specification.md    # (already present)
    ├── implementation-plan.md       # this file
    ├── Requirements.md
    ├── Design.md
    └── Tasks.md

config/pipeline/                      # spec §3.5 controlled vocabularies + §2.0.4 acquirer universe
├── p22_acquirers.yaml                # 25 acquirers, CIKs live-verified, list count settled 2026-08-31 — see docs/Tasks.md item 3
├── p22_therapeutic_area.yaml         # reviewed against the real BIO study taxonomy 2026-08-31; incl. `unclassified` fallback
├── p22_modality.yaml                 # DRAFT — still needs domain review, no primary source found for this one
├── p22_base_rates.yaml               # 15/21 therapeutic areas populated from a real primary source 2026-08-31 — see docs/Tasks.md item 2
└── p22_cvr_policy.yaml               # M6-scope, created early 2026-08-31 at user request — spec's default (value CVRs at zero)
```

### 3.2 Database additions (outside the module, per repo convention)

- `src/data/db/models/model_p22_biotech_ma.py` — all `p22_*` tables from spec §3.2 + `p22_review_item` (§3.4, plus a `created_at` column the spec's own SQL sketch omits — see migration 005) + `p22_fetch_failure` (§7.2) + `p22_price_daily`/`p22_corporate_action` (§2.0.7, added v0.6).
- `src/data/db/migrations/versions/003_p22_biotech_ma_schema.py` — `down_revision="002_kestrel"`.
- `src/data/db/migrations/versions/004_p22_price_archive.py` — `down_revision="003_p22_biotech_ma"`, the v0.6 price-archive tables.
- `src/data/db/migrations/versions/005_p22_review_item_created_at.py` — `down_revision="004_p22_price_archive"`, adds `p22_review_item.created_at`.
- `src/data/db/repos/repo_p22_biotech_ma.py` — `P22Repo`, session-in-constructor, `upsert_financial_fact_bitemporal()` as the one generic restatement-safe write path; `upsert_price_daily()`/`get_adjusted_close()` for the v0.6 price archive; `list_companies()` as the alias-matching job's match target; `get_review_item()`/`resolve_review_item()` for the review-queue CLI; `get_companies_without_verified_alias()` (M3) backing `features/quality.py`'s spec §8.2 check.
- `src/data/db/services/database_service.py` — add `p22: P22Repo` to `ReposBundle` and its construction in `uow()`. One-line addition, existing fields untouched.

### 3.3 Raw zone convention

The spec calls for S3-or-local, partitioned `source/date/`, immutable, gzipped JSON (§1 diagram, §7.3). This repo's existing convention (P15/P20) is `DATA_CACHE_DIR/<source>/...`. P22 follows the same root but adds the spec's idempotency requirement explicitly: `DATA_CACHE_DIR/p22/raw/<source>/<YYYY-MM-DD>/<content-hash>.json.gz`, where `<content-hash>` is a SHA-256 of the normalized payload — identical payloads across runs are automatically deduplicated (§7.3), and every write is `known_from`-stamped in a companion manifest row rather than relying on filesystem mtime.

---

## 4. Milestone tracking (spec §9)

| # | Deliverable | Status |
|---|---|---|
| M1 | Ingest + storage | **Done** — this plan |
| M2 | Entity resolution | **Functionally done** — roster build, alias matching, per-quarter re-computation, review-queue CLI all done (§4.2, §4.3); size/asset-floor filters and deal cross-reference remain, both blocked on later milestones |
| M3 | Feature store | **In progress** — scaffolding (`FeatureContext` incl. `get_trailing_average`, registry), Block C (all 6 features tested; `cash_runway_months`/`dilution_risk`'s runway leg now real), `pandera` checks, lookahead-audit logic, financial-fact normalizer (now incl. `total_debt`/`short_term_investments`/`quarterly_opex_burn`), CT.gov trial normalizer, acquirer-roster loader, Orange Book patent-expiry normalizer, 4 config YAML drafts all done (§4.4, §4.5, §4.6); `p22_asset` normalization and Blocks A/B/D/E/F not started — blocked on the vendor decision, real base-rate data, curated acquirer dates, the CT.gov intervention-linkage decision, and 8-K/13F/13D infrastructure, not on more coding |
| M4 | Rule-based scoring | Not started |
| M5 | Block G — process signals | Not started |
| M6 | Labels + backtest | Not started |
| M7 | Return model | Not started |
| M8 | Calibrated model | Not started |
| M9 | Partnership structures | Not started |
| M10 | API + alerts | Not started |

---

## 4.1 Live-verified corrections (2026-08-30)

Every M1 client was exercised against its real live source (not just mocked) before being
considered done — this surfaced several places where a literal reading of the spec's endpoint
shapes doesn't match reality:

- **CT.gov `fields` param** — only `NCTId` and `hasResults` are accepted as bare names; every other
  field in spec §2.2's list (`briefTitle`, `overallStatus`, ...) 400s and needs its full
  `protocolSection.<module>.<field>` path. `config.CLINICALTRIALS_FIELDS` uses the qualified paths;
  there is no flat `locationCountries` field in v2, so the full `locations` array is pulled instead.
- **CT.gov version history has no documented endpoint.** `/api/v2/studies/{nctId}/history` 404s.
  The data only exists behind `/api/int/studies/{nctId}/history` — undocumented, backs CT.gov's own
  history-viewer UI, returns `{"changes": [...]}` with per-version `moduleLabels` (which modules
  changed, not a field-level diff). Same risk class as P20's pdufa.bio dependency: could change or
  disappear without notice, and there's no documented alternative for this spec-required data
  (§2.2, "Critical").
- **openFDA `sponsor_name` search is case-sensitive** against the stored (uppercase) values —
  `sponsor_name:Pfizer` 404s, `sponsor_name:PFIZER` returns 191 real records. The client uppercases
  the search term.
- **Purple Book has no stable "latest" URL** — one dated CSV per month
  (`.../PurpleBook/{year}/purplebook-search-{Month}-data-download.csv`), each a full ~2,270-row
  snapshot with an `N/R/U` change-flag column, not a diff despite the "Monthly ... Changes Report"
  title. `orange_book_client.discover_latest_purple_book_url` derives the current URL from the
  downloads listing page. The CSV also has 3 preamble rows before the real header, located by
  content (`N/R/U` prefix) rather than a fixed offset.
- **Retry logic bug, found while fixing the above:** every client's original retry loop called
  `resp.raise_for_status()` on the success path, which raises `HTTPStatusError` for *any* 4xx/5xx —
  so a genuine 400 (bad query) or 404 (no results) was being retried 5 times with exponential
  backoff before giving up, instead of failing fast. Fixed by centralizing retry logic in
  `ingest/http_retry.py`, which retries only 429/5xx and returns non-retryable responses (2xx or
  other 4xx) immediately so the caller can inspect the status.

Orange Book (`ORANGE_BOOK_ZIP_URL`, tilde-delimited `products.txt`/`patent.txt`/`exclusivity.txt`,
including the exact `Patent_Expire_Date_Text` column spec §2.3 names) and SEC DERA discovery
(`SEC_DERA_LANDING_PAGE`, `2010q1`-present archive links) were both verified correct as originally
written — no changes needed there.

## 4.2 M2 slice: universe/roster + alias matching (2026-08-30, updated same day)

**Built and tested:**

- `ingest/entity_resolution.py` — turns landed DERA rows into `UniverseCandidate`s: reporting-status
  eligibility (10-K/10-Q within the trailing 6 months, spec §2.0.3), current ticker/exchange resolution
  via `company_tickers_exchange.json` (net new — this endpoint wasn't reused from anywhere existing),
  and a name-based SPAC heuristic. Name normalization (`normalize_company_name`) implements spec §3.3's
  exact token list.
- `jobs/run_entity_resolution.py` — writes non-SPAC-flagged candidates to `p22_company`
  (`is_active` = reporting eligibility); SPAC-flagged candidates go to `p22_review_item`
  (`item_type='entity_match'`) instead of being silently dropped or silently excluded, since the
  heuristic is name-based, not authoritative. Registered in `register_jobs.py`, scheduled 30 minutes
  after the SEC universe ingest it depends on.
- `ingest/alias_matching.py` — the spec §3.3 two-step resolver (deterministic normalized-name match,
  then `rapidfuzz.fuzz.token_set_ratio >= 88` routed to the review queue, never auto-accepted). Added
  `rapidfuzz` as a new dependency — nothing in this repo already does fuzzy string matching. Tested
  against synthetic sponsor-name variants (typos, suffix differences).
- `jobs/run_alias_matching.py` — wires the resolver up. Before writing extraction code, live-verified
  (2026-08-30) the actual field paths against both real APIs rather than trusting the spec's field
  names: CT.gov's `protocolSection.sponsorCollaboratorsModule.leadSponsor.name` and openFDA's
  top-level `sponsor_name` both confirmed as documented/expected. One live finding worth noting:
  a real CT.gov `leadSponsor.name` value came back as a merger-notice sentence ("Pfizer's Upjohn has
  merged with Mylan to form Viatris Inc.") rather than a clean company name — not special-cased, it
  just flows into `match_alias`, fails to score above threshold against anything, and correctly lands
  in `unresolved` rather than being force-matched. The job reads the latest landed
  `clinicaltrials_studies`/`openfda_drugsfda` raw-zone partitions paired with the ingestion timestamp of
  each (`raw_zone.read_latest_partition_with_known_from` at the time this was written; renamed and
  generalized to `read_latest_partition_with_manifest` in §4.4 below)
  and the resolved roster (new `P22Repo.list_companies`), then calls `resolve_aliases` once per source.
  Registered in `register_jobs.py`, scheduled daily after both ingest jobs land that day's data.
  Collaborator names (also present in the landed CT.gov payload) are deliberately not extracted as
  alias candidates — spec §3.3/§2.2 point at `leadSponsor` specifically.

**Deliberately not attempted this pass, and why:**

- **Historical ticker/exchange resolution off 10-K/10-Q cover pages** — investigated and found to need
  new infrastructure, not just more time. See §4.3.
- **Size floor and asset floor** (spec §2.0.3) — size floor needs market cap, blocked on the vendor
  decision (§2.0.6); asset floor needs `p22_trial` populated and linked to a resolved company, which
  needs CT.gov data normalized out of the raw zone (M3 work). Both fields exist on `UniverseCandidate` as
  explicit `None`, not silently skipped or defaulted to `True`.
- **Per-quarter point-in-time re-computation** — done in §4.3 below.

## 4.3 M2 continuation: per-quarter re-computation, review-queue CLI, a bug fix, and one negative finding (2026-08-30)

**Built and tested:**

- `entity_resolution.build_universe_history()` — the point-in-time re-computation spec §2.0.3 requires
  ("applied per `as_of`, not once"). Walks every landed DERA quarter and computes eligibility for that
  quarter's own `as_of` from the cumulative union of everything filed up to and including it, not from
  today's roster — proven by a test where a company's `eligible_reporting` flips from `True` (shortly
  after its last filing) to `False` (a year later) purely as a function of which quarter is being asked
  about. Reads via the new `universe_snapshot.all_landed_quarters()`, which — unlike
  `latest_universe_rows()` — walks *every* ingest-date partition, not just the most recent one; DERA
  history is landed 15+ years at once but dated by *ingest* day, so restricting to the latest ingest date
  would silently drop every quarter except whatever the most recent run happened to land.
  **Deliberately not persisted anywhere** — no consumer (the M6 backtest harness) exists yet to define
  what a per-quarter storage shape needs to answer; inventing a `p22_company_history` table now, before
  that's known, risks building the wrong one. `eligible_exchange` in every historical quarter still comes
  from the *current* ticker/exchange snapshot — this function doesn't change that limitation, only the
  reporting-status and SPAC filters are genuinely point-in-time.
- `ingest/review_queue.py` + `cli/review_queue_cli.py` (spec §3.4: "a minimal review UI or, acceptably
  for v1, a CLI"). `confirm_item()` dispatches on `payload['reason']` to the correct downstream write —
  `spac_name_heuristic` -> `upsert_company` (using ticker/exchange/reporting-eligibility now carried in
  the payload itself, added this pass, rather than re-deriving them), `fuzzy_alias_candidate` ->
  `add_company_alias` — and raises `UnknownReviewItemReasonError` rather than silently marking an
  unrecognized item confirmed with no write, which would quietly lose the candidate. `reject_item()`
  only updates status; nothing downstream. The CLI (`status`/`list`/`show`/`confirm`/`reject`) is a
  human-run interactive tool under a new `cli/` directory, deliberately separate from `jobs/run_*.py`
  (which are scheduler-invoked and print `__SCHEDULER_RESULT__`) — the two have different callers and
  different contracts.
- **A real bug, found and fixed while building the above:** neither the deterministic nor the
  soon-to-be-fuzzy-confirmed `add_company_alias` write in `resolve_aliases` was setting `known_from` —
  it silently defaulted to `NULL`. This is exactly the failure mode spec §3.4 calls out by name:
  "confirmation writes back with `known_from` set to the underlying filing date, not the review date...
  getting this backwards silently destroys the backtest." Fixed by threading `known_from` through end to
  end: `raw_zone.read_latest_partition_with_manifest()` (new — pairs each landed payload with its full
  manifest, later generalized further in §4.4 to also expose `entity`) ->
  `resolve_aliases(candidates: List[Tuple[str, datetime]], ...)` ->
  the deterministic write immediately, or the fuzzy review-item payload (`payload['known_from']`) for
  `confirm_item` to apply later. For a non-filing source like a CT.gov sponsor string, "the underlying
  filing date" doesn't literally apply — the closest analog, and what's used, is when we actually
  observed the string (the raw-zone ingestion timestamp), not "now."
- `p22_review_item.created_at` (migration `005_p22_review_item_created_at`) — the spec's own §3.4 SQL
  sketch doesn't have this column, but "queue depth and median age by item_type are reported in every
  run" (§3.4) is unanswerable without one. `queue_depth_report()` is now logged at the end of both
  `run_entity_resolution.py` and `run_alias_matching.py` (the two current review-item producers).

**One negative finding, live-verified rather than assumed:**

- **Historical ticker/exchange resolution via `dei:TradingSymbol`/`dei:SecurityExchangeName`** (spec
  §2.0.2) turns out not to be reachable the way both the spec and this plan's earlier draft assumed.
  Checked SEC's XBRL `companyfacts` API (`https://data.sec.gov/api/xbrl/companyfacts/CIK...json`, what
  `EdgarDownloader.load_company_facts` already lands) against three real CIKs, including Meta/Facebook —
  a company with a well-documented historical ticker change (FB -> META, 2022), specifically chosen to
  rule out "maybe this filer just doesn't tag it." In all three, the `dei` facts object contained only
  `EntityCommonStockSharesOutstanding`/`EntityPublicFloat` — never `TradingSymbol`, `SecurityExchangeName`,
  or even `EntityRegistrantName`. SEC's companyfacts aggregation appears to only surface numeric-context
  XBRL facts, not the string-type cover-page identity facts, regardless of what the filer tagged inline
  in the filing's own HTML. Getting the real values requires fetching and parsing each filing's own
  iXBRL-tagged cover-page document directly — new scraping infrastructure `EdgarDownloader` does not have
  today, not a read of data this repo already lands. This is now a scoped, correctly-understood gap in
  `docs/Tasks.md`, not an unexamined one.

## 4.4 M3 slice: feature-store scaffolding, Block C, data quality, lookahead audit (2026-08-30)

**Built and tested:**

- `raw_zone.read_latest_partition_with_manifest()` — generalized from the M2-era
  `read_latest_partition_with_known_from` to expose the full manifest dict (not just `known_from`),
  since the financial-facts normalizer below also needs `entity` (the CIK). `run_alias_matching.py`
  migrated to the new name; its behavior is unchanged.
- `ingest/financial_facts.py` + `jobs/run_financial_facts_normalization.py` — the first real (not
  synthetic-only) M3 data path. Live-verified two XBRL tags (`cash_and_equivalents`,
  `shares_outstanding`) across three real biotech filers (Moderna, Sarepta, Alnylam) before writing
  extraction code against them — same discipline as every prior live-verification pass this session.
  A genuine correctness trap surfaced and was fixed while building this: SEC's XBRL companyfacts API
  re-reports an unchanged prior period's balance as a comparative column in every subsequent filing;
  naive re-processing of every entry would misdate that repetition as a fact first known on the later
  filing, corrupting the bitemporal history with a false-looking-safe but wrong `known_from`.
  `extract_fact_series` dedupes by `period_end`, keeping the earliest-filed entry, and separately
  detects (logs, does not silently apply) a genuine value change for an already-seen period — a known,
  documented limitation (restatements), not a silent bug.
- `features/context.py` (`FeatureContext`) + `features/registry.py` — the spec §4 scaffolding.
  Lookahead safety is centralized in `FeatureContext.get_latest_fact`, which delegates to the
  already-tested `P22Repo.get_financial_facts_as_of` — a feature function has no code path that could
  see a fact before its `known_from`.
- `features/block_c.py` — all 6 spec §4.3 features (`enterprise_value`, `cash_runway_months`,
  `ev_to_cash`, `dilution_risk`, `atm_capacity_pct`, `size_band`), unit-tested against both the real
  computation (synthetic fixtures with fabricated-but-labeled-as-fake facts) and the null path (spec
  §8.1). `None` propagates automatically wherever an input isn't normalized yet — no feature function
  contains a special case for "the vendor isn't wired up" or "burn rate isn't computed"; it just reads
  a metric that doesn't exist and gets `None` back, exactly as the store already guarantees.
- `features/quality.py` + `P22Repo.get_companies_without_verified_alias()` — pandera schemas for the
  two spec §8.2 bounds that apply to already-built features (`cash_runway_months`, `enterprise_value`/
  `market_cap`), a parameterized `loe_date` bound (found and fixed a real pandera/pandas interop bug
  while testing this: comparing a `datetime64[ns]` column against a plain `datetime.date` bound raises
  `TypeError` rather than failing the check cleanly — bounds must be `pandas.Timestamp`), and the
  "every company has a verified alias" set-membership check (not a column bound, so not a
  `DataFrameSchema` — a plain assertion function instead, DB-tested).
- `features/lookahead_audit.py` — spec §8.3's stratified-sampling and assertion logic
  (`stratified_sample`, `assert_lookahead_safe`,
  `assert_known_from_is_filing_date_not_period_or_crossing_date`). Pure, fully unit-tested, including
  the "fewer available samples than the floor" and "deterministic with a seeded RNG" edge cases.
  **Deliberately not wired to real DB data or CI** — see the "not attempted" note below.
- 4 config YAML drafts (`config/pipeline/p22_acquirers.yaml`, `p22_therapeutic_area.yaml`,
  `p22_modality.yaml`, `p22_base_rates.yaml`) — every field this session could not verify or curate
  with real authority is `null` or flagged in a header comment, never filled with a plausible guess.
  See `docs/Tasks.md` "Decisions needed" for exactly what's missing from each.
- **Correction:** `pandera` was added to `requirements.txt` for real this pass. `docs/Requirements.md`
  had claimed since M1 planning that it was "already a repo dependency" — that was never actually
  true; it wasn't installed or present anywhere in `requirements.txt`. Verified this time via a real
  `pip install` + `DataFrameSchema.validate()` call before trusting the claim, rather than repeating
  an unverified assumption forward again.

**Deliberately not attempted this pass, and why:**

- **Blocks A, B, D, E, F** — every one is blocked on real data or config this session flagged rather
  than fabricated: Block A needs the (unreviewed-draft) acquirer config plus market cap plus deal
  history (M6); Block B needs the (mostly-null) base-rate config plus `p22_trial`/`p22_asset`; Block D
  is computed from A-C's own outputs so it's blocked transitively; Block E needs 8-K/DEF 14A
  text-extraction infrastructure that doesn't exist; Block F needs 13F integration, which is M5/M6
  scope per the spec's own milestone table, not M3's.
- **`p22_trial`/`p22_asset`/`p22_patent_expiry` normalization** from the already-landed CT.gov/openFDA/
  Orange Book raw-zone payloads — the same shape of work as `ingest/financial_facts.py`, and the real
  prerequisite for Block B and the M2 asset-floor filter, but not started this pass.
- **Wiring the lookahead audit to real data and CI** — spec §8.3's stratified sample requires coverage
  of vendor-sourced facts, 13F holdings, and 13D/process events, and this repo has zero rows in any of
  those three categories today. Running the "mandatory, blocks the build" gate against an empty
  high-risk population would pass vacuously — a false signal that the safety property has been
  verified when it has never actually been exercised against the cases that matter. This is worse than
  not having the gate yet, so it isn't wired in until real data exists to sample.
- **Rest of `ingest/financial_facts.FACT_TAG_MAP`** (short-term investments, total debt, quarterly
  operating cash flow) — GAAP tag names vary by filer for these concepts in a way `cash_and_equivalents`/
  `shares_outstanding` didn't, and quarterly burn additionally needs quarter-delta derivation from
  XBRL's cumulative year-to-date duration contexts. Both need more live verification than this pass's
  time budget allowed for; guessing at tag names for a metric that feeds a probability/risk computation
  is exactly the failure mode this session's discipline exists to avoid.

## 4.5 M3 continuation: CT.gov trial normalization (2026-08-30, later same day)

**Built and tested:**

- `ingest/trial_normalization.py` + `jobs/run_trial_normalization.py`, registered in `register_jobs.py`
  after Alias Matching. First normalizer to read `run_clinicaltrials_ingest.py`'s landed
  `clinicaltrials_studies` payloads (previously only consumed for alias candidates, never turned into
  `p22_trial` rows). Field paths re-verified live against a real CT.gov study before writing extraction
  code (`designModule.enrollmentInfo.count`, `designModule.designInfo.allocation`,
  `statusModule.primaryCompletionDateStruct.date`, `contactsLocationsModule.locations[].country`,
  `outcomesModule.primaryOutcomes[].measure`) — all confirmed to match `CLINICALTRIALS_FIELDS`.
- `P22Repo.upsert_trial` — a plain upsert keyed on `nct_id`, not a bitemporal chain like
  `p22_financial_fact`. This is a deliberate, different choice from the financial-fact write path: a
  trial has one current state, and CT.gov's own version-history endpoint (already landed as
  `clinicaltrials_history`, still unconsumed) is the correct place for the change-over-time signal spec
  §2.2 wants, not a row-per-observation history in `p22_trial` itself.
- A real, live-confirmed finding that changed the plan: CT.gov's `designInfo.allocation` field has
  three values, not two — `RANDOMIZED`, `NON_RANDOMIZED`, and `NA` (for single-arm/observational
  designs). An earlier draft of this normalizer mapped anything non-`RANDOMIZED` to `is_randomized =
  False`; that's wrong for `NA`, which means "the question doesn't apply to this trial," not "this
  trial was not randomized." Fixed before writing any DB rows against it — `NA` maps to `None`.
- A second finding, from the same live study checked (`NCT05668741`, a Vertex-sponsored cystic-fibrosis
  trial with Moderna as collaborator): `armsInterventionsModule.interventions` listed two DRUG-type
  entries, `"VX-522 mRNA therapy"` (the asset under test) and `"IVA"` (ivacaftor, a Vertex-owned
  approved comparator co-administered in the study). CT.gov has no field distinguishing "the sponsor's
  own pipeline asset" from "an existing drug used as a comparator/combination partner" — a naive
  "first DRUG intervention" rule would have picked correctly here by luck, but there's no guarantee of
  ordering and no field to depend on. `p22_trial.asset_id` is written `None` on every row rather than
  guessed. Logged as `docs/Tasks.md` "Decisions needed" item 8 — this is a new open decision this pass
  surfaced, not one of the seven already logged.
- Orange Book file format live-verified (2026-08-30) ahead of building the patent-expiry normalizer
  next: downloaded the real current ZIP and confirmed `products.txt`/`patent.txt`/`exclusivity.txt`
  column headers and the `Patent_Expire_Date_Text` date format (`"Aug 24, 2026"`, i.e. `%b %d, %Y`)
  match the documented/spec-quoted shape exactly. Normalization code itself not yet written — see below.

**Deliberately not attempted this pass, and why:**

- **`p22_asset` population** — blocked transitively by the intervention-linkage finding above (can't
  create a correctly-linked asset without deciding how to resolve it) and additionally by
  `p22_asset.therapeutic_area` being `NOT NULL` in the schema with no conditions-text classifier built
  or decided on (see `docs/Tasks.md` item 4/8).
- **`p22_patent_expiry` normalization** — the Orange Book file format itself is now live-verified (see
  above), but `Applicant_Full_Name` (the only company identifier `products.txt` carries — no CIK) has
  nothing to resolve against yet: `p22_acquirers.yaml` is a draft config file only, never loaded into
  `p22_company` by any job. Building the patent-expiry normalizer before that loader exists would mean
  every row silently fails to match an acquirer — logged as part of `docs/Tasks.md` item 3 rather than
  built against an empty roster.

## 4.6 M3 continuation: acquirer roster, patent expiry, financial-fact tag expansion (2026-08-30, later same day)

Response to "implement what can still be implemented, leave those items which require [the user's]
decision/discussion for later" — everything below was tractable without a domain decision, either
because it was pure mechanical plumbing (the acquirer loader) or because the open question was really
"has this been live-verified," not "what does a human want" (the tag mapping).

**Built and tested:**

- `ingest/acquirer_config.py` + `jobs/run_acquirer_load.py` + `P22Repo.upsert_acquirer_company` —
  loads `p22_acquirers.yaml` acquirer *identity* into `p22_company`. Explicitly does NOT write
  `bloc`/`entry_date`/`exit_date` to any column (none exists in the spec's schema); those stay in the
  config for Block A to read directly. Ticker-keyed merge (not `cik`-keyed, since the config's CIKs
  are all `null`) that checks for an existing row under ANY role before inserting — an acquirer that's
  also a DERA-resolved target (common for large-cap pharma) gets its role merged to `both` instead of
  getting a duplicate, `cik`-less second identity. DB-tested for the new-row, merge, and
  repeated-call-is-idempotent cases.
- `ingest/patent_expiry_normalization.py` + `jobs/run_patent_expiry_normalization.py` +
  `P22Repo.upsert_patent_expiry`/`list_acquirer_companies`/`get_patent_expiries_for_acquirer` —
  normalizes landed Orange Book `products.txt`+`patent.txt` into `p22_patent_expiry`. Orange Book file
  format live-verified against the real, current ZIP ahead of writing extraction code (same discipline
  as every prior pass): confirmed `~`-delimited columns and the `Patent_Expire_Date_Text` format
  (`"Aug 24, 2026"`). Scoped deliberately to `patent.txt` only (not `exclusivity.txt`) and to
  deterministic-only applicant->acquirer matching — see `docs/Tasks.md` for the reasoning on both.
  `upsert_patent_expiry` is idempotent on `(acquirer_id, application_no, loe_date, source)`, DB-tested.
- `ingest/financial_facts.py` extended — `FACT_TAG_MAP` changed from `metric -> single tag` to
  `metric -> list of candidate tags, merged`. Added `total_debt` (3 candidates, all live-verified) and
  `short_term_investments` (1 candidate, live-verified for only 1 of 3 filers — see that module's
  docstring for why no fallback was added rather than guessed). The Alnylam CIK's real XBRL history
  (fetched live) showed `LongTermDebt` used through 2022 and `ConvertibleDebtNoncurrent` from 2025
  onward with a clean non-overlapping split — direct evidence that a "merge every candidate's
  entries" design is necessary (a "first tag with data, stop" design would have silently dropped the
  newer tag's history). New `DURATION_DELTA_TAG_MAP` + `extract_quarterly_delta_series` derive
  `quarterly_opex_burn` from `NetCashProvidedByUsedInOperatingActivities`'s cumulative-YTD entries —
  live-verified against real Moderna data (fetched fresh) that the entries are genuinely cumulative
  (`start` always the fiscal year's first day), confirming the quarter-delta derivation this session
  had previously only flagged as theoretically necessary is actually required, and building it: group
  by `start`, sort by `end`, difference consecutive cumulative values.
- `features/context.FeatureContext.get_trailing_average()` + `features/block_c.cash_runway_months`
  rewired to use it. Block C, not the normalizer, is where `quarterly_opex_burn`'s raw signed XBRL
  delta (negative = cash used) gets turned into a burn magnitude — kept consistent with
  `financial_facts.py`'s own stated boundary ("this module does not reinterpret the sign... that's
  Block C's job").

**Deliberately not attempted this pass, and why:**

- **`p22_acquirers.yaml`'s actual curation** (real entry/exit dates, real CIKs) — genuinely needs a
  domain reviewer, per `docs/Tasks.md` item 3; the loader built above only needed the file's current
  (draft) contents to exist as `p22_company` rows, which is a different question.
- **`exclusivity.txt` normalization** — `Exclusivity_Code` -> the 4-value `exclusivity_type` enum is a
  domain-classification decision (dozens of real-world codes, no clean spec-given mapping), the same
  character of decision as therapeutic-area classification. Not attempted; see
  `ingest/patent_expiry_normalization.py`'s docstring.
- **Fuzzy/review-queued patent-applicant matching** — would need `review_queue.py`'s confirm dispatch
  extended to know how to write a `p22_patent_expiry` row from a confirmed review item. A real,
  contained gap, not a decision — just not built this pass; deterministic-only matching is safe
  (lower recall, but nothing wrong gets written) in the meantime.
- **`rd_expense`** (`ResearchAndDevelopmentExpense`) — live-verified present for all 3 filers, trivial
  to add, but no built feature reads R&D spend yet. Not added speculatively; see `financial_facts.py`'s
  docstring.
- **`p22_asset` population, Blocks A/B/D/E/F** — unchanged from §4.5; still blocked on the items
  logged there and in `docs/Tasks.md`.

## 4.7 Decisions walkthrough (2026-08-31)

Per the user's request to work through `docs/Tasks.md` "Decisions needed" step by step. Outcomes:

- **Item 1 (vendor)** — decided: FMP. Web search found Basic/Starter both capped at ~5yr history,
  only Premium unlocks 30yr (spec's "Starter ~$15/mo" recommendation looks stale). Bulk-backfill
  infrastructure built ahead of the account decision, at user request (`ingest/fmp_client.py`,
  `fmp_universe.py`, `fmp_backfill.py`, `cli/fmp_backfill_cli.py`) — live-verified against the
  account's real active key in the process, catching 2 real bugs (a dead endpoint URL; a
  multi-exact-name-match ticker bug) before they shipped, and surfacing a NEW, more confusing finding:
  the account can fetch some symbols (MRNA, PFE) but gets 402 on others (AMGN, GILD, SRPT) across
  every date range — looks like a per-symbol entitlement list, not a date cap, and needs checking
  against the FMP dashboard directly. See docs/Tasks.md item 1 for the full detail.
- **Item 2 (base rates)** — mostly resolved: found the actual study is freely downloadable (a newer
  edition of the same BIO/Biomedtracker lineage spec §4.2 cites), downloaded and read it directly,
  filled 15/21 `by_therapeutic_area` entries with real cited figures. 6 remain `null` by user decision
  (genuinely no match in this report).
- **Item 3 (acquirer curation)** — resolved: CIKs live-verified for 22/25 (3 confirmed to have none),
  list grown from 22 to 25 with 3 user-approved additions, entry/exit dates accepted as "good enough"
  given none of the 25 has been acquired.
- **Item 4 (taxonomy review)** — resolved as a side effect of item 2 (same source).
- **Item 6 (IBKR)** — confirmed genuinely blocked on the user's environment (needs a live TWS/Gateway
  connection); deferred to M6/M7 by user decision, not re-investigated further this pass.
- **Item 7 (CVR policy)** — resolved: `config/pipeline/p22_cvr_policy.yaml` created early, at user
  request, with spec's own recommended v1 convention.
- **Item 8 (asset linkage)** — partially resolved: single-intervention trials now link to a real
  `p22_asset` row via a new best-effort therapeutic-area classifier; multi-intervention trials remain
  unlinked (harder case, not attempted).

Only item 5's already-noted "financial-fact tag mapping" resolution (from the prior pass) and these
outcomes needed no further discussion — every other item had a genuine user decision embedded in it.

## 4.8 FMP quota exhaustion, and building the daily price job first (2026-09-01)

Live-testing the FMP backfill CLI against the real account surfaced a genuine quota wall (`429
Limit Reach`, no rate-limit headers, persisted through a 90s cooldown) — not the per-second pacing
issue it first looked like. This, combined with the earlier per-symbol `402` discovery, means the
account's real limits still need checking against the FMP dashboard directly before any bulk
backfill runs (see item 1). No further FMP API calls were made once this was found.

Separately, the user clarified the intended sequencing: build the **daily current-price job first**,
independent of FMP entirely, then buy FMP Premium for a one-time historical backfill afterward. This
was always the intended architecture (IBKR/yfinance for ongoing current prices, FMP only for the
delisted-ticker historical gap — see `ingest/vendor_market_data.py`), just not yet built. Built and
shipped the same day: `ingest/yfinance_client.py`, `ingest/price_ingest.py`,
`jobs/run_price_ingest.py`, registered as a real daily job. yfinance was picked over the
originally-planned IBKR specifically because it needs no live broker session and its raw-vs-adjusted
behavior is checkable without one — and checking it live surfaced the same retroactive-split-
adjustment trap suspected for IBKR (item 6), confirmed here rather than left theoretical, with a
design (narrow trailing-window fetches only) that sidesteps it. Also found and fixed a real bug the
same day: `upsert_acquirer_company` crashed on a real CIK collision (Bristol-Myers Squibb) the first
time it ran against real curated CIKs — see `docs/Tasks.md`'s Implementation Status entry. The full
job then ran successfully against the real, now-migrated-and-populated local DB: 860 companies, 4,003
price rows, 6 corporate actions, 16 genuine failures (delisted tickers).

## 5. Open items carried into M2+/M3+

- **Vendor selection** (§2.4) — blocks `dilution_gate`, Block A/C, and the entire backtest (§0.3). Tracked in `docs/Tasks.md`.
- **Historical ticker/exchange resolution** (§2.0.2) — needs a new per-filing iXBRL cover-page fetch/parse
  capability; see §4.3's negative finding. Not scheduled to a milestone yet.
- **SC 14D9 / DEFM14A / S-4 EdgarDownloader support** — scoped to M6 (§2.1 table above).
- **IBKR split-adjustment behavior** (§2.0.7 above) — needs live verification before M6/M7 price-archive ingest is built against it.
- **Size/asset floor eligibility filters, per-quarter universe persistence, deal cross-reference** — see §4.2/§4.3 above; all blocked on later milestones (vendor decision, M3 trial normalization, M6 deal data), not on more M2 effort.
- **Blocks A/B/D/E/F, `p22_trial`/`p22_asset`/`p22_patent_expiry` normalization, real base-rate data,
  acquirer-config curation, rest of `FACT_TAG_MAP`, lookahead audit wired to real data** — see §4.4
  above and `docs/Tasks.md` "Decisions needed" for the full, consolidated list.
