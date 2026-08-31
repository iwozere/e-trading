# Design

## Purpose
P22 ranks US-listed biotech companies by likelihood of acquisition within 12–24 months, and
explains why. Full rationale in `docs/pipeline-specification.md` §0. This document covers the
*implementation* architecture — the specification is normative for behavior, this file is
normative for how that behavior maps onto this repo's code layout.

## Architecture

### High-level flow (M1 + M2 + M3-so-far slice)

```
   EdgarDownloader        ClinicalTrialsClient    OpenFDAClient   OrangeBookClient   p22_acquirers.yaml
   (submissions,          (CT.gov API v2,         (Drugs@FDA)     (quarterly ZIP)    (hand-curated,
    XBRL company facts)    version history)                                          spec §2.0.4)
        │                       │                     │                │                  │
        └───────────────────────┴─────────┬───────────┴────────────────┘                  │
                                           │                                                │
                                  ingest/raw_zone.py                                        │
                            (content-addressed, gzip JSON,                                  │
                             DATA_CACHE_DIR/p22/raw/<source>/<date>/)                        │
                                           │                                                │
    ┌───────────────┬──────────────────────┼──────────────────┬───────────────────┐        │
    │               │                      │                  │                   │        │
 entity_       run_alias_          financial_facts.py   trial_normalization.py  patent_expiry_    │
 resolution.py matching.py         reads sec_company_    reads clinicaltrials_  normalization.py   │
 build_universe/  reads CT.gov/    facts, extracts        studies + p22_company  reads Orange Book  │
 write_universe   openFDA +        cash/debt/shares/      (by CIK) — asset_id    products+patent,   │
    │             p22_company,     burn (tag fallback     always None (see       matches Applicant  │
    ▼             resolve_aliases  merge + quarter-        Design Decisions)     _Full_Name against  │
p22_company          │             delta derivation —        │                  acquirer roster     │
                     ▼             see Design Decisions)      ▼                 (deterministic only) │
           p22_company_alias /            │              p22_trial                    │             │
           p22_review_item                ▼                                           ▼             │
                                  p22_financial_fact                          p22_patent_expiry       │
                                           │                                                          │
                                           ▼                                     run_acquirer_load.py ◄┘
                               features/context.py (FeatureContext,             loads acquirer IDENTITY
                               incl. get_trailing_average)                      into p22_company only —
                               features/block_c.py (spec §4.3, tested;          bloc/entry_date/exit_date
                               cash_runway_months/dilution_risk's runway        stay in the config file
                               leg now real — see below)                       (see Design Decisions)
```

Blocks A, B, D, E, F, G (scoring, dossiers) are M3-partial/M4+ and not present yet — see
`implementation-plan.md` §4 for milestone status. `p22_trial`, `p22_patent_expiry`, and the acquirer
roster are all built (above); `p22_asset` is not — blocked on a real CT.gov-intervention-to-asset
linkage decision, `docs/Tasks.md` "Decisions needed" item 8.

### Component design

- **`ingest/raw_zone.py`** — the single write path for landing external payloads. Enforces:
  immutability (files are never overwritten — content hash is part of the path), idempotency
  (`(source, entity, as_of_date)` re-fetch produces the same hash and is a no-op — spec §7.3), and
  partitioning (`source/date/`). Every other ingest client calls into this rather than writing
  files directly.
- **Per-source clients** (`clinicaltrials_client.py`, `openfda_client.py`, `orange_book_client.py`,
  `purple_book_client.py`) — each owns exactly one external API's request/parse/retry logic and
  returns plain dicts/lists to the caller; they do not touch the DB or the raw zone directly. Job
  scripts (`jobs/run_*.py`) are the glue: client → raw_zone write → (M2+) repo write.
- **`ingest/vendor_market_data.py`** — a `Protocol` (`MarketDataProvider`) defining the bitemporal
  contract spec §2.4 requires (`known_from` semantics, restatement-as-new-row), plus
  `NullMarketDataProvider`, which raises `NotImplementedError`. This exists so M1–M2 code that will
  eventually depend on vendor data (e.g. future `dilution_gate` computation) can be written against
  a stable interface now, with the real adapter swapped in later without touching call sites.
- **`P22Repo`** (`src/data/db/repos/repo_p22_biotech_ma.py`) — the only code path allowed to write
  `p22_*` tables. `upsert_financial_fact_bitemporal()` is the generic restatement-safe write: it
  closes the prior row's `valid_to` and inserts a new row, never an in-place `UPDATE ... SET value`,
  because an in-place update on a bitemporal table destroys the point-in-time record the whole
  system's correctness guarantee (spec §3.1) depends on. `upsert_price_daily()`/`get_adjusted_close()`
  are the price-archive equivalent (see below).
- **`ingest/price_archive.py`** (spec §2.0.7, added v0.6) — pure split-adjustment math, deliberately
  kept free of any DB dependency so it's unit-testable on its own. `P22Repo.get_adjusted_close`
  combines it with a live raw-price + corporate-action lookup (a local, function-scoped import inside
  the repo method — see that file's top-of-module comment — rather than a module-level import, mirroring
  the one existing precedent for `src/data/db` referencing `src/ml/pipeline` code, `model_short_squeeze.py`).
  Raw price rows are never rewritten: `upsert_price_daily` uses `ON CONFLICT DO NOTHING`, not
  `DO UPDATE`, enforcing the "as traded, never rewritten" rule at the DB layer rather than trusting
  every future call site to honor it.
- **`ingest/entity_resolution.py`** / **`ingest/alias_matching.py`** (spec §2.0.2-3, §3.3, M2) — turn
  landed DERA rows into `p22_company` (including `build_universe_history()`, the per-quarter point-in-time
  re-computation spec §2.0.3 requires), and external sponsor/applicant name strings
  (`jobs/run_alias_matching.py`, reading the latest landed `clinicaltrials_studies`/`openfda_drugsfda`
  raw-zone partitions paired with their manifest data via `raw_zone.read_latest_partition_with_manifest`,
  and the roster via `P22Repo.list_companies`) into `p22_company_alias`. Both route anything not
  confidently resolved to `p22_review_item` rather than silently dropping or silently guessing — a
  name-based SPAC heuristic isn't authoritative enough to exclude a company outright, and a fuzzy alias
  match is explicitly "never auto-accepted" per spec §3.3.
- **`ingest/review_queue.py`** / **`cli/review_queue_cli.py`** (spec §3.4, M2) — the review-queue
  confirm/reject logic and its human-run CLI. `confirm_item()` dispatches on `payload['reason']` to the
  correct downstream write and threads `known_from` from the *original* candidate observation, never the
  review timestamp (spec §3.4's explicit warning: getting this backwards "silently destroys the
  backtest"). Kept as a separate `cli/` directory rather than `jobs/`, since this is interactively
  human-run, not scheduler-invoked — it doesn't print `__SCHEDULER_RESULT__` and has no cron entry.
- **`ingest/financial_facts.py`** (spec §2.1, §3.1, M3) — normalizes landed SEC XBRL `companyfacts`
  payloads into `p22_financial_fact` bitemporal rows. Deliberately narrow `FACT_TAG_MAP` (2 metrics),
  live-verified across multiple real filers rather than assumed from tag names alone — see that
  module's docstring and `docs/Tasks.md` "Decisions needed" for the rest of the mapping work. Guards
  against a real trap found while building it: XBRL re-reports an unchanged prior period's balance as
  a comparative column in every later filing, which naive re-processing would misdate as newly known
  on the later filing.
- **`features/context.py` / `features/registry.py` / `features/block_c.py` / `features/quality.py` /
  `features/lookahead_audit.py`** (spec §4, §8.2, §8.3, M3) — the feature-store scaffolding and the
  first fully-implemented block. `FeatureContext` is the single point where every feature function
  reads the store, so lookahead safety (spec §3.1/§8.3) is enforced once, not per feature. Block C's 6
  functions are complete and tested against both the real-computation and null paths (spec §8.1); most
  return `None` today purely because their inputs (`market_cap`, quarterly burn) aren't normalized
  into the store yet, not because the feature logic is incomplete. `quality.py` and `lookahead_audit.py`
  are similarly complete-but-not-yet-wired-to-real-data — see Design Decisions below for why that's a
  deliberate choice, not an oversight.

### Error handling

- Every failed external fetch, after retries, is logged to `p22_fetch_failure` (spec §7.2). No
  ingest job may swallow a failure silently — `raw_zone.write()` raises on unrecoverable I/O errors
  and the calling job script is responsible for catching, logging to `p22_fetch_failure`, and
  continuing with the remaining work rather than aborting the whole run.
- Rate-limit backoff (429/403/5xx) is handled inside each client via
  `src.data.utils.rate_limiting.RateLimiter`, matching spec §7.2's exponential-backoff-with-jitter,
  max-5-attempts requirement.

## Data Flow
- **Input:** SEC EDGAR (submissions, XBRL facts), SEC DERA (quarterly `sub.txt`, universe basis),
  ClinicalTrials.gov (study records + version history), openFDA (Drugs@FDA approvals), FDA Orange
  Book / Purple Book (quarterly product/patent/exclusivity data).
- **Landing:** raw, immutable, gzipped JSON under `DATA_CACHE_DIR/p22/raw/`.
- **Normalization (M2, partial):** `entity_resolution.build_universe()`/`write_universe()` read the
  latest landed DERA snapshot and write `p22_company`. `jobs/run_alias_matching.py` reads the latest
  landed CT.gov/openFDA payloads and `p22_company`, and calls `alias_matching.resolve_aliases()`.
  Everything else (financial facts, trials, patent expiry, deals, Block G sources) still needs its own
  M2+/M3 normalization path into the bitemporal `p22_*` tables via `P22Repo`.
- **Price archive (schema only, v0.6):** `p22_price_daily`/`p22_corporate_action` exist and
  `P22Repo.get_adjusted_close` is fully implemented and tested, but nothing writes real price data
  yet — blocked on the deferred market-data vendor decision (§2.4).
- **Financial facts (M3, partial):** `ingest/financial_facts.py` normalizes 2 metrics
  (`cash_and_equivalents`, `shares_outstanding`) from landed `sec_company_facts` into
  `p22_financial_fact`. Everything else Block A-C need (market cap, short-term investments, debt,
  quarterly burn) is either vendor-blocked or needs more tag-mapping work — see `docs/Tasks.md`.
- **Features (M3, partial):** `features/block_c.py`'s 6 functions read through `FeatureContext` and
  are ready to compute real values the moment their inputs exist; today only `cash_and_equivalents`-
  dependent intermediate math is exercised (still `None` end-to-end, since every Block C output needs
  at least one metric not yet normalized). Blocks A, B, D, E, F are not started.

## Design Decisions

- **Postgres over DuckDB** (spec offered either) — this repo already runs Postgres with a mature
  Alembic/SQLAlchemy layer; introducing DuckDB for one pipeline would be a second warehouse engine
  for no benefit. See `implementation-plan.md` §1.
- **Custom DB-backed scheduler over Prefect/Dagster** (spec §7.1) — matches every other pipeline in
  this repo (P15, P18, P20). See `implementation-plan.md` §1.
- **`p22_*` table prefix**, mirroring P20's `k20_*` — keeps pipeline-owned tables visually grouped
  and makes `grep`-ability of "what does P22 own" trivial.
- **Bitemporal writes are centralized in one repo method**, not reimplemented per feature/table,
  because this pattern is new to the codebase (no other pipeline needed it) and a second, slightly
  different implementation of "close `valid_to`, insert new row" is exactly the kind of subtle
  divergence that produces an undetected lookahead leak (spec §3.1, §8.3).
- **Vendor adapter deferred behind a `Protocol`**, not stubbed out entirely — per 2026-08-30
  decision to pick a market-data vendor later rather than block M1 schema/ingest work on a
  procurement decision.
- **Raw price storage, read-time adjustment, never write-time adjustment** (spec §2.0.7, added
  v0.6) — adjusting at write time makes every split a retroactive rewrite of the whole history for
  that ticker, which corrupts point-in-time market cap and leaks future split information into past
  `as_of` decisions. The adjustment math lives in a DB-free pure module (`ingest/price_archive.py`)
  specifically so its lookahead guard is unit-testable without standing up Postgres.
- **Ambiguous entity-resolution/eligibility signals go to the review queue, not silently dropped or
  silently accepted** (spec §3.3, §3.4) — a name-based SPAC heuristic and a sub-100 fuzzy alias match
  are both "candidate generators, not classifiers" in the spec's own words (§2.6.1 uses that phrase for
  a different signal, but the principle is general here too). M2's `write_universe`/`resolve_aliases`
  apply it consistently: confident matches write directly, everything else queues for a human, nothing
  vanishes.
- **Review-item payloads carry everything a later confirmation needs to write correctly, not just
  enough to display** — `known_from` (the raw-zone ingestion timestamp) for alias candidates,
  ticker/exchange/reporting-eligibility for SPAC candidates. The alternative — re-deriving those values
  at confirm time — would either re-run the same lookups (and risk them disagreeing with what the
  candidate was actually built from) or silently fall back to "now," which is precisely the bug spec
  §3.4 warns against by name.
- **Per-quarter universe history (`build_universe_history`) is computed but not persisted** — no
  consumer (the M6 backtest harness) exists yet to define the right storage shape for a point-in-time
  series, and the spec's own §2.0.3 text doesn't specify one beyond "the per-quarter set is the eligible
  universe for that `as_of`." Persisting speculatively risks building a schema M6 then has to migrate
  away from; the pure computation is built, tested, and ready to be wired into a job once M6 defines what
  it needs.
- **A feature function never special-cases "the data isn't wired up yet"** (spec §4: "returning `None`
  is meaningful and must propagate as missing, never as zero") — `features/block_c.py` reads whatever
  metrics exist via `FeatureContext.get_latest_fact`, and `None` propagates automatically when a
  metric hasn't been normalized into the store (e.g. `market_cap`, blocked on the vendor decision).
  This means a feature function is correct and finished the moment its logic is right, independent of
  whether upstream data happens to exist yet — no `if not VENDOR_MARKET_DATA_AVAILABLE: return None`
  branches scattered through feature code, and no risk of that branch going stale once the vendor
  finally is wired up.
- **Config YAMLs the spec calls for (`p22_acquirers.yaml`, `p22_therapeutic_area.yaml`,
  `p22_modality.yaml`, `p22_base_rates.yaml`) are drafted, not fabricated-to-look-complete** —
  spec §4.2 explicitly warns that "shipping with 2 of ~20 therapeutic areas calibrated... mis-scores
  90% of the universe," and several of these files require either a primary source this session
  doesn't have (the BIO/Informa base-rate study) or genuine curatorial judgment the spec itself asks
  a human to make (acquirer entry/exit dates). Every incomplete or unverified field is `null` or
  flagged in a header comment, never filled with a plausible-looking placeholder — a wrong-looking
  gap is safe (it's visible and will trigger `base_rate_fallback`/similar); a right-looking guess is
  not, because nothing downstream would know to distrust it. See `docs/Tasks.md` "Decisions needed."
- **The lookahead audit's sampling/assertion logic (`features/lookahead_audit.py`) is built now but
  not wired to real data or CI** — spec §8.3 requires the sample be *stratified* across three named
  high-risk categories (vendor facts, 13F, 13D/process events), none of which have any rows in this
  repo yet. Running the "mandatory... blocks the build" gate against an empty population for those
  categories would silently pass without ever having tested the thing it exists to catch — a false
  sense of the safety property being verified is worse than an honestly-absent gate. Wire it in once
  vendor/13F/13D ingestion lands (M5/M6+), not before.
- **`p22_trial` is a plain upsert keyed on `nct_id`, deliberately not bitemporal** like
  `p22_financial_fact` — a trial has one current state (status/enrollment/phase evolve as CT.gov
  re-fetches land, and only the latest is useful for scoring), and the actual "what changed and when"
  signal spec §2.2 asks for lives in the already-landed, still-unconsumed `clinicaltrials_history`
  raw-zone source. Building a second bitemporal chain for `p22_trial` would duplicate machinery that
  doesn't serve this table's actual access pattern.
- **`p22_trial.asset_id` is written `None` unconditionally, not linked by a "first DRUG intervention"
  heuristic** — live-checked against a real multi-drug study (Vertex/Moderna cystic-fibrosis trial
  listing both the asset under test and an approved comparator as DRUG-type interventions), CT.gov has
  no field marking which intervention is the sponsor's own pipeline asset. A positional heuristic would
  sometimes be right by luck and sometimes silently wrong, feeding Block B a fabricated-looking but
  incorrect asset link — worse than the honest gap. See `docs/Tasks.md` "Decisions needed" item 8.
- **Acquirer-config loading is split into "identity" (mechanical, built now) and "curation" (a real
  decision, still open)** — whether the ~21 companies `p22_acquirers.yaml` names exist as
  `p22_company` rows is unrelated to whether their entry/exit dates are accurate. `bloc`/`entry_date`/
  `exit_date` are read by `ingest/acquirer_config.py` but never written to a DB column — no such
  column exists in the spec's schema, and inventing one to store obviously-placeholder values would
  make the placeholder-ness less visible, not more. They stay in the config file for Block A to read
  directly once built, at which point the curation gap becomes impossible to miss.
- **`upsert_acquirer_company` merges by ticker and checks for an existing row under ANY role, not
  just an unmatched one** — a large-cap acquirer is routinely ALSO a DERA-resolved `target`-role
  company (its own SIC code qualifies it). Matching only against `cik IS NULL` rows would have missed
  that overlap and created a second, `cik`-less identity for a company that already has one with a
  real `cik` — silently fragmenting that company's data across two `company_id`s. Checking by ticker
  regardless of existing role and merging to `role='both'` avoids that.
- **`extract_fact_series`'s tag-fallback merges every candidate's entries rather than stopping at the
  first with data** — live XBRL data from a real filer (Alnylam) showed exactly why: it reported
  `LongTermDebt` through 2022 and switched to `ConvertibleDebtNoncurrent` from 2025, with no overlap.
  "First candidate with any data" would have picked `LongTermDebt` and silently discarded three years
  of more-recent debt history. The same dedup/restatement-warning logic that already guards the
  single-tag case covers a genuine same-period conflict between two merged tags for free, since it
  operates on periods, not tag identity.
- **`quarterly_opex_burn` stores the raw signed XBRL delta; Block C, not the normalizer, flips the
  sign into a burn magnitude** — keeps the layering `ingest/financial_facts.py`'s own docstring states
  explicitly: normalizers extract and derive what the source data says, feature functions interpret it
  for a specific formula's needs. A different future feature reading the same metric might want the
  signed value directly, which a pre-flipped stored value would make awkward to answer.
- **Orange Book patent-applicant matching is deterministic-only, not fuzzy** — unlike CT.gov/openFDA
  sponsor-alias matching, a fuzzy match here is logged and dropped, not routed to the review queue.
  Queuing it safely would require `review_queue.py`'s confirm dispatch to know how to write a
  `p22_patent_expiry` row from a confirmed item, which doesn't exist yet, and writing a fuzzy match
  directly without review was rejected for the same reason it's rejected everywhere else in this
  build. Deterministic-only, with low recall until the queue is extended, was judged the safer gap.
- **Asset linkage is split by ambiguity, not attempted all-or-nothing** — a trial with exactly one
  DRUG/BIOLOGICAL intervention has no ambiguity about which intervention is the sponsor's own asset;
  a trial with multiple does (the Vertex/Moderna VX-522+IVA case). Rather than leaving `asset_id`
  unconditionally `None` until the hard case is solved, `ingest/asset_normalization.py` (user-approved
  2026-08-31) resolves the unambiguous subset now. This is the same shape of decision as deterministic-
  vs-fuzzy alias matching: ship what's genuinely safe, leave the genuinely uncertain case explicitly
  unhandled rather than blocking on it or guessing at it.
- **The therapeutic-area classifier is a disclosed best-effort heuristic, with its own fallback value
  in the controlled vocabulary (`unclassified`)** — rather than either fabricating a confident-looking
  classification or leaving `p22_asset.therapeutic_area` unfillable (it's `NOT NULL`), a keyword
  classifier does its best and openly says when it couldn't. `unclassified` is documented in
  `p22_therapeutic_area.yaml` as a classifier-fallback marker, not a 21st real disease area — so
  downstream code and reviewers can filter on it to find candidate misclassifications rather than
  silently trusting every asset's therapeutic area.

## Integration Patterns
- Job scripts follow the P20 contract exactly: `PROJECT_ROOT` on `sys.path`, `run_common.py`'s
  `setup_run_logging()`, a `run()` function, `print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")`
  on success — so the existing scheduler's subprocess harness needs zero changes to run P22 jobs.
- `P22Repo` is added as a field on `ReposBundle` (`database_service.py`), following the exact
  pattern `kestrel: KestrelRepo` already uses — no changes to the `uow()` context-manager contract.
