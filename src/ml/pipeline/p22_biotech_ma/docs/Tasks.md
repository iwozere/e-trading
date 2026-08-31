# Tasks

## Decisions needed (walk through with a domain reviewer before relying on the affected features)

These are business/domain calls this session could not make from the code or spec text alone —
either they need real published data this session doesn't have access to, or they're genuine
curatorial judgment calls the spec itself says should be hand-made. Each is flagged at its point of
use in the code/config; this section exists so they're all in one place to walk through, per request
2026-08-30. Ordered roughly by how much they block.

1. **Market-data vendor selection** (spec §2.0.6/§2.4) — blocks `market_cap` everywhere it's used:
   Block A's entire capacity model, Block C's `enterprise_value`/`ev_to_cash`/`size_band`/
   `atm_capacity_pct`, and the whole backtest. **Decided 2026-08-31: FMP.** Spec's own recommendation
   (Starter, ~$15/mo), and confirmed the same day that this repo already has a real, working FMP
   integration (`src/data/downloader/fmp_data_downloader.py` — `FMPDataDownloader`, already used by
   P20 Kestrel, P05, and the Telegram screener), so P22's vendor adapter should wrap/reuse that class
   rather than build a new client from scratch.
   **Not yet unblocked — waiting on the user to check the account, not a decision anymore:**
   `FMP_API_KEY` is already configured in production, but `p20_kestrel/ingest/revisions_ingest.py`'s
   own comment notes `period=quarter` on the analyst-estimates endpoint "returns 402 Payment Required
   ... on the account's current plan" — a real signal the currently-active tier is limited, and
   delisted-ticker historical price/market-cap data (spec's whole reason for picking FMP) is a
   paid-tier feature that may not be covered. User is checking/upgrading the existing FMP plan.
   Next step once confirmed: write a thin `MarketDataProvider` adapter over `FMPDataDownloader`
   (`ingest/vendor_market_data.py`'s `NullMarketDataProvider` gets replaced) and run the
   delisted-ticker validation against ~20 known acquisitions spec calls for, live, before trusting it
   for real scoring.
2. ~~**`config/pipeline/p22_base_rates.yaml` is ~90% incomplete**~~ — **mostly resolved 2026-08-31.**
   The actual primary source turned out to be freely available: "Clinical Development Success Rates
   and Contributing Factors 2011-2020" (BIO, QLS Advisors, Informa UK Ltd, Feb 2021) — a newer,
   free-to-download edition of the same BIO/Biomedtracker study lineage spec §4.2 cites
   (https://go.bio.org/rs/490-EHZ-999/images/ClinicalDevelopmentSuccessRates2011_2020.pdf, verified
   no paywall). Downloaded and read directly; `by_therapeutic_area` now has 15 of 21 entries filled
   from this report (up from 2), each citing the exact figure/table it came from. **Still open, by
   user decision 2026-08-31 (not filled with a guess):** 6 areas this report has no usable number for
   at all (`immunology`, `cardiometabolic`, `rare_metabolic`, `gene_cell_therapy`,
   `rare_orphan_disease`, `vaccines` — see that file's header for why each has no match). The 5
   top-level aggregate rates (`loa_from_phase_1_overall` etc.) were deliberately left untouched —
   still spec's originally-quoted figures, not this report's — a separate decision not yet asked
   about. `base_rate_fallback` will still fire for the 6 remaining null areas and for
   `gene_cell_therapy`/platform-classified assets generally.
3. ~~**`config/pipeline/p22_acquirers.yaml` needs real curation**~~ — **resolved 2026-08-31.** CIKs
   live-verified for 22 of 25 acquirers against SEC's own registries (the other 3 — Roche, Bayer,
   Astellas — genuinely have no SEC CIK, confirmed by an empty EDGAR company search, not an
   unverified gap). List grown from 22 to 25 (matching spec's "~25" target exactly) with 3
   user-approved additions (Astellas, Daiichi Sankyo, CSL), each with public deal-history rationale
   in the file's comments. Entry/exit dates: by user decision, `exit_date: null` is accepted for all
   25 (none has been acquired) and the 2010-01-01 `entry_date` placeholder is accepted as "good
   enough for now" rather than individually researched (all are mega-caps plausibly already at
   acquirer scale by 2010; AbbVie already has its real 2013 spinoff anchor). The loader that turns
   this file into `p22_company` roster rows (`ingest/acquirer_config.py` + `jobs/run_acquirer_load.py`)
   was already built the same day, earlier — see Implementation Status. **What's still open:** Block A
   itself can't run yet regardless — `cash_capacity`/`equity_capacity` need real market-cap data,
   i.e. item 1.
4. ~~**`config/pipeline/p22_therapeutic_area.yaml` needs domain review against the study's taxonomy**~~
   — **done 2026-08-31**, as part of item 2's resolution (the study is now available — see item 2).
   No vocab values needed adding/removing; the mapping decision itself lives in
   `p22_base_rates.yaml`'s comments. **`p22_modality.yaml` still needs domain review** — it has no
   spec-given values to start from at all, and the BIO study above is organized by disease area, not
   modality, so it doesn't help here either. Genuinely still open.
5. ~~**Financial-fact tag mapping is incomplete**~~ — **resolved 2026-08-30, later same day.** Not
   actually a business decision — it needed live verification, not domain judgment, and that's now
   done: `total_debt` (fallback chain across `LongTermDebtNoncurrent`/`LongTermDebt`/
   `ConvertibleDebtNoncurrent`, merged not first-wins — live data showed Alnylam migrated between two
   of these tags mid-history) and `quarterly_opex_burn` (derived via quarter-delta from XBRL's
   cumulative YTD duration contexts, `extract_quarterly_delta_series`) are both live-verified and
   built; `cash_runway_months`/`dilution_risk` now compute real values. `short_term_investments` is
   live-verified for only 1 of 3 filers checked (Sarepta) — Moderna/Alnylam report neither it nor
   spec's suggested alternative in the periods checked, and rather than fabricate a fallback tag,
   `None` is correct for a company with no live-verified match. See Implementation Status for detail.
6. **IBKR split-adjustment behavior** (spec §2.0.7) — still not live-verified; confirmed 2026-08-31
   this needs a real TWS/Gateway connection on the user's own machine, not something verifiable
   remotely — user doesn't have it running right now. Genuinely blocked on the user's environment,
   not on more investigation; verify when M6/M7 price-archive ingest is actually being built, closer
   to when TWS/Gateway would need to be running anyway. See "Known Issues" below.
7. ~~**CVR valuation policy**~~ — **done 2026-08-31**, at the user's request, ahead of its natural M6
   milestone. `config/pipeline/p22_cvr_policy.yaml` created with spec's own recommended v1 convention
   (value CVRs at zero, `upfront_per_share` as the return basis). Not wired to any code yet (M6
   doesn't exist) — same as every other `config/pipeline/*.yaml` file today.
8. ~~**CT.gov intervention -> company-asset linkage isn't a solved mechanical mapping**~~ — **partially
   resolved 2026-08-31**, by user decision, to the safe subset: a trial with **exactly one**
   DRUG/BIOLOGICAL intervention has no ambiguity (there's only one candidate), so
   `ingest/asset_normalization.py` now resolves/creates a `p22_asset` and links `p22_trial.asset_id`
   for those. Multi-intervention trials (the original Vertex/Moderna VX-522+IVA example) are still
   left unlinked — genuinely unsolved, not attempted. The `p22_asset.therapeutic_area` `NOT NULL`
   blocker is also addressed for this subset: `ingest/therapeutic_area_classifier.py`, a best-effort
   keyword classifier over CT.gov `conditions` text, with an explicit `unclassified` fallback (added
   to `p22_therapeutic_area.yaml`) rather than a forced guess. **This classifier is disclosed as
   imperfect, not validated against real clinical taxonomy** — any asset it classifies should be
   treated as a candidate classification pending review, not ground truth, especially before it feeds
   a real Block B computation that branches on therapeutic area.

## Implementation Status

### ✅ COMPLETED FEATURES (M1)
- [x] `docs/implementation-plan.md` — reuse map, spec/repo deviations resolved
- [x] Bitemporal Postgres schema (`p22_*` tables, spec §3.2 + `p22_review_item` + `p22_fetch_failure`)
- [x] Alembic migration `003_p22_biotech_ma_schema.py`
- [x] `P22Repo` with generic bitemporal restatement-safe write (`upsert_financial_fact_bitemporal`)
- [x] Raw-zone writer (`ingest/raw_zone.py`) — content-addressed, immutable, partitioned
- [x] SEC EDGAR raw-zone landing (submissions + XBRL company facts, via `EdgarDownloader`)
- [x] ClinicalTrials.gov API v2 client + raw-zone landing
- [x] openFDA Drugs@FDA client + raw-zone landing
- [x] FDA Orange Book client (quarterly ZIP) + raw-zone landing
- [x] FDA Purple Book client (quarterly CSV) + raw-zone landing
- [x] SEC DERA Financial Statement Data Sets client (quarterly `sub.txt`) + raw-zone landing (spec §2.0, added v0.5 — universe construction basis)
- [x] Market-data vendor adapter `Protocol` + `NullMarketDataProvider` stub
- [x] Job scripts + `register_jobs.py` for all M1 ingest sources
- [x] Shared GET-with-retry helper (`ingest/http_retry.py`) — retries only 429/5xx
- [x] Every M1 client live-verified against its real source (not just mocked), 2026-08-30 — see
      `docs/implementation-plan.md` §4.1 for the corrections this surfaced
- [x] Price archive schema (`p22_price_daily`, `p22_corporate_action`) + read-time split-adjustment
      (spec §2.0.7, added v0.6) — migration `004_p22_price_archive`, pure math in `ingest/price_archive.py`,
      `P22Repo.get_adjusted_close`. Ingest job itself still blocked on the vendor decision (§2.4).

### 🔄 M2 Entity resolution — blocked on later milestones, not more M2 work
- [x] `p22_company` roster build from landed DERA rows — reporting-status eligibility, current
      ticker/exchange resolution, name-based SPAC heuristic (flagged to review queue, not auto-dropped).
      `ingest/entity_resolution.py`, `jobs/run_entity_resolution.py`, registered in `register_jobs.py`.
- [x] Alias matching (spec §3.3): deterministic normalized-name match + `rapidfuzz` token-set-ratio ≥ 88
      fuzzy match routed to the review queue. `ingest/alias_matching.py`, fully unit-tested.
- [x] Alias matching wired into a job (2026-08-30): CT.gov `leadSponsor.name` / openFDA `sponsor_name`
      field paths live-verified against the real APIs first (same discipline as §4.1's M1 corrections —
      see `ingest/alias_matching.py`'s module docstring for what was found, including a CT.gov sponsor
      name that was a merger-notice sentence rather than a clean company name). `jobs/run_alias_matching.py`
      reads the latest landed `clinicaltrials_studies`/`openfda_drugsfda` raw-zone partitions
      (`raw_zone.read_latest_partition`, new generic helper) and `p22_company` (new `P22Repo.list_companies`),
      calls `resolve_aliases` per source, and is registered in `register_jobs.py` (daily, after both
      ingest jobs land that day's data).
- [x] Per-quarter point-in-time re-computation of eligibility for the backtest (spec §2.0.3: "applied per
      `as_of`, not once"), 2026-08-30 — `entity_resolution.build_universe_history()` walks every landed
      DERA quarter (`universe_snapshot.all_landed_quarters()`, new — reads across *all* ingest-date
      partitions, not just the latest one) and re-derives eligibility from the cumulative union of
      filings up to and including each quarter's own end date. Pure/DB-free and fully unit-tested.
      **Deliberately not persisted anywhere yet** — no consumer (the M6 backtest harness) exists to
      define what a `p22_company_history` storage shape should look like; wiring it into a job and
      picking that shape is M6 work, not M2's.
- [x] Review-queue CLI (spec §3.4), 2026-08-30 — `ingest/review_queue.py` (`confirm_item`/`reject_item`/
      `queue_depth_report`, unit-tested against a mock repo) + `cli/review_queue_cli.py` (argparse:
      `status`/`list`/`show`/`confirm`/`reject`, run interactively by a human, not by the scheduler).
      `confirm_item` dispatches on `payload['reason']` to the correct downstream write
      (`spac_name_heuristic` -> `upsert_company`, `fuzzy_alias_candidate` -> `add_company_alias`) and
      raises rather than silently no-op-confirming an item type it doesn't recognize. Fixed a real gap
      surfaced while building this: `add_company_alias` calls in `resolve_aliases` (both the immediate
      deterministic write and the review-item payload for a later fuzzy confirm) were not setting
      `known_from` at all, defaulting it to `NULL` — the exact bug spec §3.4's "confirmation writes back
      with `known_from` set to the underlying filing date, not the review date" warns against. Now
      threaded through from the raw-zone landing timestamp (new `raw_zone.read_latest_partition_with_known_from`)
      end to end. Also added `created_at` to `p22_review_item` (migration `005_p22_review_item_created_at`)
      — the spec's own §3.4 SQL sketch omits it, but "queue depth and median age... reported in every run"
      is unanswerable without one; `run_entity_resolution.py`/`run_alias_matching.py` now log
      `queue_depth_report()` in their summary on every run.
- [x] Investigated historical ticker/exchange resolution for delisted names (spec §2.0.2), 2026-08-30 —
      **the spec's suggested approach doesn't work as described.** Live-verified against SEC's XBRL
      `companyfacts` API (what `EdgarDownloader.load_company_facts` already lands) across 3 CIKs including
      Meta (a known FB->META ticker change): `dei:TradingSymbol`/`dei:SecurityExchangeName` — and even
      `dei:EntityRegistrantName` — are **never** present in that API's aggregated `dei` facts, only
      numeric ones like `EntityCommonStockSharesOutstanding`. Those cover-page values exist only as
      inline XBRL in each filing's own HTML document. Real implementation needs per-filing document
      fetch + iXBRL cover-page parsing — new scraping infrastructure `EdgarDownloader` doesn't have
      today, not a read of already-landed data. Scoping this out until that infrastructure is built is
      a deliberate decision, not a gap nobody looked at. `eligible_exchange` stays `None` (not `False`)
      for any CIK the current-snapshot map doesn't cover, per the existing design.
- [ ] Size floor ($25M market cap) and asset floor (≥1 Phase I+ program) eligibility filters (spec
      §2.0.3) — blocked on the vendor decision (size) and on `p22_trial` existing and being linked to a
      resolved company (asset floor) — CT.gov data still only lands in the raw zone, nothing normalizes
      it into `p22_trial` yet (that's M3 work). Both fields exist on `UniverseCandidate`, explicitly
      `None`, not defaulted or guessed.
- [ ] Cross-reference roster disappearances against `p22_deal` to classify acquired/delisted/late-filing
      (spec §2.0.1) — `p22_deal` isn't populated until M6.

### 🔄 IN PROGRESS — M3 Feature store
- [x] `config/pipeline/p22_acquirers.yaml`, `p22_therapeutic_area.yaml`, `p22_modality.yaml`,
      `p22_base_rates.yaml` drafted, 2026-08-30 — all four explicitly flagged incomplete/needs-review
      in their own headers; see "Decisions needed" above. Not loaded by any code yet (no
      config-loader/`config_hash` mechanism built — that's forward-looking M4 scoring infra, not
      needed until something actually reads these files).
- [x] `p22_base_rates.yaml`'s `by_therapeutic_area` populated from a real primary source, 2026-08-31 —
      found that spec §4.2's cited study lineage has a newer, freely-downloadable edition ("Clinical
      Development Success Rates and Contributing Factors 2011-2020," BIO/QLS Advisors/Informa UK Ltd,
      Feb 2021), verified no paywall, downloaded, and read directly (its own "14 major disease areas"
      language matches spec's prose closely enough this is almost certainly the edition spec was
      written against). 15 of 21 areas now filled (up from 2), each citing its source figure; the
      remaining 6 (`immunology`, `cardiometabolic`, `rare_metabolic`, `gene_cell_therapy`,
      `rare_orphan_disease`, `vaccines`) genuinely have no usable number in this report and are left
      `null` by explicit user decision, not a guess — see that file's header. Also resolved item 4's
      taxonomy-mapping review (same source), recorded in `p22_base_rates.yaml`'s comments rather than
      `p22_therapeutic_area.yaml` itself, since no vocab value needed adding/removing.
- [x] `ingest/financial_facts.py` + `jobs/run_financial_facts_normalization.py`, 2026-08-30 — the
      first real (not synthetic-only) M3 data path: normalizes landed `sec_company_facts` XBRL
      payloads into `p22_financial_fact` bitemporal rows. Live-verified 2 tags across 3 real biotech
      filers (`cash_and_equivalents`, `shares_outstanding`) before committing to the mapping — see
      that module's docstring for the rest of `FACT_TAG_MAP`'s open scope (item 5 above). Also fixed
      a real correctness trap found while building this: XBRL re-reports an unchanged prior-period
      balance as a comparative column in every subsequent filing, which a naive re-processing would
      treat as a brand-new fact known only as of the later filing — dedup-by-period-end, keep
      earliest-filed, is now explicit and tested. Registered in `register_jobs.py`.
- [x] `raw_zone.read_latest_partition_with_manifest()` — generalized `read_latest_partition_with_known_from`
      (added for M2's alias-matching job) to surface the full manifest dict, not just `known_from`,
      since the financial-facts normalizer also needs `entity` (the CIK). `run_alias_matching.py`
      updated to the new name; behavior unchanged.
- [x] `features/context.py` (`FeatureContext`) + `features/registry.py` (`register_feature`/
      `get_feature`) — the spec §4 scaffolding every feature function is built against
      (`def feature(company_id, as_of, ctx) -> float | None`). Lookahead safety is enforced once,
      centrally, in `FeatureContext.get_latest_fact` (delegates to the already-lookahead-safe
      `P22Repo.get_financial_facts_as_of`), not re-implemented per feature function.
- [x] `features/block_c.py` — all 6 spec §4.3 Financial Screen features implemented and unit-tested
      against synthetic fixtures, both the real-computation path and the null path (spec §8.1).
      `enterprise_value`/`ev_to_cash`/`size_band`/`atm_capacity_pct` correctly return `None` today
      (no code change needed) because `market_cap`/`atm_shelf_remaining` aren't normalized into the
      store yet — they'll start returning real values the moment those upstream pieces exist.
      **Update, 2026-08-30, later same day:** `cash_runway_months`/`dilution_risk`'s runway leg are no
      longer blocked — see the `financial_facts.py`/`get_trailing_average` entries below.
- [x] `features/quality.py` — pandera schemas for the two spec §8.2 bounds that apply to Block C
      (`cash_runway_months ∈ [0,120]`, `enterprise_value` unbounded/nullable but `market_cap ≥ 0`),
      a `loe_date` bound function (depends on `as_of`, spec §8.2), and
      `assert_every_company_has_a_verified_alias` (spec §8.2's "no company row without a verified
      alias" — a set-membership check across two tables, not a column bound, so not a pandera schema).
      New `P22Repo.get_companies_without_verified_alias()` backs it, DB-tested.
- [x] `features/lookahead_audit.py` — spec §8.3's mandatory sampling (`stratified_sample`, guarantees
      minimum coverage of the three named high-risk categories rather than uniform sampling) and
      assertion (`assert_lookahead_safe`, `assert_known_from_is_filing_date_not_period_or_crossing_date`)
      logic. Pure and fully unit-tested. **Deliberately not wired to real DB data or CI yet** — the
      three high-risk categories spec §8.3 names (vendor-sourced facts, 13F holdings, 13D/process
      events) have zero rows in this repo today, so a real audit run right now would be vacuous (a
      pass that never exercised the categories that matter), which is worse than no gate — see that
      module's docstring. Wire this in once vendor/13F/13D ingestion exists (M5/M6+), not before.
- [x] Added `pandera>=0.24.0` to `requirements.txt` — **correction, not a new decision**: `docs/
      Requirements.md` already claimed this was "already a repo dependency" before this session; it
      was not actually installed or present in `requirements.txt` anywhere in the repo. Verified via
      `pip install` + a real `DataFrameSchema.validate()` call before trusting the claim this time.
- [ ] Blocks A, B, D, E, F (spec §4.1, §4.2, §4.4, §4.5, §4.6) — not started. All are blocked on real
      data this repo doesn't have normalized yet: Block A now has real patent-expiry data
      (`p22_patent_expiry`, see below) but still needs curated acquirer entry/exit dates (item 3) +
      market cap (item 1) + deal history (M6) + `p22_asset`/`p22_trial` population; Block B needs real
      base rates (item 2) + `p22_trial`/`p22_asset`; Block D is computed from Blocks A-C's own outputs
      so it's blocked transitively; Block E needs 8-K/DEF 14A text-parsing infrastructure that doesn't
      exist; Block F needs 13F integration (M5/M6 scope per spec's own milestone table). Not attempted
      this pass rather than built against fabricated/guessed inputs.
- [x] `p22_trial` normalization from landed CT.gov `clinicaltrials_studies` payloads, 2026-08-30 —
      `ingest/trial_normalization.py` + `jobs/run_trial_normalization.py`, registered in
      `register_jobs.py` (after Alias Matching). Field paths live-verified against a real CT.gov
      response before writing extraction code (same discipline as the rest of this build). Every
      column `CLINICALTRIALS_FIELDS` actually supports is populated (`phase`, `status`, `enrollment`,
      `primary_completion_date` incl. `YYYY-MM`-only dates, `countries`, `primary_endpoint_text`,
      `is_randomized` — with CT.gov's `NA` allocation correctly mapped to `None`, not `False`, since
      "not applicable" isn't "not randomized"). `uses_biomarker_selection`, `has_active_comparator`,
      and `endpoint_changed_midtrial` are always written `None` — the fields to fill them honestly
      (`eligibilityModule`, `armGroupsModule[].type`, and the version-history diff respectively) aren't
      fetched/built yet, not overlooked; see that module's docstring. `P22Repo.upsert_trial` added
      (plain upsert keyed on `nct_id`, not a bitemporal chain — CT.gov re-fetches naturally overwrite a
      trial's current state; the change-over-time signal lives in the separate, still-unused
      `clinicaltrials_history` raw-zone source). **`asset_id` was originally always `None`** — see
      "Decisions needed" item 8, since **resolved (2026-08-31) for single-intervention trials** — see
      that item and the `asset_normalization.py` entry below.
- [x] `ingest/asset_normalization.py` + `ingest/therapeutic_area_classifier.py`, 2026-08-31 —
      resolves item 8's safe subset: a trial with exactly one DRUG/BIOLOGICAL intervention has no
      ambiguity about which intervention is the sponsor's own asset, so `p22_asset` rows are now
      resolved/created (deduped per `(company_id, name)`) and linked via `p22_trial.asset_id` for
      those trials — wired into `jobs/run_trial_normalization.py` (now passes `company_id` through)
      and `ingest/trial_normalization.write_trial_records` (new optional `company_id` param, backward
      compatible — omitting it preserves the old never-link behavior, so existing callers/tests are
      unaffected). Multi-intervention trials are still unlinked, unchanged. `therapeutic_area`
      (`NOT NULL` on `p22_asset`) comes from a new best-effort keyword classifier over CT.gov
      `conditions` text, with an explicit `unclassified` fallback added to `p22_therapeutic_area.yaml`
      rather than a forced guess — disclosed as imperfect in its own docstring, not validated against
      real clinical taxonomy. New `P22Repo.upsert_asset`/`get_asset_by_company_and_name`, DB-tested.
- [x] `ingest/acquirer_config.py` + `jobs/run_acquirer_load.py`, 2026-08-30, later same day — loads
      `p22_acquirers.yaml` into `p22_company` (role `acquirer`, or `both` if the ticker already
      matches a resolved DERA target row — `P22Repo.upsert_acquirer_company` merges by ticker rather
      than duplicating an identity, since the config's CIKs are all `null`; DB-tested for both the
      new-row and merge-into-existing-row cases). Deliberately loads only *identity* — `bloc`/
      `entry_date`/`exit_date` are read as data but never written to any DB column (none exists;
      they stay in the config for Block A to read directly once built). This was NOT gated on
      "Decisions needed" item 3: whether the roster's dates/CIKs are accurate is a curation question,
      but whether the ~21 already-named companies exist as `p22_company` rows is a separate,
      mechanical one — see that module's docstring for the reasoning.
- [x] `ingest/patent_expiry_normalization.py` + `jobs/run_patent_expiry_normalization.py`, 2026-08-30,
      later same day — normalizes landed Orange Book `products.txt`+`patent.txt` into
      `p22_patent_expiry` (Block A input). Orange Book file format live-verified against the real,
      current ZIP (`Patent_Expire_Date_Text` format `"Aug 24, 2026"` confirmed). Only `patent.txt` is
      normalized — `exclusivity.txt`'s `Exclusivity_Code` space (`NCE`, `ODE-###`, `PED`, `GAIN`, ...)
      isn't collapsed onto the 4-value `exclusivity_type` enum, since that mapping is itself a
      domain-classification decision of the same character as therapeutic-area classification, not
      attempted here; every row this module writes is a genuine patent, so `exclusivity_type="patent"`
      is a safe constant, not a guess. Applicant-name -> acquirer-roster resolution reuses
      `alias_matching.match_alias` but is **deterministic-only** — a fuzzy match is logged (with its
      score) and NOT written or queued, because writing it safely would mean extending
      `review_queue.py`'s confirm dispatch to know how to write a `p22_patent_expiry` row from a
      confirmed item, which wasn't built this pass (a real, contained gap, not an oversight — logged
      here). `therapeutic_area` and `ttm_revenue_usd` are always `None` (same classification gap /
      spec's own "highest-effort part of the build" scoping, respectively).
      `P22Repo.upsert_patent_expiry` added — idempotent on `(acquirer_id, application_no, loe_date,
      source)` since the spec's own schema gives this table no natural unique key.
- [x] `ingest/financial_facts.py` extended, 2026-08-30, later same day — `total_debt`
      (`LongTermDebtNoncurrent`/`LongTermDebt`/`ConvertibleDebtNoncurrent`, all live-verified, MERGED
      not first-wins, since live data caught Alnylam mid-migration between two of these tags with no
      overlap — merging correctly picks up both eras) and `short_term_investments`
      (`ShortTermInvestments`, live-verified for 1 of 3 filers) added to `FACT_TAG_MAP`, now a
      `metric -> list of candidate tags` map. **Correction vs. spec §2.1's own suggested debt tag
      list**, which names `ConvertibleNotesPayable`: checked live and absent from all 3 filers;
      `ConvertibleDebtNoncurrent` is what's actually in use. New `extract_quarterly_delta_series` +
      `DURATION_DELTA_TAG_MAP` derive `quarterly_opex_burn` from
      `NetCashProvidedByUsedInOperatingActivities`'s cumulative-YTD XBRL entries (live-verified
      against real Moderna data that the entries genuinely are cumulative, confirming the derivation
      was necessary, not just theoretically possible) — groups by fiscal-year `start`, diffs
      consecutive `end`-sorted cumulative values. Item 5 above updated to reflect this is resolved.
- [x] `features/context.FeatureContext.get_trailing_average()` added, 2026-08-30, later same day —
      averages the most-recent-N known values of a metric (spec §4.3: "trailing-4Q average"), reusing
      `get_latest_fact`'s same lookahead-safe read. `features/block_c.cash_runway_months` rewired to
      use it (with `quarterly_opex_burn`'s raw-signed value flipped to a burn magnitude in Block C,
      not the normalizer — kept explicit per `financial_facts.py`'s own "this module doesn't
      reinterpret the sign" boundary). `cash_runway_months` and `dilution_risk`'s runway leg now
      compute real (non-`None`) values whenever a company has cash and burn history on file;
      `dilution_risk`'s catalyst leg is still `None` (needs `catalyst_days_to_next`, not built).

### 🚀 PLANNED ENHANCEMENTS (by milestone, spec §9)
- [ ] **M4 — Rule-based scoring:** `fit()` pairwise gates (§4.4), Phase 1 composite (§5.1).
- [ ] **M5 — Block G:** 8-K strategic-alternatives phrase detection (reuse
      `EdgarDownloader.efts_text_search`), Schedule 13D ingest (reuse
      `EdgarDownloader.download_13dg_filings`), tiering logic (§5.2), verification gate (§4.7).
- [ ] **M6 — Labels + backtest:** add SC 14D9 / DEFM14A / S-4 support to `EdgarDownloader` (reuse
      `efts_filings_search`, EFTS indexes these directly); hand-verified deal-label dataset with
      `deal_type` classification and reverse-merger exclusion (§2.5); walk-forward harness against
      all three baselines (§0.3); `cvr_policy.yaml` decision (§10, "On CVR valuation" — recommended
      v1 convention: value CVRs at zero).
- [ ] **M7 — Return model:** `E[return | deal]`; `expected_value` becomes default ranking (§5.4).
- [ ] **M8 — Calibrated model:** only if M6 shows lift over the naive-informed baseline.
- [ ] **M9 — Partnership structures:** manual EX-10 enrichment, scoped to top 200 by composite.
- [ ] **M10 — API + alerts:** FastAPI read endpoints; idempotent change alerts.

## Technical Debt
- None yet — M1 is a fresh build.

## Known Issues / Open Decisions
- **Purple Book has no stable "latest" URL** — discovered live 2026-08-30 while building the
  client: FDA publishes one dated CSV per month
  (`.../PurpleBook/{year}/purplebook-search-{Month}-data-download.csv`), each a full ~2,270-row
  snapshot with that month's New/Updated rows flagged in an `N/R/U` column, not a diff, despite the
  file's "Monthly ... Changes Report" title row. `orange_book_client.discover_latest_purple_book_url`
  derives the current URL from the downloads listing page rather than hardcoding one — verified
  end-to-end against the live site (2,273 rows, correct columns including
  `Exclusivity Expiration Date`). The CSV also has 3 preamble rows before the real header;
  `_parse_purple_book_csv` locates the header by content (`N/R/U` prefix), not a fixed row offset.
- **CT.gov version history has no documented public endpoint** — `/api/v2/studies/{nctId}/history`
  404s; the real data is behind the undocumented `/api/int/studies/{nctId}/history` (backs CT.gov's
  own history-viewer UI). Same risk class as P20's pdufa.bio dependency: could change or vanish
  without notice, and there's no documented alternative for spec §2.2's "Critical" requirement.
  Monitor; if it breaks, there is currently no fallback source for this data.
- **CT.gov `fields` param requires fully-qualified paths**, not the bare names spec §2.2 lists —
  fixed in `config.CLINICALTRIALS_FIELDS`; see `docs/implementation-plan.md` §4.1.
- **openFDA `sponsor_name` search is case-sensitive** — fixed by uppercasing the search term in
  `openfda_client.py`; see `docs/implementation-plan.md` §4.1.
- **Delisted-ticker historical price vendor not selected** (spec §2.0.6, narrowed from the earlier
  broader §2.4 framing now that §2.0's source-capability matrix assigns live-name prices to the
  already-integrated IBKR downloader and fundamentals to EDGAR). This blocks `E[return | deal]`
  labeling (M6/M7) specifically, not Block A/C live scoring. Spec's own recommendation: **FMP
  Starter (~$15/mo)** — validate delisted-ticker coverage against ~20 known acquisitions before
  committing. Decide before M6. `ingest/vendor_market_data.py` is ready to receive a real
  implementation behind its `Protocol` once a vendor is picked.
- **IBKR pacing limits** (§2.0.5) — a naive loop over ~700 tickers for daily-price backfill will
  trigger pacing violations and silent truncation. Whatever M3 code pulls IBKR history per ticker
  must batch/throttle deliberately; do not assume the existing `ibkr_downloader` call sites already
  handle a 700-ticker sweep gracefully without checking.
- **IBKR may not be usable for the raw-price archive at all** (spec §2.0.7, added v0.6) — the existing
  `ibkr_downloader` requests `whatToShow="TRADES"`, and IBKR's documented behavior is that TRADES bars
  are split-adjusted server-side with no raw-print option. Not live-verified this pass (no IBKR session
  available). If confirmed, either accept IBKR rows as not-truly-raw and lean on the SEC-filing
  corporate-action reconciliation job to flag discontinuities, or source raw prints from whichever
  vendor gets picked for the delisted-ticker gap below and drop IBKR from the price-archive role
  entirely. See `docs/implementation-plan.md` §2.0.7. **Verify before M6/M7 price-archive ingest is built.**
- **openFDA rate limit / API key** — M1 client runs unauthenticated (240 req/min, 120k/day per
  openFDA's published free tier as of this writing). Revisit if daily universe size makes that
  tight; add `OPENFDA_API_KEY` to `config/donotshare` if so.
- **No raw-zone cleanup/retention job.** `DATA_CACHE_DIR/p22/raw/` grows unbounded with daily
  snapshots. Not a problem at M1 volumes; revisit before M3 once actual disk growth is observed.
- **CT.gov / openFDA rate limits in `ingest/rate_limits.py` are conservative defaults**, not
  confirmed against each API's current published limits at implementation time — verify against
  live docs before removing the "conservative" qualifier from the module docstring.
- ~~**`pyyaml` is not pinned in the root `requirements.txt`**~~ — **fixed 2026-08-30, later same
  day**: pinned as `PyYAML>=6.0.3` (the version already installed in `.venv`) at the user's request.

## Testing Requirements
- [x] Unit tests: raw-zone dedup/hashing, each client (mocked HTTP), DB model shape (no live DB)
- [x] Unit tests: price-archive adjustment math (`test_price_archive.py`, no DB — including the
      lookahead-guard case), entity resolution incl. `build_universe_history` (`test_entity_resolution.py`),
      alias matching incl. field-extraction and `known_from` threading (`test_alias_matching.py`),
      raw-zone latest-partition read incl. manifest pairing (`test_raw_zone.py`), quarter-spanning
      DERA read (`test_universe_snapshot.py`), review-queue confirm/reject/depth-report
      (`test_review_queue.py`), financial-fact XBRL normalization incl. the comparative-column dedup
      trap (`test_financial_facts.py`), feature context/registry (`test_feature_context.py`,
      `test_feature_registry.py`), Block C incl. every null path (`test_block_c.py`), pandera schemas
      (`test_quality.py`), lookahead-audit sampling/assertions (`test_lookahead_audit.py`), CT.gov trial
      normalization incl. the `NA`-allocation and partial-date edge cases (`test_trial_normalization.py`),
      acquirer-config parsing incl. round-tripping the real repo config file
      (`test_acquirer_config.py`), Orange Book patent-expiry normalization incl. the unmatched-product
      and blank-date drop cases (`test_patent_expiry_normalization.py`), the `total_debt` tag-migration
      merge and `quarterly_opex_burn` quarter-delta derivation incl. the separate-fiscal-years and
      comparative-column-dedup cases (`test_financial_facts.py`), `get_trailing_average`
      (`test_feature_context.py`), CT.gov single-intervention asset linkage incl. the deduping and
      multi-intervention-stays-unlinked cases (`test_asset_normalization.py`), keyword therapeutic-
      area classification incl. the heme-vs-solid-oncology ordering and never-guessed-category cases
      (`test_therapeutic_area_classifier.py`) — 229 tests total in the non-DB suite as of 2026-08-31.
- [ ] Real-Postgres integration tests for `P22Repo.upsert_financial_fact_bitemporal` restatement
      behavior, the price-archive round trip (`upsert_price_daily` immutability,
      `get_adjusted_close`'s lookahead guard through the repo layer), `upsert_trial`'s
      keyed-on-`nct_id` update-in-place behavior, `upsert_acquirer_company`'s ticker-merge/idempotency
      behavior, `upsert_patent_expiry`'s idempotent-insert behavior, and the `upsert_asset`/
      `get_asset_by_company_and_name` round trip — present in `tests/db/test_repo_p22_bitemporal.py`
      (isolated to its own subdirectory so its autouse DB fixture doesn't force the rest of the suite
      to connect — see `docs/implementation-plan.md` §3), opt-in via `ETRADING_TEST_DB_URL` like the
      rest of the repo's DB-touching tests. Attempted 2026-08-30: fails locally with
      `database "postgres" does not exist` — the same pre-existing dev-machine Postgres limitation
      noted elsewhere in this repo's memory, not something these tests introduced. Not run in the
      default `pytest` invocation until CI is wired for it (tracked here, not yet done).
- [ ] M3/M5/M6: the mandatory §8.3 lookahead-audit test, wired against real DB data — the sampling
      and assertion logic itself is built and tested (`features/lookahead_audit.py`), but running it
      for real needs the three named high-risk categories (vendor facts, 13F, 13D/process events)
      populated first; see that module's docstring and "Decisions needed" above.

## Documentation Updates
- [ ] Update this file at the start of each new milestone's work, not retroactively.
