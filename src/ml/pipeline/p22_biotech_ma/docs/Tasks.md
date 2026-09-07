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
   P20 Kestrel, P05, and the Telegram screener).

   **Web-search-based plan comparison done 2026-08-31** (FMP's own pricing pages 403 every fetch
   attempt — Cloudflare-blocked — so this is search-index-derived, not a page read): Basic AND
   Starter are BOTH capped at ~5 years of history; only **Premium** (~$69/mo vs. Starter's ~$29/mo)
   unlocks up to 30 years. Spec's "Starter (~$15/mo)" recommendation looks stale against current FMP
   pricing/tiers. User is checking/considering the existing account.

   **Live-verified against the account's real, currently-active key, 2026-08-31 (a second, separate
   check, same day) — and it complicates the "5-year cap" framing above:**
   - `/stable/historical-price-full` (what `FMPDataDownloader.get_ohlcv` calls) is **dead (404)**,
     even for an obviously-valid symbol (MRNA). The correct current endpoint is
     `/stable/historical-price-eod/full` — confirmed working, and confirmed to return full history
     back to a company's own IPO date (MRNA: back to 2018-12-07, its actual IPO), not truncated to a
     5-year rolling window.
   - **But** the same key gets `402 Payment Required` ("this value set for 'symbol' is not available
     under your current subscription") for AMGN, GILD, and SRPT — every date range tried, `/full` and
     `/light` alike — while MRNA and PFE work fine. This looks like a **per-symbol entitlement list**,
     not a date-depth cap at all. Whether upgrading to Premium changes symbol coverage, date depth, or
     both is not something this session could determine from outside — **needs checking directly
     against the FMP account dashboard**, and is now the real open question, more than "how many years
     of history."
   - `/stable/search-name` (company name search, for the delisted-ticker-with-no-symbol-on-file gap)
     is real and working — confirmed live. A single query can return multiple exact-name matches
     across exchanges (e.g. Moderna's real NASDAQ listing AND an unrelated Frankfurt cross-listing,
     both literally named "Moderna, Inc.") — `ingest/fmp_backfill.resolve_ticker_by_name` handles this
     (prefers a USD/US-exchange candidate), a real bug this live check caught before it shipped.
   - The historical-price response has no `adjClose` field at all — whether `close` is raw/unadjusted
     or already split/dividend-adjusted is **still unresolved**, deliberately deferred (see
     `ingest/fmp_client.py`'s docstring) — land raw now, verify against a known split event once real
     data is examined, not before.

   **Built and ready, 2026-08-31, ahead of the account decision** (per user request, "ready by the
   time I buy premium"): `ingest/fmp_client.py` (the two live-verified endpoints above),
   `ingest/fmp_universe.py` (splits the target universe into "already has a ticker" vs. "needs
   name-search resolution first" — FMP is ticker-keyed and most delisted-before-we-resolved-them
   companies only have a CIK on file, not a ticker), `ingest/fmp_backfill.py` (orchestration: resolve
   unresolved names, land full raw price history per ticker, resumable via new
   `raw_zone.has_any_landed`), and `cli/fmp_backfill_cli.py` (human-run: `test-search`, `backfill
   --dry-run`, `backfill --limit N` for a small test batch, `backfill` for the full run). **Explicitly
   NOT built yet**: normalizing landed payloads into `p22_price_daily`/`p22_corporate_action` — a
   deliberate choice, not a gap, since that step isn't time-boxed to a Premium month (raw zone is
   immutable) and depends on resolving the raw-vs-adjusted question above against real data first. Nor
   is a `MarketDataProvider` Protocol implementation (`ingest/vendor_market_data.py`'s
   `NullMarketDataProvider`) — same reasoning.

   **Narrowed 2026-09-07: FMP is now needed ONLY for delisted-ticker historical prices (M6 backtest
   labeling), not for live Block A/C scoring.** The "point-in-time `market_cap` may not need a
   dedicated FMP endpoint" idea above is now built: `ingest/market_cap.py` +
   `jobs/run_market_cap_compute.py` derive `market_cap = raw_close(t) × shares_outstanding(t)` for
   any CURRENTLY-LISTED company, using data already normalized (`price_ingest.py`'s yfinance daily
   close + `financial_facts.py`'s `shares_outstanding`) — zero FMP dependency. Both inputs are kept
   raw/as-filed (never split-adjusted), per `price_archive.py`'s "raw-on-raw" requirement. Registered
   daily in `p22_specs.py`, after Daily Price Ingest + Financial Facts Normalization. This unblocks
   Block A's `equity_capacity`/`dry_powder` (market_cap leg) and finishes Block C's
   `enterprise_value`/`ev_to_cash`/`size_band` for every currently-listed company — see
   `features/block_c.py`'s updated docstring. Delisted companies (acquired, no longer trading) still
   get `None` here, same as before — that specific gap is what the FMP Premium decision is actually
   for now, needed for M6, not M3/Block A live scoring.
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
   itself isn't built yet (see M3 in-progress list) — `equity_capacity`'s market-cap leg is unblocked
   as of 2026-09-07 (item 1), but `cash_capacity` still needs EBITDA/net_debt normalized (not in
   `FACT_TAG_MAP` yet) and `currency_quality` needs a peer-group `fwd_pe` percentile rank + a realized-
   vol stability factor, neither built. `deal_cadence_3y`/`stock_deal_propensity` are separately
   hard-blocked on `p22_deal` (M6), not on market data at all.
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
6. ~~**IBKR split-adjustment behavior**~~ — **superseded 2026-09-01, not resolved.** Still not
   live-verified (still needs a real TWS/Gateway connection this session never had), but IBKR is no
   longer in the critical path for the ongoing/current-price role that motivated checking it: that
   role is now served by yfinance instead (see the new "Daily current-price ingest" entry below),
   which sidesteps this question entirely rather than answering it. IBKR remains a candidate for a
   *different* future role (e.g. options IV / short interest, spec §2.0.5) where this question would
   need revisiting on its own merits, not carried forward from this decision.
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
9. **EBITDA and forward-P/E data source for Block A** (spec §4.1's `cash_capacity`/`currency_quality`)
   — new, found 2026-09-08 while building `features/block_a.py`. FMP's `/stable/ratios`/
   `/analyst-estimates` endpoints would supply both (live-verified real, working responses for PFE/
   ABBV/JNJ), but the account's current tier 402s for the other 22 of 25 `p22_acquirers.yaml`
   tickers, including top-5-by-revenue names like Merck — a narrower and more consequential
   entitlement gap than item 1's delisted-ticker-history finding, since it blocks live scoring for
   the acquirer universe Block A exists to model, not just backtest labeling. `ingest/fmp_client.py`'s
   `fetch_ratios`/`fetch_analyst_estimates` are built and live-verified, ready to normalize the
   moment coverage is confirmed sufficient (upgrading the account, or an alternative vendor with
   broader forward-estimates coverage for this specific universe — not assumed to be the same
   answer as item 1's plan choice, since forward-consensus-estimates coverage and historical-price
   depth are different products even within one vendor).
10. **`stability_factor(trailing 12m realized vol)` has no defined formula** (spec §4.1's
    `currency_quality = percentile_rank(fwd_pe) × stability_factor(vol)`) — new, found 2026-09-08.
    Spec names the input and the role but never gives the function shape (unlike, say,
    `max_dilution_tolerance = 0.15`, which spec states directly). Inventing one (e.g. `1/(1+vol)`)
    would be exactly the kind of fabricated business logic this codebase's discipline forbids
    elsewhere (cf. §4.2's orphan-modifier warning: "any implementation that adds a constant... is
    wrong and must fail review"). `features/block_a.py`'s `percentile_rank` implements the OTHER
    (unambiguous) half of `currency_quality` for real; `equity_capacity` reads `currency_quality`
    itself as an already-computed fact, so it's ready the moment both halves exist — this decision
    plus item 9 above are what's actually blocking it, not missing code.
11. **`target_leverage_ratio` (Block A `cash_capacity`) and `assumed_peak_sales` per therapeutic
    area (Block A `pipeline_gap_by_ta`) both need domain curation** — new, found 2026-09-08. Spec
    names both inputs but gives no values for either (unlike `p22_base_rates.yaml`'s anchor figures,
    which spec states directly) — same character as `p22_modality.yaml`'s still-open taxonomy
    review (item 4). `features/block_a.py`'s `TARGET_LEVERAGE_RATIO` module constant is `None` until
    curated (`cash_capacity` returns `None` while it is); `pipeline_gap_by_ta` takes
    `assumed_peak_sales_by_ta` as an explicit caller-supplied parameter rather than a guessed
    constant.
12. **`config/activist_filers.yaml` needs a domain reviewer to grow it toward spec's "~30"
    target and disambiguate multi-CIK names** — new, found 2026-09-08. Seeded with 11
    individually live-verified single-CIK healthcare-specialist funds (see that file's header);
    several other well-known generalist activists (Starboard Value, Icahn Enterprises, Third
    Point, JANA Partners) each resolved to multiple distinct real CIKs (affiliated
    LP/GP/co-investment entities) with no mechanical way to tell which one actually files the
    firm's 13Ds — left out rather than guessed, same discipline as item 3's acquirer-roster CIKs.
13. **`lead_asset_poa`'s later-phase and orphan-status combination formula is underspecified** —
    new, found 2026-09-08 building `features/block_b.py`. `p22_base_rates.yaml`'s
    `by_therapeutic_area` figures are LOA-FROM-PHASE-1 (already conditional on being no further
    than Phase 1); spec gives separate, TA-AGNOSTIC `phase_2_success`/`phase_3_to_filing`/
    `filing_to_approval` rates for later stages but never says how to recompose a TA-specific
    later-phase estimate from a TA-specific Phase-1 figure plus TA-agnostic later-stage rates.
    `orphan_by_phase` compounds this: it's a full alternate phase-conditional rate table (its own
    Phase I/II/III POS for orphan vs. non-orphan), not a simple multiplier, so how it's meant to
    combine with `by_therapeutic_area` at all is unclear — asked as a genuine question, not
    resolved by guessing. `features/block_b.lead_asset_poa` computes a real value only for a lead
    asset still at Phase I or earlier (where `by_therapeutic_area` applies directly with no
    composition needed) and returns `None` for anything further along, pending this decision.

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
- [x] `ingest/market_cap.py` + `jobs/run_market_cap_compute.py`, 2026-09-07 — derives
      `market_cap = raw_close(t) × shares_outstanding(t)` and writes it as an ordinary
      `p22_financial_fact` row, resolving item 1's "may not need a dedicated FMP endpoint" idea into
      real code. New `P22Repo.get_latest_raw_close_as_of` (DB-tested) reads the RAW, unadjusted close
      — deliberately not `get_adjusted_close` — pairing raw price with as-filed
      `shares_outstanding`, per `price_archive.py`'s "raw-on-raw" requirement. Registered daily in
      `p22_specs.py` after Daily Price Ingest + Financial Facts Normalization. Uses the same
      rejection-breakdown-`Counter` diagnostic pattern as P20's `sleeve_a.py`/`sleeve_c.py` (a
      `_logger.warning` with the top skip reasons whenever nothing computes) so a systematic gap is
      visible in the run summary, not a silent all-`None` funnel. Only reaches currently-listed
      companies (no yfinance ticker for a delisted one) — see item 1's narrowed framing above.
- [x] `features/block_a.py`, 2026-09-08 — `revenue_at_risk_3y`/`_5y`, `cash_capacity`,
      `equity_capacity`, `dry_powder` implemented and registered (spec §4.1), plus a standalone
      `percentile_rank` utility and `pipeline_gap_by_ta` (deliberately NOT a `@register_feature` —
      spec's own table marks it "Per TA," i.e. dict-shaped, not the single `float | None` every
      other feature returns; see that function's docstring, same spec/infra mismatch class as
      Block D's pairwise `fit()`). New `P22Repo.count_phase3_assets_by_therapeutic_area` (DB-tested)
      + `FeatureContext.get_phase3_asset_count_by_ta` back `pipeline_gap_by_ta`'s count leg — real
      today for single-intervention-trial-linked assets. **Every function is correct and unit-tested
      against synthetic fixtures (both the real-computation and null paths, spec §8.1), but almost
      all return `None` in production today** — same "scaffolding ahead of the blocker" pattern as
      Block C's `market_cap` history, except Block A has more independent blockers than Block C ever
      did: `existing_net_debt` and `market_cap` are real (already normalized), but `ebitda` and
      `currency_quality`'s two halves (items 9-10) and `target_leverage_ratio`/`assumed_peak_sales`
      (item 11) are all still missing, each for a different reason (vendor entitlement, undefined
      spec formula, and un-curated business assumptions respectively) — see those items. Also built
      `ingest/fmp_client.py`'s `fetch_ratios`/`fetch_analyst_estimates` (live-verified against all 25
      acquirers, only 3 covered on the current plan — item 9) ahead of resolving that gap, same
      "ready when the decision lands" precedent as the FMP historical-price work in item 1.
      `deal_cadence_3y`/`stock_deal_propensity` are NOT stubbed even as always-`None` functions —
      both need `p22_deal`, a table that doesn't exist until M6; a stub reading a nonexistent table
      would be pure theater (module docstring).
- [x] `features/block_b.py`, 2026-09-08 — `phase_max`, `asset_count_ph2plus`, `catalyst_window`,
      `lead_asset_poa` implemented and registered (spec §4.2), all reading
      `FeatureContext.get_trials_for_company` (new — one query returning the company's whole
      trial/asset portfolio, since these features need portfolio-level aggregation, not a single
      fact lookup like Block A/C). New `ingest/base_rates_config.py` is `p22_base_rates.yaml`'s
      first real reader — that file has been fully curated since 2026-08-31 (item 2) but had no
      consumer until now. **"Lead asset" is a disclosed proxy**: `p22_asset.is_lead` is always
      `None` (`asset_normalization.py`'s known gap), so the furthest-progressed asset (highest
      phase reached) stands in for it everywhere spec says "lead asset."
      **Real today**: `phase_max`/`asset_count_ph2plus` (only need `p22_trial`/`p22_asset`,
      already flowing); `catalyst_window`'s ordinary forward-looking buckets (from
      `primary_completion_date`, already normalized) — its `post_positive_0-180` bucket can never
      fire (needs positive-readout detection this repo doesn't have, undisclosed anywhere as a
      gap until now). `lead_asset_poa` is real ONLY for a lead asset still at Phase I or earlier —
      new "Decisions needed" item 13 explains why later phases return `None` rather than a guessed
      base-rate composition formula; falls back to `loa_from_phase_1_overall` (logged, per spec's
      `base_rate_fallback` requirement) for the 9 `by_therapeutic_area` entries still `null`.
      **Not implemented at all this pass, no stub functions**: `has_positive_ph3` (needs CT.gov
      `hasResults` + 8-K discontinuation-announcement detection, neither built, and spec gives no
      phrase list for the latter the way it did for `process_events.py`'s strategic-alternatives
      detection), `pdufa_pending` (needs a PDUFA-date data source — none identified; P20 Kestrel's
      `pdufa.bio` dependency is flagged elsewhere in this repo's memory as fragile/undocumented,
      not assumed reusable here without its own check), `endpoint_stability` (needs
      `endpoint_changed_midtrial`, always `None`, `trial_normalization.py`'s disclosed gap),
      `trial_design_quality` (needs `has_active_comparator`/`uses_biomarker_selection`, both
      always `None` — a partial 2-of-4 composite was considered and rejected: it would look like a
      real quality score while actually encoding "we don't know" as "not present," a worse error
      than `None`), `ev_to_risk_adjusted_npv` (needs a per-asset peak-sales/multiple assumption —
      same character as Block A's `assumed_peak_sales_by_ta`, not yet a separate curation item
      since no consumer exists to make concrete what "per asset" would even need).
- [ ] Blocks D, E, F (spec §4.4, §4.5, §4.6) — not started. Block D is computed from Blocks A-C's
      own outputs so it's blocked transitively. Block E needs 8-K/DEF 14A text-parsing
      infrastructure that doesn't exist. Block F needs 13F integration (M5/M6 scope per spec's own
      milestone table). Not attempted this pass rather than built against fabricated/guessed inputs.
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
- [x] FMP historical bulk-backfill infrastructure, 2026-08-31 — `ingest/fmp_client.py`,
      `ingest/fmp_universe.py`, `ingest/fmp_backfill.py`, `cli/fmp_backfill_cli.py`, new
      `raw_zone.has_any_landed()`, new `P22Repo.list_companies_full()`. Built ahead of the vendor
      account decision (item 1) at user request, so the download itself doesn't waste a paid-tier
      month on development time. Live-verified against the account's real, currently-active key
      (caught and fixed 2 real bugs before they shipped: a dead endpoint URL, and a
      multiple-exact-name-match ticker-resolution bug) — see item 1 for the full findings, including
      the still-open per-symbol-entitlement discovery. Deliberately does NOT write
      `p22_price_daily`/`p22_corporate_action` or implement `MarketDataProvider` — see item 1's "Built
      and ready" note for why both are correctly deferred past the download step.
- [x] `ingest/acquirer_config.py` + `jobs/run_acquirer_load.py`, 2026-08-30, later same day — loads
      `p22_acquirers.yaml` into `p22_company` (role `acquirer`, or `both` if the company already
      matches a resolved DERA target row). Deliberately loads only *identity* — `bloc`/`entry_date`/
      `exit_date` are read as data but never written to any DB column (none exists; they stay in the
      config for Block A to read directly once built). This was NOT gated on "Decisions needed" item
      3: whether the roster's dates/CIKs are accurate is a curation question, but whether the ~21
      already-named companies exist as `p22_company` rows is a separate, mechanical one — see that
      module's docstring for the reasoning.
      **Real bug found and fixed 2026-09-01**, the first time this ran against real data with real
      acquirer CIKs (the earlier decisions-walkthrough pass): `upsert_acquirer_company` originally
      looked up by `ticker` only, regardless of whether `cik` was given. That crashed with a real
      `UniqueViolation` on Bristol-Myers Squibb — its already-resolved `p22_company` row (from DERA)
      has CIK `0000014272`, but SEC's own current ticker snapshot maps that CIK to `CELG-RI` (a
      leftover Celgene contingent-value-right security ticker from the BMY/Celgene merger), not
      `BMY`. The ticker lookup found no match and tried to INSERT a duplicate with the same
      (unique-constrained) CIK. Now checks `cik` first, and updates `ticker`/`name` on a CIK-matched
      merge (the curated config is more authoritative than a stale snapshot ticker) — DB-tested for
      this exact scenario. All 25 acquirers now load cleanly against the real, populated DB.
- [x] **Daily current-price ingest — a NEW, third price source, separate from FMP**, 2026-09-01 —
      `ingest/yfinance_client.py`, `ingest/price_ingest.py`, `jobs/run_price_ingest.py`, registered in
      `register_jobs.py` (daily, weekdays, after US market close). At the user's explicit request
      ("build daily job first, buy Premium after") to decouple the ongoing/current-price role from
      the one-time FMP historical backfill entirely — confirms neither was ever meant to depend on
      the other. Uses **yfinance** (free, no API key, already a repo dependency), not IBKR — spec
      §2.0.5 originally assigned IBKR to this role, but IBKR needs a live TWS/Gateway session (not
      available in any session so far) and has its own unverified raw-vs-adjusted question (item 6).
      **Live-verified correctness trap, caught before it shipped**: yfinance's `Close`, even with
      `auto_adjust=False` ("not adjusted"), is retroactively split-adjusted across a stock's ENTIRE
      history — confirmed against NVDA's real 2024-06-10 10-for-1 split, where a wide-range fetch
      shows `Close` running smoothly through the split date with no discontinuity, despite the real
      pre-split price being ~10x higher. This is the exact risk class flagged for IBKR (item 6),
      now *confirmed*, not just suspected, for yfinance too. The design deliberately sidesteps it:
      this client is ONLY ever called with a narrow trailing window (`YFINANCE_LOOKBACK_DAYS`, config
      default 7 days), never a historical backfill — a bar landed shortly after its own trading day
      is genuinely raw, since no future split has happened yet to retroactively adjust it.
      `P22Repo.upsert_price_daily`'s existing never-rewrite behavior is a second line of defense.
      **Run successfully against the real, now-populated production DB, 2026-09-01**: 860 companies
      attempted, 4,003 daily price rows written, 6 corporate actions detected (splits/dividends),
      16 failed (genuinely delisted tickers, e.g. `$APTN`/`$CLYD` — "possibly delisted"). Notably
      covers acquirers with no SEC CIK at all (`BAYRY`, `RHHBY`, `IPN.PA`) since yfinance only needs
      a ticker, not a CIK — reaches further than FMP's CIK-oriented historical gap for this purpose.
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

### 🔄 IN PROGRESS — M5 Block G (spec §2.6, §4.7, §5.2)
- [x] 8-K strategic-alternatives phrase detection (spec §2.6.1), 2026-09-08 —
      `ingest/process_events.py` + `jobs/run_process_events_ingest.py`, registered daily in
      `p22_specs.py`. Uses `EdgarDownloader.download_8k_filings` (the universe-wide daily 8-K
      index, previously built but never called by anything in this repo — P17's CatalystAgent
      docstring names it as its intended reader) filtered to Item 7.01/8.01 (never 1.01, which
      only fires once a deal is signed) and to in-universe CIKs, then phrase-matches the primary
      document body via a NEW public `EdgarDownloader.fetch_filing_document` (thin wrapper over
      the existing private `_fetch_filing_document`, so P22 doesn't reach into a private method of
      a shared downloader). **Disclosed scope gap, not an oversight**: only the primary document is
      scanned, not the EX-99.1 press-release exhibit spec also names — `download_8k_filings`'s
      index gives one `primary_document` filename per filing, not the full exhibit list; resolving
      that needs a second per-filing fetch not added this pass (see that module's docstring).
      `p22_strategic_process_phrases.yaml` holds spec's own exact phrase list verbatim (not a
      curated file needing domain review, unlike `p22_base_rates.yaml`) — the one
      `"{ADVISOR}"`-templated phrase is handled as a two-substring match (`"engaged"` ...
      `"as financial advisor"`), since no maintained advisor-name list exists to resolve the
      template against.
      Every match is a CANDIDATE, never auto-scored (spec §4.7's verification gate):
      `P22Repo.upsert_corporate_process_event` writes it `is_verified = FALSE` (idempotent on
      `(company_id, accession_no)` — the job has no high-water mark and will re-scan overlapping
      windows) and a matching `p22_review_item` (`strategic_alternatives_candidate`, new reason
      wired into `ingest/review_queue.py`'s confirm dispatch — confirm flips `is_verified = TRUE`
      via `set_process_event_verified`, no new row created, unlike the two pre-existing reasons —
      see that module's updated docstring for why). Negative phrases ("concluded its review...")
      are checked FIRST, before strong/moderate, so a conclusion announcement that happens to
      contain the substring "strategic alternatives" isn't misclassified as an open process.
- [x] `features/block_g.py`, 2026-09-08 — `BlockG` dataclass + `build_block_g` (assembles it from
      three verification-gated, lookahead-safe repo reads) + `apply_process_tier` (direct,
      tested port of spec §5.2's own tiering pseudocode). Deliberately NOT a `@register_feature` —
      spec itself frames Block G as categorically different from Blocks A-F (tiered, never folded
      into the weighted composite), so it doesn't fit the single-float feature-function contract.
      **Real today**: `process_state`/`process_scope`/`days_since_process_open`, the moment a
      strategic-alternatives candidate above clears review. **Correctly defaults to "none seen"
      today, not yet real**: everything activist/partnership-derived
      (`has_13d_activist`/`activist_intent_max`/`activist_escalation`/`has_strategic_toehold`/
      `strategic_toehold_pct`/`partner_structure_max`/`partner_equity_pct`/`partner_identity`) —
      `p22_activist_position`/`p22_partnership_structure` ingest (spec §2.6.2/§2.6.3) isn't built
      yet, tracked as the next M5 slice below, not a bug here.
      **Real bug found and fixed while building this**: the three new `P22Repo` read methods
      (`get_verified_process_events`/`get_verified_activist_positions`/
      `get_verified_partnership_structures`) initially checked `is_verified` (or, for
      `activist_position`, nothing — no such column) with NO `known_from <= as_of` lookahead
      filter at all — exactly the gap spec §4.7 calls out by name ("this is the most likely place
      in the system for a subtle lookahead leak"). Fixed before it ever shipped by adding the same
      `tzinfo=timezone.utc`-explicit bound `get_financial_facts_as_of` already uses, with a DB
      regression test (`test_get_verified_process_events_enforces_verification_and_lookahead_gates`)
      proving a verified-but-not-yet-known row is correctly invisible.
- [x] **Real bug found and fixed, 2026-09-08, in `EdgarDownloader.download_13dg_filings`** (a
      shared method, not P22-only — see below): live-verified against a real 2026 QTR3 EDGAR
      quarterly form.idx that the method's `_13DG_FORM_TYPES` allowlist (`"SC 13D"`/`"SC 13G"`
      family) matched almost nothing real. The actual index uses `"SCHEDULE 13D"`/
      `"SCHEDULE 13D/A"`/`"SCHEDULE 13G"`/`"SCHEDULE 13G/A"` — 14,486 real filings that quarter,
      vs. only 4 stray legacy `"SC 13D/A"` rows using the old naming. This method had been
      **silently returning an empty DataFrame for virtually every real 13D/G filing since it
      shipped**, undetected because its one regression test happened to use the rare
      "SC 13D/A"-format fixture, which coincidentally still worked. **Three OTHER production
      pipelines call this method and were affected**: `p15_hidden_deps/p15_daily.py`,
      `p18_institutional_flow_tracker/processors/form4_monitor.py`,
      `p19_penny_intraday/structural/profiler.py` — none of them P22 code, found purely because
      P22's own Block G work needed this method to actually work. Fixed via a new
      `_13DG_FORM_TYPE_ALIASES` map (real-index string -> the short canonical form every existing
      caller already expects, so no downstream caller needed to change), plus a new regression
      test (`test_download_13dg_parses_the_real_schedule_prefix_form_type`) using the real prefix.
      A second, related discovery from the same live-verification pass: `download_13dg_filings`'s
      own docstring claim "EFTS does not index SC 13D/G filings" is ALSO only true for the wrong
      form string — EFTS (`efts_filings_search`) DOES index them under the real `"SCHEDULE 13D"`
      family, which is what made the more efficient ingest design below possible; that docstring
      itself isn't corrected in this pass (out of scope — the method's own behavior, not its
      prose, was the actionable bug) but is worth fixing next time that file is touched.
- [x] **Schedule 13D/13D-A/13G/13G-A ingest** (spec §2.6.2), 2026-09-08 —
      `ingest/activist_positions.py` + `jobs/run_activist_positions_ingest.py`, registered daily.
      Uses `EdgarDownloader.efts_filings_search` (CIK-targeted, chunked at 100) with the real
      `"SCHEDULE 13D"` family form strings against the WHOLE P22 universe's CIK list — far more
      efficient than scanning `download_13dg_filings`'s industry-wide daily index and having to
      fetch every single filing's document just to learn which company it's about (that index is
      filer-centric: its own `cik` column is the FILER's CIK, not the subject company's — verified
      live against a real filing before ruling that approach out). EFTS's `_id` field already names
      the exact primary document (`"{accession}:primary_doc.xml"`), so no filename-guessing is
      needed. For each candidate hit, the primary document is fetched (new public
      `EdgarDownloader.fetch_filing_document`, added for `process_events.py` above, reused here)
      and its SGML header parsed for the SUBJECT COMPANY's CIK (matched against the P22 universe)
      and every FILER's CIK/name (one `p22_activist_position` row per filer, idempotent on
      `(company_id, filer_cik, form_type, filed_date)` via new `P22Repo.upsert_activist_position`).
      `filer_type` is populated MECHANICALLY (not a guess): `'activist'` via the new
      `config/activist_filers.yaml` membership check, `'strategic_corporate'` via a
      `p22_company.role in ('acquirer','both')` lookup — both real set-membership/DB checks, unlike
      `stated_intent`. `pct_of_class` is populated from a live-verified `<percentOfClass>` XML tag
      **only when every occurrence in the document agrees on one value** — real filings can carry
      several DIFFERENT percentages across cover pages for different reporting persons even under
      one nominal "FILED BY" company, and misattributing one to the wrong filer would be worse than
      `None`. **`stated_intent` is always `None` — deliberately not attempted.** Spec says to
      classify Item 4 (Purpose of Transaction) text via the review queue, but (a) Item 4 is dense
      boilerplate legal prose (live-verified against a real filing demanding a CEO's removal — not
      cleanly reducible to `passive|engagement|board_seats|sale_demand` by keyword) and (b) unlike
      `process_events.py`'s phrase list, spec gives no candidate phrases to match against at all;
      properly doing this also needs `review_queue.py`'s confirm dispatch to accept a multi-valued
      classification, not just confirm/reject, which isn't built. `p22_activist_position` has no
      `is_verified` column at all (spec's own schema) — a real SEC filing already IS the
      verification, so every row here is written directly, no review-queue step, unlike
      `process_events.py`'s keyword candidates.
      **`config/activist_filers.yaml` seeded, not complete** — 11 of spec's "~30
      healthcare-specialist CIKs" target, each individually live-verified against SEC EDGAR's own
      company-search endpoint (a single, unambiguous CIK with real SC 13D history), not fabricated.
      Several other well-known names (Starboard Value, Icahn Enterprises, Third Point, JANA
      Partners) resolved to MULTIPLE distinct CIKs per name with no mechanical way to pick the
      right one from outside — left out rather than guessed. **New "Decisions needed" item 12**
      below: needs a domain reviewer to grow the list and disambiguate those names.
- [ ] Incumbent-partner / option-to-acquire structures (spec §2.6.3) — NOT built, and structurally
      can't be yet: spec itself scopes this to "the top 200 companies by composite score from the
      Block A-E model," which requires M4's scoring layer to exist first. `p22_partnership_structure`
      already has its full schema + `P22Repo.upsert`/`get_verified_partnership_structures` support
      (added this pass, for `features/block_g.py` to read once rows exist) — only the ingest and the
      M4 ranking it depends on are missing.

### 🚀 PLANNED ENHANCEMENTS (by milestone, spec §9)
- [ ] **M4 — Rule-based scoring:** `fit()` pairwise gates (§4.4), Phase 1 composite (§5.1).
- [ ] **M5 — Block G remaining work:** `stated_intent` classification (needs `review_queue.py`
      support for multi-valued confirm, see above), incumbent-partner structures (blocked on M4's
      composite ranking).
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
- **Fetch-failure logging (spec §7.2) is only wired into SEC EDGAR ingest and daily price ingest**,
  fixed 2026-09-05 alongside the bugs below. `P22Repo.log_fetch_failure` existed since M1 but was
  never called anywhere — every other client's failures (CT.gov, openFDA, Orange Book, Purple Book,
  SEC DERA universe) only reach the log file, not `p22_fetch_failure`, and those jobs don't open a DB
  session at all today. Extending them the same way needs each client's return contract to
  distinguish "legitimately empty" from "request failed after retries" (they currently both return
  `[]`/`None` indistinguishably) — a real but separate refactor, not done speculatively here.

## Known Issues / Open Decisions
- ~~**Naive `as_of` upper bound in `get_financial_facts_as_of`**~~ — **fixed 2026-09-05**:
  `datetime.combine(as_of_date, datetime.max.time())` had no `tzinfo`, while `known_from`
  (`TIMESTAMPTZ`) is always written UTC-aware. A naive bound would be interpreted in the DB session's
  own timezone rather than UTC — silently shifting the single most safety-critical lookahead guard in
  the system (spec §3.1, §8.3) by hours if that session timezone is ever not UTC. Fixed by adding
  `tzinfo=timezone.utc` explicitly (matching the existing correct pattern in
  `src/data/pipeline/dependency_status.py`); regression test sets the DB session's own timezone away
  from UTC to prove the bound doesn't ride along with it
  (`tests/db/test_repo_p22_bitemporal.py::test_lookahead_filter_uses_utc_not_db_session_timezone`).
- ~~**`p22_fetch_failure` was dead code**~~ — **partially fixed 2026-09-05**: `P22Repo.log_fetch_failure`
  is now called from `ingest/sec_raw_ingest.py` (per-CIK submissions/company-facts failures) and
  `jobs/run_price_ingest.py` (per-ticker: no bars returned, or a write failure). `run_sec_ingest.py`
  now opens a `DatabaseService().uow()` for this (it previously ran DB-free). See Technical Debt above
  for what's still not wired.
- ~~**`run_price_ingest.py` held one uow open across the whole daily loop with no write-failure
  isolation**~~ — **fixed 2026-09-05**: a DB write failure for one ticker (distinct from a fetch
  failure, which `fetch_recent_daily_bars` already handles internally) used to leave the session's
  transaction aborted for the rest of the loop, and would have rolled back every already-written
  ticker from earlier in the run when the outer `uow()` rolled back on exit — the exact "one bad
  write nukes the whole run" failure mode Design.md's error-handling contract exists to prevent. Now
  each ticker's DB write runs inside its own `uow.s.begin_nested()` SAVEPOINT, so one failure rolls
  back only that ticker and is logged via `log_fetch_failure`, not the whole day's run.
- ~~**Review queue had no dedup — grew unbounded with duplicate `entity_match` items**~~ — **fixed
  2026-09-05**: `run_alias_matching.py` runs daily and re-extracts the FULL current CT.gov/openFDA
  snapshot every time, and `alias_matching.resolve_aliases` had no check against already-pending
  review items or even against repeats within a single run (the same sponsor string routinely appears
  across dozens of trials for one company). Fixed with two changes: (1) candidates are now deduped by
  name before matching (keeping the earliest `known_from`); (2) `resolve_aliases` takes an
  `already_queued` set of `(candidate_name, matched_company_id, source)` triples — built by
  `run_alias_matching.py` from `get_pending_review_items(item_type="entity_match")` before matching —
  and skips re-queuing a fuzzy match already sitting there pending (counted under the new
  `fuzzy_already_queued`, not silently dropped). Deliberately scoped to *pending* items only —
  suppressing a previously *rejected* candidate forever is a judgment call, not a mechanical dedup,
  and was left alone.
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
- ~~**P22 ClinicalTrials Ingest timed out in production (2026-09-02)**~~ — **fixed same day**: the
  job's first full-universe run (1705 companies) hit its 7200s timeout having covered only
  215 (`ls .../clinicaltrials_studies/2026-09-02 | grep -c manifest`). Root-caused from the real
  prod log (`/opt/apps/e-trading/results/p22_biotech_ma/2026-09-02/pipeline.log`, mounted read-only
  at `R:\` — no SSH needed): the undocumented `/api/int/studies/{id}/history` endpoint throttles far
  harder than the public `/api/v2/studies` endpoint, and sharing one 5rps limiter between both let
  history 429s (1600/8132 history requests, ~20%) burn ~5390s of the 7200s run in exponential-backoff
  sleeps — not raw request volume (only 8686 total requests were issued the entire run). Fixed with
  two changes: (1) `clinicaltrials_history_limiter` (2 rps, unverified starting guess — same
  "no published limit" situation as below) now used only for the history endpoint, keeping the
  public endpoint's 5 rps limiter unaffected by its 429s; (2) `run_clinicaltrials_ingest.py` now
  skips the history re-fetch for any study whose `lastUpdatePostDate` (already present in the
  sponsor-search response) hasn't changed since the previous landed partition
  (`raw_zone.read_partition_before`, new) — both cuts total request volume from day 2 onward and is
  more correct for a "what changed since yesterday" job. `timeout_seconds` widened to 21600 to cover
  the one-time cold-start pass (no prior partition exists yet, so day 1 still fetches every study's
  history); `max_instances=1` on the schedule means this can't overlap the next day's fire. Revisit
  shrinking the timeout once a non-cold-start run's real duration is observed.
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
      (`test_therapeutic_area_classifier.py`), FMP historical-price/name-search client incl. the
      402/404/unexpected-shape cases (`test_fmp_client.py`), the known-vs-unresolved-ticker universe
      split incl. dedup-by-CIK-keeping-latest-name (`test_fmp_universe.py`), backfill orchestration
      incl. the live-caught multi-exact-match tie-break and skip-already-landed resumability
      (`test_fmp_backfill.py`), yfinance daily-bar parsing incl. the narrow-window-request assertion
      (`test_yfinance_client.py`), daily price/corporate-action normalization incl. forward/reverse
      split ratio handling (`test_price_ingest.py`), derived `market_cap` incl. the rejection-
      breakdown aggregation (`test_market_cap.py`), Block A incl. every null path and the
      floored-at-zero/window-boundary cases (`test_block_a.py`), FMP ratios/analyst-estimates client
      incl. the 402/unexpected-shape cases (`test_fmp_client.py`), 8-K strategic-alternatives
      phrase classification incl. the negative-checked-first and advisor-placeholder cases
      (`test_process_events.py`), Block G tiering incl. tier-precedence (`test_block_g.py`),
      13D/G header parsing incl. the multi-filer-block and single-agreeing-percentage cases
      (`test_activist_positions.py`), Block B incl. the lead-asset-proxy and base-rate-fallback
      cases (`test_block_b.py`), `p22_base_rates.yaml` loading (`test_base_rates_config.py`) —
      390 tests total in the non-DB suite as of 2026-09-08 (plus 4 more in
      `src/data/downloader/tests/` for the new `EdgarDownloader.fetch_filing_document` public
      wrapper and the `SCHEDULE 13D` form-type-prefix bug fix, outside this module's own count).
- [ ] Real-Postgres integration tests for `P22Repo.upsert_financial_fact_bitemporal` restatement
      behavior, the price-archive round trip (`upsert_price_daily` immutability,
      `get_adjusted_close`'s lookahead guard through the repo layer), `get_latest_raw_close_as_of`
      incl. the null-`known_from` exclusion case, `count_phase3_assets_by_therapeutic_area` incl. the
      one-asset-two-trials and combined-phase cases, the Block G verification/lookahead gates on
      `get_verified_process_events`/`get_verified_partnership_structures`, `upsert_trial`'s
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
