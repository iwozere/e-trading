# Tasks — P19 v2 (Structural Integrity rework)

Companion to [`design-v2.md`](design-v2.md) / [`requirements-v2.md`](requirements-v2.md).
Supersedes the "🚀 PLANNED" section of `Tasks.md` for Phase 1.5 onward; Phase 0/1
entries there stay as the historical record of what's already shipped.

## Implementation Status

### ✅ Task 0 — Verify Pi status (resolved 2026-08-18, checked directly)

`R:\results\p19_penny_intraday\shadow.sqlite` exists and has real data:
**1,741 rows, 3 trading days (2026-08-13, 2026-08-14, 2026-08-18), 60 distinct
tickers**, all `trigger_reason` empty (expected — Trigger Engine genuinely
unbuilt). `watchlist.json` exists for every weekday back to 2026-06-28, so the
watchlist-build job has been running all along; the *poll* step only started
landing rows in the last week, plausibly unblocked by the 2026-08-16
numba/Python-3.14 scheduler crash-loop fix (commit `30e13ee`). `GRSD` — the
spec's own worked FPI example — is already in the current watchlist.

`R:\data-cache\edgar\` is also warm: 716 `companyfacts`, 436 `submissions`,
an 80-day 8-K index, and a **daily universe-wide Form 4 cache running since
2026-06-12** — see design-v2.md §0.1 and §3.1, this changed the Form 4 design
materially (no new per-ticker EDGAR calls needed; §3.1 widens the existing
cache instead).

- [ ] The additive `ALTER TABLE` migration (design-v2.md §6) must be tested
      against an actual copy of this file (or an equivalent fixture) before
      it's trusted against the real one — 1,741 rows is small enough to do
      this cheaply.
- [ ] Confirm the shadow-poll cron is *staying* up (not a one-week fluke) —
      check `ScheduleRun` rows for the P19 Shadow Poll job over the next
      several trading days before assuming continuous collection.

### ✅ Phase 1.5 — Schema v2 + Layer 0 core subset + outcome labels (built 2026-08-18)

All code + tests below are implemented and passing (0 pyright / 0 mypy on
every touched file; 137/137 P19+EDGAR tests pass — the one unrelated failure
in the full suite is `test_wikipedia_downloader.py`, pre-existing, untouched
by this work). **Not yet done**: applying `insert_p19_v2_schedules.sql` on
the Pi and confirming a real run — that's an operational step, see the new
carry-over list at the end of this section.

**Storage**
- [x] `models/structural_profile.py` — `StructuralProfile` dataclass (spec §4.0)
      + `disqualifier_severities` addition (design-v2.md §2) + `to_dict()`/`from_dict()`.
- [x] `shadow_store.py`: v2 columns added (`_V1_COLUMNS`/`_V2_COLUMNS` split);
      idempotent `ALTER TABLE ADD COLUMN` migration guarded by
      `PRAGMA table_info`; `severity` stays for old-row reads, no longer
      written; `momentum_score`/`momentum_tier` added. Also added
      `get_eod`/`polls_for_date_ticker`/`update_same_day_labels`/
      `update_forward_labels`/`tickers_for_date_needing_labels`/
      `tickers_needing_label_backfill`/`dates_needing_label_backfill`.
- [x] Migration test (`test_shadow_store.py`): a manually-created v1-shaped
      table (mirroring the real schema, not just a fixture) reads back `NULL`
      in every v2 column after opening with the current `ShadowStore`; a
      second reopen doesn't duplicate columns.

**`EdgarDownloader` / shared cache changes** (`src/data/downloader/edgar_downloader.py`)
- [x] `_parse_form4_xml` generalised to all transaction codes +
      `acquired_disposed_code`, `is_10b5_1_plan` (footnote-text heuristic —
      see the docstring for why, and design-v2.md §3.1), `is_derivative`
      (always `False`; derivative transactions aren't parsed, no consumer
      needs them). `download_form4_filings`'s signature and cache path are
      unchanged; its *contents* now include all codes — confirmed safe via
      `test_edgar_form4_all_codes.py` (7 parser tests + 1 end-to-end
      cache-write test) and by re-running P18's existing `test_form4_monitor.py`
      unmodified (still 100% green — it already self-filters to sale codes).
      - [x] **Flagged to whoever owns P20 Kestrel**: `filings_ingest.py`'s
        `_process_form4` was silently finding zero buy-code rows every day
        since the parser never emitted them; this change fixes that as a
        side effect. (Not touched under this doc — P20's own code is unchanged.)
- [x] `grading.py`'s P1/P2/N14 read `edgar/13f/form4/{date}.csv.gz` directly
      via `profiler.py`'s `_load_form4_window` (walks the trailing 100
      calendar days, `download_form4_filings(force=False)` per weekday so a
      missing day self-heals) — no new `EdgarDownloader` method needed.
- [ ] `efts_text_search(...)` — **still Phase 3, not built**, as scoped.

**`structural/` package** (`p19_penny_intraday/structural/`)
- [x] `xbrl_facts.py`: `shares_outstanding_series`, `split_adjust`, `cagr`,
      `decumulate_quarterly`, `cash_and_burn`, `operating_cash_flow_quarterly`,
      `proceeds_from_issuance`, `buybacks_quarterly`. **Required tests present
      and passing** (`test_xbrl_facts.py`, 14 tests): split-adjustment
      (`test_split_adjust_reverse_split_prevents_negative_cagr` — asserts the
      unadjusted CAGR *is* negative, the adjusted one is positive, i.e. proves
      the trap and the fix in one test) and de-cumulation
      (`test_decumulate_quarterly_no_q4_spike_artefact`). **Caught a real bug
      during review**: the first de-cumulation implementation didn't seed the
      YTD group's baseline from the fiscal year's own discrete Q1, so H1 read
      as a full-cumulative spike instead of H1-minus-Q1 — fixed before
      shipping (see the "seeded from the fiscal year's own discrete Q1" note
      in the function's docstring).
- [x] `grading.py`: N1,N2,N3,N4,N7,N8,N10,N11,N13,N14 / P1,P2,P3,P4,P5,P6,P7 +
      §5 baby-shelf N9 arithmetic + `dilution_urgency` + `insider_conviction`
      (renormalised over resolved signals only) + grade assignment (§7.5) +
      `coverage`. `test_grading.py`, 10 tests.
      - [x] FPI regression: `test_fpi_unresolvable_grades_c_never_a_or_b` —
        confirmed against a synthetic all-`None` profile.
      - [x] **Caught a real bug during review**: P3 (flat/declining share
        count, a positive signal) was initially aliased directly to the
        N3/N4 disqualifier result object — same resolved-ness is correct, but
        `.fires` needs the *opposite* polarity (P3 fires on low CAGR, N3/N4
        on high). Fixed with its own `_SignalResult`; regression test
        `test_p3_fires_on_flat_share_count_not_on_n3n4_disqualifier`.
      - [x] N10 magnitude threshold via `atm_proceeds_pct_mcap_threshold` config.
- [x] `cache.py`: per-ticker JSON cache, weekly TTL, daily delta check via
      `latest_filing_date` param. `test_structural_cache.py`, 7 tests.
- [x] `profiler.py`: orchestrates cache-check → fetch (EdgarDownloader +
      widened Form4 cache + yfinance) → grade → cache-write;
      `refresh_watchlist(entries, as_of, force)` entry point.
      `test_profiler.py`, 6 tests (mocked EdgarDownloader/yfinance, no network).
- [x] yfinance `.splits` — direct call in `profiler.py`, no new downloader.

**Momentum tier (log-only, spec §16 item 5)**
- [x] `metrics.py`: `classify_momentum_tier(signal, trigger_cfg) -> (score, tier)`.
      9 tests covering the gate (vol AND (price OR catalyst), and the loose
      OR-gate variant), T0→T3 boundaries.
- [x] `models/intraday_signal.py`: `momentum_score`/`momentum_tier` added;
      `severity` field kept (old-row backward compat) but no longer written;
      structural-axis fields (`structural_grade`, `dilution_urgency`,
      `insider_conviction`, `runway_quarters`, `disqualifiers`,
      `structural_coverage`) and all 8 outcome-label fields added.
- [x] `shadow_loop.py`: `_apply_momentum_and_structural` calls
      `classify_momentum_tier` and denormalises the cached `StructuralProfile`
      per poll (spec §12.1 — no join, point-in-time snapshot; a name with no
      cached profile yet still logs with `structural_grade=""`, decision #7).
      3 new tests in `test_shadow_loop.py`.

**Outcome labels**
- [x] `metrics.compute_same_day_labels` (pure function) + `shadow_loop.py`
      `_backfill_same_day_labels()`: `high_time`/`close_retention`/
      `mae_from_alert`/`mfe_from_alert`, using the first T1+ poll as the
      "simulated trigger point" alert price. 6 direct unit tests +
      2 `shadow_loop` integration tests (incl. the no-trigger-crossed case).
- [x] `label_backfill.py` (new): `ret_t1/t3/t5/t10` via `DataManager`;
      `dilution_event_within_5d` via `EdgarDownloader.get_recent_filings` +
      `resolve_tickers_to_ciks`; `reverse_split_within_180d` via
      `yf.Ticker().splits`. Self-gating on a 16-calendar-day age floor.
      `test_label_backfill.py`, 7 tests (EdgarDownloader/DataManager/yfinance mocked).

**CLI / scheduling**
- [x] `run_p19.py`: `profile-structural [--force]` and `label-backfill`
      subcommands wired (not directly unit-tested — matches the existing
      convention that this thin argparse-dispatch layer isn't tested
      separately from the classes it calls, same as v1's three subcommands).
- [x] `bin/scheduler/insert_p19_v2_schedules.sql` (new, additive to v1's file):
      Structural Profile at 13:10 UTC (after build-watchlist), Label Backfill
      daily at 12:00 UTC (self-gating, safe to run before data is ready).
      **Not yet applied on the Pi** — operational carry-over, see below.

**Reporting**
- [x] `shadow_report.py`: `by_grade` counts (flags any grade with n<30 per
      spec §15), `structural_coverage` percentiles, `unprofiled_count`,
      `low_coverage_count` (>20% of the day's names below coverage 0.4 flags
      as a possible FPI-share issue, StructuralSignals.md §2). One row per
      *ticker* (SQLite `MAX(ts)` grouping), not per poll. 5 new tests in
      `test_shadow_report.py`.

**Docs**
- [ ] `README.md` — not yet updated for the `structural/` package (v1's
      Status section still reads "Phase 0 — scaffold", stale even before this
      round; do together).
- [ ] `pipeline-specification-v2.md` §19 status line — not yet updated to
      reflect Phase 1.5 shipping (still reads "⚠️ do now").

**New operational carry-overs (this round)**
- [ ] Apply `bin/scheduler/insert_p19_v2_schedules.sql` on the Pi.
- [ ] Verify `profile-structural` actually runs cleanly against live EDGAR —
      everything here is tested against mocks; the real SEC endpoints,
      real Form 4 XML shapes (especially the `is_10b5_1_plan` footnote
      heuristic — see the flagged uncertainty in `edgar_downloader.py`), and
      real companyfacts payloads for the current watchlist (including `GRSD`,
      the FPI case) haven't been exercised yet.
- [ ] Once a few days of `profile-structural` runs exist, spot-check
      `shadow_report`'s new `by_grade`/`unprofiled_count`/`low_coverage_count`
      output against them.

---

## ✅ Phase 3 — N5/N6/N15/N16/P8/P9/P11, filings poll, sentiment attach (built 2026-08-18)

All code + tests below implemented and passing (0 pyright / 0 mypy on every
touched file; 148/148 P19 tests + 43/43 EDGAR-downloader tests — same one
unrelated `test_wikipedia_downloader.py` failure as Phase 1.5, still
untouched by this work). **Not yet done**: applying the filings-poll
scheduler entry on the Pi and confirming a real run — see the carry-over list
at the end of this section. See design-v2.md §9 for the full design writeup.

**Caught a real bug during review**: `_extract_auditor_name`'s fallback path
(no `/S/` signature line found → search the whole document) directly
contradicted its own neighbouring comment about why that's unsafe, and a
regression test proved it — an addressee line reading "...Shareholders **&**
Board of Directors..." matched the firm-designation marker and returned it as
the auditor name. Fixed by returning `None` (unresolved) when no signature
line is found at all, rather than falling back to a document-wide scan;
regression test `test_extract_auditor_name_only_searches_after_the_signature_line`.

**`EdgarDownloader` extensions**
- [x] `efts_text_search(cik, phrases, forms, start_dt, end_dt)` — built as
      speced (§3.2), verified live against the real EFTS endpoint.
- [x] `efts_filings_search(ciks, forms, start_dt, end_dt)` — new, multi-CIK
      no-phrase lookup for the intraday poll; verified live that `ciks`
      accepts a comma-list OR filter (unlike `forms`).
- [x] `get_auditor_name(cik, start_dt, end_dt, forms)` + `_extract_auditor_name`
      — EX-23.1 consent-exhibit extraction, verified live against a real
      filing (DeltaSoft Corp) during design; `test_edgar_efts_text_search.py`
      (new, 15 tests) covers both against mocks + fixed real-filing text.
- [x] `_fetch_filing_document` — single-URL document fetch by known filename
      (the EFTS hit's own `_id`), reusing the existing rate-limit/retry
      pattern rather than `_fetch_filing_xml`'s candidate-name guessing.

**`structural/` package**
- [x] `xbrl_facts.has_near_term_debt_maturity` (P9) — `LongTermDebtCurrent`
      fallback chain, verified live against Apple's companyfacts for the
      instant-fact shape. 4 tests in `test_xbrl_facts.py`.
- [x] `grading.py`: N5, N6, N15, N16, P8, P9, P11 evaluators +
      `_eval_short_interest_conditional`'s pre-grade branching (P11 excluded
      from `insider_conviction` at C/D, bumps `dilution_urgency` instead).
      12 new tests in `test_grading.py` (22 total), including the P11
      conditional-branch regression.
- [x] `profiler.py`: `_fetch_text_signals` (N5/N6/N16, per-(date,form) EFTS
      windows, not comma-lists), `_load_dg_window`/`_filter_dg` (P8, reusing
      P18's daily 13D/G cache), `_fetch_short_interest` (P11, direct
      yfinance `.info` call), `_infer_is_fpi`/`_earliest_filing_date` (N15).
      8 new tests in `test_profiler.py` (13 total).

**`StructuralProfile` / config**
- [x] New fields: `auditor_name`, `auditor_whitelisted`, `is_fpi`,
      `inst_13dg_activity_2q`, `no_debt_maturity_24m`,
      `short_interest_pct_float`, `days_to_cover`.
- [x] `P19StructuralConfig`: `n5_severity` (default `"C"`, spec's own
      precision-unmeasured fallback), `n5_convert_phrases`,
      `n6_going_concern_phrase`, `auditor_whitelist` (static list, not a live
      PCAOB integration — design-v2.md §9.2), `n15_ipo_window_months`,
      `n15_float_threshold_shares`, `p11_si_threshold`,
      `p11_days_to_cover_threshold`, `p11_dilution_urgency_bump`; P8/P9/P11
      added to `insider_conviction_weights`.

**Intraday filings poll (spec §9)**
- [x] New module `filings_poll.py`: `FilingsPoll` class, own SQLite table
      (`filings_events.sqlite`, dedup'd on `(date, ticker, accession, item)`)
      — deliberately **not** writing to the shared universe-wide 8-K index
      P17 depends on (design-v2.md §9.4 explains the collision risk this
      avoids). 6 tests in `test_filings_poll.py`.
- [x] `run_p19.py`: `filings-poll` subcommand.
- [x] `bin/scheduler/insert_p19_v2_schedules.sql`: new "P19 Intraday Filings
      Poll" job, `*/30 13-21 * * 1-5`.

**Sentiment attach (spec §10)**
- [x] New module `sentiment_cache.py`: `SentimentCache`, throttles the
      multi-provider batch fetch to once per TTL window (default 60 min)
      regardless of the 15-minute shadow-poll cadence — design-v2.md §9.8
      explains why per-poll would make P19 the heaviest consumer of these
      rate limits in the codebase. 4 tests in `test_sentiment_cache.py`.
- [x] `shadow_loop.py`: `_fetch_sentiment`/`_apply_sentiment`, reuses the
      previously-unpopulated `IntradaySignal.sentiment` scaffold field — no
      shadow-store schema change needed. 4 new tests in `test_shadow_loop.py`.

**Shadow store / reporting**
- [x] `is_fpi` added to schema v2's column set (still additive/idempotent —
      v2 hadn't shipped to the Pi yet, so no v3 migration layer needed) and
      to `IntradaySignal`/`shadow_loop.py`'s denormalisation.
- [x] `shadow_report.py`: `_fpi_share_stats` + a >20%-FPI-share flag
      (StructuralSignals.md §2). 2 new tests in `test_shadow_report.py`.

**Explicitly still deferred** (design-v2.md §9.10 has the full rationale for
each): **N12** (warrant overhang — no XBRL tag, no safe arithmetic shortcut
like N9's baby-shelf rule), **N9 above $75M float** (spec's own fallback is
to stay boolean until proven the limiting factor), **P10** (insider ownership
stability — needs Form 3 ingestion, not built). All three fields stay `None`
and depress `coverage`, never silently assumed resolved. FPI grade `"U"`
(distinct from plain `"C"`) also still deferred — not enough live-watchlist
FPI-share data yet to know if it's worth the formula change.

**New operational carry-overs (this round)**
- [ ] Apply the updated `bin/scheduler/insert_p19_v2_schedules.sql` on the Pi
      (adds the Intraday Filings Poll job on top of Phase 1.5's two).
- [ ] Verify `efts_text_search`/`efts_filings_search`/`get_auditor_name`
      against the real live watchlist — everything here is tested against
      mocks except the ad-hoc live-endpoint probes done during design (logged
      in design-v2.md §9.1/§9.2), which used specific known filings, not the
      actual watchlist population.
- [ ] Confirm `SENTIMENT_*` provider env vars/credentials are actually
      configured on the Pi — `collect_sentiment_batch_sync` degrades
      gracefully per-provider if not, but sentiment fields will just stay
      empty silently otherwise.
- [ ] Once a few days of `filings-poll` runs exist, spot-check
      `filings_events.sqlite` against the day's actual EDGAR filings for a
      couple of tickers by hand.

## 🚀 Roadmap (not detailed — see design-v2.md §Roadmap)

- [ ] **Phase 2**: Disposition Engine (matrix already specified, spec §8),
      dedup/escalation State Store, Alert Manager + per-disposition caps,
      Telegram delivery. The §8.2 escalation rule can now wire directly to
      Phase 3's `filings_poll.py` event log.
- [ ] **Phase 4**: Optuna calibration of both axes (spec §15, reuse P17
      `strategy_sim.py` harness), LULD halt detection, optional LLM alert
      summarizer, replace linear RVOL session-fraction with real U-shaped
      volume profile (v1 carry-over, unchanged).

## Known Issues / Constraints (carried from v1, still true)

- Primary feed = IBKR Gateway (delayed, free); ~100 market-data line budget;
  unique clientId (p19=19); Gateway must be up, handle reconnects.
- Free REST tiers lack real-time volume — kept as fallback / price cross-check
  only.

## Open Questions

From spec §17, ordered by what Phase 1.5 can/can't resolve:

- [x] *(v1 Q4)* Shadow store backend → **resolved this round: stay SQLite**,
      not DuckDB (design-v2.md decision #2).
- [x] Legacy shadow data status → **resolved this round: confirmed present**,
      1,741 rows / 3 days / 60 tickers (Task 0 above); migration path stays
      simple at this volume.
- [x] Form 4 buy-code access → **resolved this round: no new EDGAR call
      needed**, widen the existing daily bulk cache instead (design-v2.md
      §3.1). Dropped the `submissions.json`-scope question this replaced —
      moot now.
- [ ] Ticker→CIK coverage for the penny universe, especially FPIs and recent
      uplistings — measure once Layer 0 runs against a real watchlist. `GRSD`
      (already on the current watchlist) is a live FPI test case.
- [ ] EFTS phrase-match precision for N5 — still open. N5 is now **built**
      (fires at grade C per the spec's own fallback, `n5_severity` config),
      but the ~50-hand-labelled-filing precision study that would justify
      promoting it to D hasn't been done — that's manual research work, not
      something this round's implementation resolves.
- [ ] Shelf remaining-capacity: is the §5 baby-shelf arithmetic estimate good
      enough for the `dilution_urgency` fit, or does N9 above $75M need the
      prospectus parser? Test against `dilution_event_within_5d` once labels
      exist (Phase 4). Still deferred this round too (design-v2.md §9.10).
- [x] Does EFTS's live endpoint accept `ciks` + `q` params as assumed
      (design-v2.md §3.2)? → **resolved this round: yes, verified live.**
      `q` does an Elasticsearch `match_phrase`, `ciks` accepts a comma-list as
      an OR filter — but `forms` does **not** OR a comma-list (exact match
      only, same quirk as the existing 8-K catalyst path); every new caller
      queries one form type at a time. No highlighted snippet is returned,
      which is why N16 uses the EX-23.1-exhibit approach instead of trying to
      parse a search-result snippet (design-v2.md §9.1/§9.2).

## Testing Requirements

- [x] config + model unit tests (v1)
- [x] watchlist builder, RVOL calc, shadow store round-trip (v1)
- [x] **Phase 1.5**: `xbrl_facts` split-adjustment + de-cumulation assertions,
      `grading` N/P evaluators against synthetic fixtures, FPI-grades-to-C
      regression, `ShadowStore` v2 migration (incl. idempotency),
      `EdgarDownloader` Form4 generalisation (with the P18 sale-only
      regression re-run unmodified), `label_backfill` per-label unit tests,
      `profiler`/`cache` orchestration (mocked, no network),
      `compute_same_day_labels` + `classify_momentum_tier` unit tests,
      `shadow_report` grade/coverage reporting. 116 tests total across the
      P19 suite + the new EDGAR Form4 test file (52 of them in the 6 new test
      files); 0 pyright, 0 mypy on every touched file.
- [x] **Phase 3**: N5/N6/N15/N16/P8/P9/P11 evaluators, EFTS text-search +
      auditor-consent-exhibit extraction (incl. a real-filing-text fixture),
      13D/G presence proxy, near-term-debt-maturity XBRL lookup, P11's
      conditional-branch regression, filings-poll dedup + item filtering +
      the shared-cache-collision-avoidance test, sentiment-cache TTL
      throttling. 148 P19 tests total (+32 from Phase 1.5's 116) + 43 EDGAR
      tests total (+15 new); 0 pyright, 0 mypy on every touched file. One
      real bug caught by a test during review: `_extract_auditor_name`'s
      no-signature fallback scanned the whole document and could false-fire
      on an unrelated `&` — fixed, see the Phase 3 section above.
- [ ] Phase 2: trigger gate + dedup/escalation state machine (unchanged scope
      from v1 Tasks.md, not started).
