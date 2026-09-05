# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] `base_plugin.py` — `PluginSpec` dataclass + `validate()` (allowlist/path-traversal)
- [x] `registry.py` — `PLUGIN_REGISTRY` (72 jobs across 15 categories), duplicate-name check
- [x] `raw_zone.py` — promoted from P22, pipeline-agnostic; P22's old path is a backward-compatible shim (all ~15 call sites + its own test suite unchanged)
- [x] `register_jobs.py` — `plan()`/`apply()`, `--dry-run` CLI diff against live `job_schedules`, merge-on-update (never clobbers an unmanaged `task_params` key)
- [x] `runner.py` — `--scope all|<category>|<name>`, `--dry-run`, mirrors scheduler's subprocess invocation shape
- [x] All 11 `bin/scheduler/insert_*.sql` files ported — `specs/{core,p05,p10,p15,p17,p18,p19,p20,p21,p22,screener,strategy_pack}_specs.py`, superseding those files and the 3 per-pipeline `jobs/register_jobs.py` modules
- [x] `specs/providers.py` — the 3 confirmed-uncalled provider downloaders (FRED, AAII, Fear & Greed weekly rebuild); see "Provider audit" below for why the other 4 candidates aren't here
- [x] README.md / docs/Requirements.md / docs/Design.md / docs/Tasks.md
- [x] `tests/test_registry.py`, `tests/test_raw_zone.py`, `tests/test_runner.py` (55 tests, all passing); 0 pyright/mypy errors
- [x] `PluginSpec.depends_on` + `dependency_status.py` (completion gate) — see docs/Design.md "Collection-before-consumption ordering". Wired into the 10 P20/P22 consumer scripts whose dependency was already explicitly documented in source comments: `run_data_health.py`, `run_gdelt_process.py`, `run_screen_turnaround.py`, `run_screen_spinoffs.py` (P20); `run_entity_resolution.py`, `run_acquirer_load.py`, `run_financial_facts_normalization.py`, `run_alias_matching.py`, `run_trial_normalization.py`, `run_patent_expiry_normalization.py` (P22). `tests/test_dependency_status.py` covers the gate's decision logic via monkeypatching (no DB needed); the actual `job_schedules`/`job_schedule_runs` query was verified manually against a live-connected DB, not in the pytest suite (no DB fixture exists for it — same gap as the 14 pre-existing DB-connectivity test errors below).

### Provider audit (why only 3 of the original 7 candidates got a new schedule)
Before writing new wrapper scripts for cboe/fred/fear_greed/aaii/wikipedia/
russell3000/openfigi, each was grepped for existing callers across `src/` —
this changed the plan significantly:
- **cboe, wikipedia (index_changes), russell3000** — already refreshed by
  `p15_daily.py` / `p15_weekly.py` (see next section). Adding separate rows
  would double the network calls against the same cache files. **Not added.**
- **russell3000** is also touched daily by P05's `universe_loader.py` (TTL
  90 days, so mostly no-ops) — another reason not to add a 4th caller.
- **fear_greed** is partially covered: P17's `reporting_agent.py` calls
  `.load()` (incremental only) on P17's daily cron. The class's own
  documented "Friday full rebuild" was never actually invoked anywhere —
  **added just that missing piece** (`Fear & Greed Weekly Archive Rebuild`).
- **fred, aaii** — no caller anywhere in `src/`. **Added**, using each
  downloader's own existing CLI (both already print
  `__SCHEDULER_RESULT__` — no new wrapper script needed).
- **openfigi** — `OpenFigiMapper` has no "download everything" concept; it
  resolves whatever CUSIPs a caller (P18's backfill) hands it, on demand.
  Nothing to schedule. **Not added, and shouldn't be.**

### Discovery: two live jobs existed in NO file anywhere
While investigating the cboe/wikipedia/russell3000 question, direct DB
queries turned up `P15 Pipeline – daily bundle` (target `p15_daily`, cron
`0 13 * * 2-6`) and `P15 Pipeline – weekly bundle` (target `p15_weekly`, cron
`0 14 * * 6`) — running live, not present in any SQL file or
`register_jobs.py`. Both are now in `specs/p15_specs.py`, transcribed
verbatim from the live row (cron, timeout, `notification_rules`, and the
exact en-dash `–` (U+2013) in each name — using a plain hyphen would have
inserted a duplicate row instead of matching the existing one). This is the
same class of "DB-only, no file source" gap P20 already had two of.

### ✅ RESOLVED — live/documented mismatches (2026-09-04, human-confirmed)

`register_jobs.py --dry-run` originally showed 4 rows whose live values
diverged from every documented source in a way that looked deliberate, not
drift. Resolved by explicit user decision, spec files updated accordingly:

- **`EMPS2 Morning Scan`**: documented wins. Spec unchanged
  (`35 9 * * 1-5`, Phase 1/2 notifications on) — `apply()` will restore
  weekday-only cron and re-enable email/Telegram notifications on live
  (live had drifted to daily-incl.-weekends with notifications silently off).
- **`EMPS2 Evening Scan (8PM CET)`**: documented wins. Spec unchanged
  (`0 14 * * 1-5`, notifications on) — `apply()` will restore the weekday
  schedule and time, and re-enable notifications (live had drifted to a
  different daily time with notifications off).
- **`EMPS3 Morning Scan`**: documented wins. Spec unchanged (notifications
  on) — `apply()` will re-enable notifications (cron was already correct).
- **`P19 Structural Profile`**: live wins. `specs/p19_specs.py` updated to
  `timeout_seconds=3600` (was `1800`) — the wider live value is being kept
  as the deliberate fix for a real production timeout; no live change needed.

### ✅ RESOLVED — orphan audit (2026-09-04)

Queried the live `job_schedules` table directly (all 53 rows, single user_id,
zero duplicates) and diffed against `PLUGIN_REGISTRY` by name. Found exactly
2 rows not in the registry — **both were jobs missed during migration, not
stale/unneeded rows**:

- **`FINRA TRF Daily Download`** (`src/ml/pipeline/p06_emps2/trf_downloader.py`,
  `job_type=data_processing`) — a clean 1:1 fit, ported verbatim into
  `specs/core_specs.py` (cron, timeout, `notification_rules` copied from the
  live row).
- **`portfolio_pnl_alert`** (`target=portfolio.pnl_alert`) — **excluded on
  purpose**. Its `job_type` is `"alert"`, dispatched by
  `SchedulerService._execute_alert_job` — a structurally different path than
  every `PluginSpec` (which hardcodes `job_type="data_processing"` and a
  subprocess/`script_path` shape via `_execute_data_processing_job`).
  Registering it here would silently flip its dispatch type on `apply()` and
  likely break it. This is exactly the "downstream consumer" boundary
  `docs/Design.md` already draws (`src/portfolio/pnl_alert/` reads data other
  plugins collect; it doesn't collect anything itself) — confirms the
  boundary is correct, not a gap. See docs/Design.md's new "Job-type
  boundary" note.

Conclusion: **there is no cruft to delete in `job_schedules` today.** No
removal SQL was written, since there's nothing for it to target.

### 🚀 PLANNED ENHANCEMENTS
- [ ] Data-completeness gate (stronger than the schedule-run-success gate):
      per data source, check e.g. companies-landed vs. universe size rather
      than just "did the job exit 0" — needed to actually catch the
      ClinicalTrials-Ingest-class failure (partial success, not a crash).
      Per-pipeline, not generic.
- [ ] Extend `depends_on` + the gate to the rest of P20/P22's chain once
      more relationships are confirmed (only the 6 P22 + 4 P20 explicitly
      documented in source comments were wired this pass — others, e.g.
      `P20 Sentiment Aggregate`'s likely dependency on GDELT Process/Social
      Sentiment Poll/AV Sentiment, were deliberately left out for now since
      they were inferred from job ordering/naming, not stated outright, and
      a wrong `depends_on` would make a job silently defer forever).
- [ ] Split the monolithic pipelines (P18, P05, P17, EMPS2/3, screeners,
      Strategy Pack) into a separate collection step + a gated consumer step
      — out of scope for this pass (see chat: user confirmed P20/P22-only
      scope for now). P18 was the pipeline that prompted this whole
      dependency-ordering feature; it's the natural next candidate.
- [x] `apply()` run for real against **production** `job_schedules` on the Pi
      (2026-09-05): `--dry-run` and the real run both reported
      `72 unchanged, 0 inserts, 0 updates` — confirms every row already
      matched the registry exactly, including the 4 rows hand-reconciled
      2026-09-04. The 3 superseded `jobs/register_jobs.py` modules
      (P20/P21/P22) are deleted; all 12 `bin/scheduler/insert_*.sql` files
      (11 pipeline files + the original core `insert_schedules.sql` — one
      more than this doc originally counted) are archived under
      `bin/scheduler/archive/`, not deleted. Runbooks that pointed at the old
      files (`p20_kestrel/README.md` + `docs/manual_run.txt`,
      `p21_momentum/README.md`, `p22_biotech_ma/README.md`,
      `p18_institutional_flow_tracker/README.md`, `bin/scheduler/README.md`)
      were updated to point at `src.data.pipeline.register_jobs` instead.
- [ ] Update P22's internal imports to `src.data.pipeline.raw_zone` directly
      and retire the shim, once there's a reason to touch those files anyway.
- [ ] Consider whether `apply()` should refuse to touch a row whose live
      values differ from every known documented source without an explicit
      override flag, rather than relying purely on human review of dry-run
      output before running `apply()`.

## Technical Debt
- None yet — this is a new module.

## Known Issues
- `bin/scheduler/insert_schedules.sql`'s `VIX Daily Monitor` row points at
  `src/data/vix.py`, which does not exist in the current codebase — the live
  `job_schedules` row was separately hand-corrected to
  `src/data/downloader/vix_downloader.py`; `specs/core_specs.py` uses the
  live/corrected path. The SQL file itself is stale and should not be used as
  a reference going forward.

## Testing Requirements
- [x] `tests/test_registry.py` — no dup names, every spec validates
- [x] `tests/test_raw_zone.py` — dedup/immutability/known_from, ported from P22
- [x] `tests/test_runner.py` — scope resolution, result parsing, subprocess smoke tests
- [ ] Integration test: `register_jobs.py --dry-run` against a scratch/staging
      DB seeded from a snapshot of production `job_schedules` (currently
      verified ad hoc against a live-connected DB, not via an automated
      fixture).

## Documentation Updates
- [x] `registry.py`'s "Migration status" docstring reflects full coverage.
- [x] The 4 flagged rows are resolved — see "RESOLVED" sections above.
