# Design

## Purpose

The codebase downloads/caches external data through ~45 independent jobs
spread across `src/data/downloader/` and five ML pipelines (P18–P22). There
was no single place that knew the full catalog of "what we download," and
scheduling had drifted into two incompatible registration paths writing to
the same `job_schedules` table: 11 hand-written SQL seed files
(`bin/scheduler/insert_*.sql`) and 3 separate Python `register_jobs.py`
modules — already the source of one production incident (P20's
`register_jobs.py` docstring records a 2026-08-30 (user_id, name) mismatch
that silently created a duplicate set of 21 jobs).

This module is a thin orchestration layer that (a) is the one catalog of
every data source in scope, (b) can run "download everything" standalone,
and (c) lets a new data source join by adding one plugin file + one registry
line.

## Architecture

```
base_plugin.py   — PluginSpec dataclass + validate() (allowlist/path-traversal check)
registry.py      — PLUGIN_REGISTRY, built from specs/*_specs.py groups
raw_zone.py      — shared content-hashed/immutable landing-zone writer
register_jobs.py — PLUGIN_REGISTRY -> job_schedules upserts, with --dry-run
runner.py        — standalone "run this scope's scripts" CLI
specs/           — one file per pipeline/group of PluginSpecs
```

### Why `PluginSpec` is data, not a class hierarchy

Every existing job is already a self-contained `run_*.py` script with its own
CLI and `__main__` block, printing `__SCHEDULER_RESULT__:{json}`. A
`PluginSpec` just describes one of these declaratively (name, cron,
script_path, timeout) — mirroring the `_JOB_SPECS` list pattern already used
and trusted in `p20_kestrel/jobs/register_jobs.py`,
`p21_momentum/jobs/register_jobs.py` and
`p22_biotech_ma/jobs/register_jobs.py`. Migrating those three pipelines was a
straight data copy (see `specs/p20_specs.py`, `p21_specs.py`, `p22_specs.py`).

### Why `runner.py` shells out instead of calling plugin code in-process

`runner.py` invokes each plugin's script as a subprocess — the same
`python <script_path> <script_args>` shape
`SchedulerService._execute_data_processing_job` uses in production
(`src/scheduler/scheduler_service.py`). This means: (a) zero changes needed
to the ~45 existing scripts to bring them into the registry, (b) a scheduled
cron run and a manual `runner.py --scope all` run of the same plugin behave
identically, (c) no cross-job state leakage risk from running dozens of
unrelated modules in one process.

### Rollout safety (production scheduler table)

`job_schedules` is live in production. `register_jobs.py --dry-run` computes
and prints a full diff (inserts / field-level changes / unchanged) against
the current table before any write — this is the primary safety net, not an
afterthought, given the target is a table `SchedulerService` reads to decide
what to run. The 11 `bin/scheduler/insert_*.sql` files remain the source of
truth for their pipelines' schedules until each is ported here and a dry-run
against production confirms an empty diff (see `docs/Tasks.md`) — they are
archived, not deleted, once superseded.

## Data Flow

1. A plugin's script fetches from its external source and caches under
   `DATA_CACHE_DIR/<source>/…` (existing per-provider convention) or, for new
   sources, via `raw_zone.write()`.
2. `register_jobs.py` reads `PLUGIN_REGISTRY` and upserts one `job_schedules`
   row per `PluginSpec`.
3. In production, `SchedulerService` (unchanged) reads `job_schedules` and
   dispatches each job on its cron via `_execute_data_processing_job`.
4. Outside the scheduler, `runner.py` reads `PLUGIN_REGISTRY` directly and
   runs a chosen scope's scripts the same way, for ad hoc full-cache rebuilds.

## Design Decisions

- **Registration is idempotent, keyed on `(user_id, name)`** — same
  convention as every predecessor `register_jobs.py`, so a re-run never
  duplicates a row; it either updates or is a no-op.
- **`job_type` is normalized to `"data_processing"` everywhere** — P21's
  original rows used `job_type="script"` with a plain (non-dotted) `target`;
  both dispatch through the same executor, so this is a one-time cosmetic
  normalization, surfaced via `--dry-run` before it's applied, not a silent
  behavior change.
- **Job-type boundary: only `data_processing`/`script` jobs belong here.**
  `PluginSpec` hardcodes `job_type="data_processing"` and a subprocess/
  `script_path` dispatch shape (`_execute_data_processing_job`). Other
  `job_schedules` job types — e.g. `"alert"`
  (`SchedulerService._execute_alert_job`, used by `portfolio_pnl_alert`),
  `"screener"`, `"report"` — are dispatched through entirely different
  executor methods and must **not** be added as `PluginSpec`s: `apply()`
  would silently flip their `job_type`/`target` shape and likely break them.
  Confirmed 2026-09-04 while auditing `job_schedules` for rows missing from
  the registry — see docs/Tasks.md's "orphan audit".
- **`SYSTEM_USER_ID` defaults to `2`**, matching the P20/P22 convention (P21's
  old script defaulted to `1`, but both read the same
  `SCHEDULER_SYSTEM_USER_ID` env var, so in any environment where that var is
  actually set — everywhere this matters — they already resolved to the same
  value).
- **Scheduler code is untouched.** `SchedulerService`'s dispatch, allowlist,
  and `__SCHEDULER_RESULT__` parsing are the ground truth this module mirrors
  (`base_plugin.py`'s `ALLOWED_SCRIPT_DIRS`, `runner.py`'s
  `_parse_script_output`) rather than a dependency on — keeping this module
  free to evolve without risking the live scheduler process.

## Collection-before-consumption ordering

Cron staggering ("run 30 minutes after X") is only a hope, not a guarantee,
and it already failed once in production: P22's `ClinicalTrials Ingest`
timed out at 7200s having covered only 215/1705 companies, yet `Alias
Matching` was still going to fire 60 minutes after ingest *started* and
would have silently read the partial data as if it were complete (see
`specs/p22_specs.py`'s comment on that job).

`PluginSpec.depends_on: List[str]` declares which other plugins' data a
plugin's script reads — checked for typos/dangling references at import
time (`registry.py`'s `_check_dependencies_exist`), and read at runtime by
`dependency_status.require_dependencies_or_defer(plugin_name)`, which a
consumer script calls at the top of its `run()`/`main()`:

```python
ready, statuses = require_dependencies_or_defer("P22 Alias Matching")
if not ready:
    return deferred_result(statuses)
```

The gate queries `job_schedules`/`job_schedule_runs` (the scheduler's own
run-history tables) for whether each dependency's most recent run *today*
completed successfully — not just whether enough wall-clock time has
passed. A dependency that hasn't run yet, is still running, crashed, or was
killed for timing out all cause the consumer to defer (log a warning, return
a `"deferred": true` result, exit 0 — the established "safe no-op" idiom
already used by `p19_penny_intraday/label_backfill.py` and
`p21_momentum/calendar.py`) rather than run on data that isn't ready.

**Known limitation**: this only checks schedule-run *success*, not data
*completeness* — a dependency that exits 0 having only partially finished
(exactly the ClinicalTrials Ingest case above, on the day it happened) is
not caught. Catching that needs a completeness check specific to what
"done" means for that data source (e.g. companies-landed vs. universe
size), which is a separate, per-pipeline enhancement — see docs/Tasks.md.

Cron timing itself is unchanged and still what triggers each job; `depends_on`
does not turn this into a DAG scheduler, it only gates what a triggered job
actually does. Currently wired into 10 P20/P22 consumer scripts (6 P22, 4
P20) whose `depends_on` relationship was already explicitly documented in
the source comments this registry was ported from — not into every plugin,
and deliberately not into the monolithic pipelines (P18, P05, P17, EMPS2/3,
screeners, Strategy Pack) that fetch and process in one script with nothing
to gate against yet.

## Integration Patterns

- New pipelines integrate by adding a `specs/*_specs.py` file, not by
  implementing an interface — see README.md "Adding a new data source".
- Downstream consumers (e.g. `src/portfolio/pnl_alert/insider_activity.py`)
  are unaffected: they read the cache each plugin's script already
  maintains and are not part of this registry.
