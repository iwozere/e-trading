# Requirements

## Python Dependencies

No new third-party dependencies — this module only uses the standard library
plus packages already required elsewhere in the repo (SQLAlchemy, via
`src.data.db`).

## External Dependencies

- `src.data.db.core.database` (`session_scope`) — DB session management.
- `src.data.db.models.model_jobs` (`Schedule`) — the `job_schedules` ORM model.
- `src.notification.logger` (`setup_logger`) — project logging convention.
- `config.donotshare.donotshare` (`DATA_CACHE_DIR`) — cache root, with the
  same import-fallback pattern used throughout `src/data/`.

## External Services

- PostgreSQL — `job_schedules` is a live production table; `register_jobs.py`
  reads and writes it via the project's existing DB connection config. No new
  service or credential is introduced.

## System Requirements

Negligible — this module orchestrates subprocesses (`runner.py`) and issues
ORM upserts (`register_jobs.py`); it does no heavy computation itself. Actual
resource needs are whatever the underlying plugin scripts already require
(documented in each pipeline's own `Requirements.md`).

## Security Requirements

- `PluginSpec.validate()` enforces the same script-path allowlist and
  path-traversal protection as
  `SchedulerService._execute_data_processing_job` (`src/scheduler/scheduler_service.py`)
  — see `base_plugin.py`'s `ALLOWED_SCRIPT_DIRS`. Keep the two lists in sync
  when onboarding a new pipeline directory.
- `register_jobs.py` writes to a live production table
  (`job_schedules`). Always run `--dry-run` first against an unreviewed
  change and read the diff before applying — see `docs/Design.md` §Rollout
  safety.

## Performance Requirements

- `register_jobs.py --dry-run` / `apply()` do one query per plugin against
  `job_schedules` — fine at the current registry size (dozens of rows); would
  need batching if the registry grows into the thousands.
- `runner.py` runs plugins sequentially, respecting each `PluginSpec`'s
  `timeout_seconds` — a full `--scope all` run's wall-clock time is the sum
  of every plugin's own runtime, not built for low-latency use.
