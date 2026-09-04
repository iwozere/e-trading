# Data Pipeline (Plugin Registry)

## Overview

Single catalog of every scheduled data-download/ingest job in the codebase,
plus the tooling to (a) regenerate their `job_schedules` rows and (b) run any
subset of them standalone. It does not fetch data itself — each plugin points
at an existing (or new, thin) `run_*.py` script that already knows how to
download and cache its own data.

## Features

- One `PLUGIN_REGISTRY` listing every data source in scope — see `registry.py`.
- `register_jobs.py` generates `job_schedules` rows from the registry, with a
  `--dry-run` mode that diffs against the live table before writing anything.
- `runner.py` runs any subset of plugins (`--scope all|<category>|<name>`) as
  a standalone CLI — a full-cache rebuild that needs neither Postgres nor
  APScheduler running.
- `raw_zone.py` — a shared, content-hashed/immutable landing-zone writer any
  plugin can use (promoted from P22's original implementation).

## Quick Start

```python
from src.data.pipeline.registry import PLUGIN_REGISTRY, get_by_category

# Everything currently registered
for spec in PLUGIN_REGISTRY:
    print(spec.name, spec.cron, spec.script_path)

# One pipeline's jobs
p22_jobs = get_by_category("p22")
```

```bash
# See what would change in job_schedules before writing anything
python -m src.data.pipeline.register_jobs --dry-run

# Apply
python -m src.data.pipeline.register_jobs

# Rebuild one category's cache on demand (e.g. after a new-machine bootstrap)
python -m src.data.pipeline.runner --scope p22
```

## Adding a new data source

1. Write (or reuse) a `run_*.py` script under an allowed directory
   (`src/data/`, `src/ml/pipeline/`, `src/scheduler/scripts/`,
   `src/screeners/`, `src/strategy_pack/`) that does the actual fetch/cache
   and prints `__SCHEDULER_RESULT__:{json}` on success — the same convention
   every existing pipeline script already follows.
2. Add a `PluginSpec` for it to the relevant `specs/*_specs.py` group (or a
   new group file, then append it to `registry.py`'s `_GROUPS`).
3. Run `python -m src.data.pipeline.register_jobs --dry-run`, review the
   diff, then apply.

No scheduler code, no new SQL file, no central dispatcher edit.

## Integration

This module integrates with:
- `src.scheduler` — `job_schedules` rows generated here are read and
  dispatched by the existing `SchedulerService`; this module never touches
  the running scheduler.
- `src.data.db` — `Schedule` model / `session_scope()` for reading and
  writing `job_schedules`.
- `src.ml.pipeline.*` — every plugin script lives in and is owned by its
  originating pipeline; this module only catalogs and (re)runs them.

## Configuration

- `DATA_CACHE_DIR` (env var, default `c:/data-cache`) — root for
  `raw_zone.py`'s default landing zone (`DATA_CACHE_DIR/raw`).
- `SCHEDULER_SYSTEM_USER_ID` (env var, default `2`) — `user_id` used for all
  rows `register_jobs.py` writes to `job_schedules`.

## Related Documentation
- [Requirements](docs/Requirements.md)
- [Design](docs/Design.md)
- [Tasks](docs/Tasks.md)
