Postgres + Redis + Dramatiq + APScheduler is a solid stack. Short answer to your specific questions:

Good — totally reasonable. If reports and screeners are truly *one-off* commands in your model, you don’t have to create dedicated `reports` / `screeners` tables. Keep the persistent DB surface minimal (only what you actually need for scheduling, auditing and recovery) and treat reports/screener *definitions* as ephemeral inputs that are snapshotted into `runs` when the job is enqueued.

Below I give a clear, small set of patterns you can choose from (with pros/cons), a recommended pattern, and concrete implementation snippets (DB, FastAPI + dramatiq flow, scheduler behaviour, config for screener sets). Pick the one you like — the recommended approach is bolded.

---

# Option summary (high-level choices)

1. Snapshot-only (recommended)

   * Don’t store reports/screeners in DB. When a user requests a one-off report/screener, create a `runs` row and store the full request JSON in `runs.job_snapshot`. Enqueue the dramatiq task. For scheduled screeners, keep `schedules` in DB (or scheduler file) but store only schedule metadata (cron, target set name, or inline filter); each tick creates a `runs` row with snapshot.
   * Pros: minimal DB schema, full reproducibility via `job_snapshot`, easy to audit.
   * Cons: you still keep schedules somewhere (DB or scheduler store) if scheduling is required.

2. File-config for static sets + snapshot-only

   * Keep your 10 screener sets as a YAML/JSON file in your repo or a config dir on the Pi (`/etc/myapp/screener_sets.yml`). Scheduler and API load sets from that file. Schedules reference set name (string) only.
   * Pros: versionable (git), easy to edit, no DB table for sets.
   * Cons: editing via UI needs file-authoring logic or a separate admin UI that edits file/pushes commit.

3. Redis-only transient definitions

   * For truly ephemeral runs and ephemeral schedules, keep definitions in Redis (with TTL). Scheduler reads Redis and enqueues runs. `runs` still stores snapshot.
   * Pros: no DB writes for definitions, fast.
   * Cons: less durable (unless you persist Redis), complex to administer for long-term schedules.

4. APScheduler jobstore (SQL or Redis) but no job tables

   * Use APScheduler’s persistent jobstore (e.g., SQLAlchemyJobStore) for schedules; still do **not** store reports/screeners themselves. Each scheduled job when triggered writes a `runs` row with snapshot and enqueues.
   * Pros: APScheduler does schedule persistence for you.
   * Cons: introduces APScheduler DB tables (but not your own report/screener tables).

---

# My recommendation (concrete)

Use **Option 1 + Option 2**:

* Keep `runs` table (Postgres) as the single source-of-truth for *executions* and snapshots.
* Keep `schedules` table in Postgres (so you can manage schedules via API / UI robustly). A `schedules` row references either:

  * a `screener_set_name` (string), or
  * an inline `task_params` JSON (so UI can schedule ad-hoc screener with arbitrary filters).
* Keep the 10 screener sets in a versioned YAML file (Git), loaded by the API & scheduler at startup (and reloadable). This avoids adding a DB table for sets while allowing UI-admin via config commits or a separate admin endpoint that writes the YAML (if you want).
* For one-off reports/screeners: API receives JSON, inserts a `runs` row with `job_type='report'|'screener'` and `job_snapshot` containing everything needed, then enqueues a dramatiq task with `run_id`. No other table needed for reports/screeners.

This gives:

* No tables for reports/screeners.
* Full reproducibility through `runs.job_snapshot`.
* Robust scheduling via `schedules` table (you probably want schedules persisted).

---

# Concrete details & examples

## Minimal DB objects you keep

* `schedules` — (only schedule metadata and target pointer)
* `runs` — (all run lifecycle + job_snapshot)

Suggested `schedules` columns (Postgres):

```sql
CREATE TABLE schedules (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT,
  name TEXT,
  job_type TEXT NOT NULL,    -- 'screener' | 'report' | 'alert' ...
  target TEXT,               -- for screener: screener_set_name OR blank
  task_params JSONB,         -- optional inline params (filters, top_n, deliver)
  cron TEXT,                 -- cron expression
  enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
```

Notes:

* `target` lets you reference the screener set by name (string). Scheduler will expand name → tickers using the YAML sets file.
* If user prefers scheduling ad-hoc payloads, store full params in `task_params`. The scheduler should snapshot that into `runs` at enqueue time.

`runs` (as we discussed before) keeps `job_snapshot` JSONB:

```sql
CREATE TABLE runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_type TEXT NOT NULL,
  job_id BIGINT,                 -- optional: schedule id or NULL for ad-hoc
  user_id BIGINT,
  status TEXT,
  scheduled_for TIMESTAMPTZ,
  enqueued_at TIMESTAMPTZ DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  job_snapshot JSONB,            -- snapshot of the ad-hoc request or expanded params
  result JSONB,
  error TEXT
);
CREATE UNIQUE INDEX ux_runs_job_scheduled_for ON runs(job_type, job_id, scheduled_for);
```

* For ad-hoc report/screener: `job_id` = `NULL`, `job_snapshot` holds full payload.
* For scheduled run: `job_id` = schedule.id, `job_snapshot` contains expanded params (e.g., full ticker list, filter, top_n, deliver info).

## Screener sets file (YAML)

Place `screener_sets.yml` under `/etc/myapp/` or `config/` in your repo:

```yaml
# screener_sets.yml
sets:
  - name: "US_large_caps"
    tickers: ["AAPL","MSFT","GOOGL","AMZN","NVDA"]
  - name: "tech_midcap"
    tickers: ["OKTA","CRWD","ZS",...]
  ...
```

Load in your FastAPI app and scheduler:

```python
import yaml, pathlib
SETS_PATH = pathlib.Path("/etc/myapp/screener_sets.yml")
def load_sets():
    return yaml.safe_load(SETS_PATH.read_text())["sets"]

# helper: get_tickers(set_name)
```

If you want UI editability, expose endpoints that write the YAML (validate input first) and reload config; or store the YAML in a Git repo and have an admin UI that commits changes.

## API flow for one-off report (no DB report table)

FastAPI endpoint:

```python
@router.post("/reports/run")
async def run_report(payload: ReportRequest):
    run_id = insert_run(
        job_type="report",
        job_id=None,
        user_id=current_user.id,
        job_snapshot=payload.dict()
    )
    dramatiq_actor_run_report.send(run_id)   # dramatiq message
    return {"run_id": run_id, "status": "enqueued"}
```

`insert_run` writes a `runs` row; `job_snapshot` contains ticker/timeframe/indicators/delivery info.

Dramatiq actor receives `run_id`, reads `runs.job_snapshot` and executes. No separate `reports` table needed.

## API flow for one-off screener

FastAPI endpoint:

```python
@router.post("/screeners/run")
async def run_screener(payload: ScreenerAdhocRequest):
    # payload contains either "screener_set_name" OR "tickers" array + "filter"
    expanded_tickers = payload.tickers or get_tickers_from_config(payload.screener_set_name)
    snapshot = {"type":"screener","tickers": expanded_tickers, "filter": payload.filter, "top_n": payload.top_n, "deliver": payload.deliver}
    run_id = insert_run(job_type="screener", job_id=None, user_id=current_user.id, job_snapshot=snapshot)
    dramatiq_actor_run_screener.send(run_id)
    return {"run_id": run_id}
```

## Scheduling a screener (persist schedule but not screener def)

If user wants to run a screener every day at 7:00:POST `/schedules`:

* store schedule row: `job_type='screener'`, `target='US_large_caps'` (or `task_params` for inline filter), `cron='0 7 * * *'`.
* Scheduler on tick:

  * load schedule and `target` (if target is a set name, expand tickers from YAML)
  * build `job_snapshot` (full tickers + filter + deliver)
  * `INSERT` into `runs`, then enqueue dramatiq task with `run_id`.

This keeps `screener` definitions out of DB; only schedules (cron) are persisted.

---

# Worker & scheduler behaviour (concrete)

### Scheduler

* Runs as separate small process (recommended).
* Every minute:

  * find enabled `schedules` whose `next_run_at <= now` (you can compute next run with `croniter` or store `next_run_at` in the schedules table and update after enqueue).
  * For each such schedule:

    * compute `scheduled_for` timestamp (cron tick).
    * build `job_snapshot`: if `target` is a set name → use YAML to expand tickers; else if `task_params` present → use it.
    * `INSERT INTO runs (run_id, job_type, job_id, user_id, scheduled_for, job_snapshot) VALUES (...)`

      * use unique index on `(job_type, job_id, scheduled_for)` to avoid double-inserts from scheduler restarts.
    * enqueue dramatiq with `run_id`.
    * update schedule’s `next_run_at`.
* This scheduler doesn’t persist screener definitions — only schedule metadata.

### Workers

* Workers read `runs` job_snapshot and operate purely from that snapshot (no need to query `reports` or `screeners` tables — since they do not exist).
* This guarantees reproducibility — the worker runs what was requested at enqueue time.

---

# Idempotency & auditing

* Because `job_snapshot` captures the exact inputs, re-running a run is straightforward and auditable.
* Use `UX` unique index to prevent duplicate schedule enqueues.
* Workers `claim` run by `UPDATE runs SET status='running', started_at=now() WHERE run_id=:run_id AND status='pending' RETURNING 1` to avoid duplicate execution.
* Keep run history for N days (archival job) — you might want to keep runs for auditing even if you didn't persist the definition.

---

# Artifact handling & delivery (unchanged)

* Save images/CSV to artifact directory keyed by `run_id`.
* Store artifact paths in `runs.result`.
* Telegram/email delivery using snapshot `deliver` info.

---

# Pros/Cons recap of recommended approach

Pros

* Minimal DB schema — no tables for one-off commands.
* Full reproducibility (snapshot in `runs`).
* Easy UI + API for one-offs and schedules.
* Screener sets are versioned and editable via YAML (or kept as code).

Cons

* If you want users to *list/save* reusable report templates via UI, you’d need to add a small `report_templates` table later.
* Editing screener sets via UI requires writing YAML or implementing an admin endpoint to persist the YAML.

---

# Example code snippets (quick)

**Insert-run helper (psuedocode)**

```python
def insert_run(conn, job_type, job_id, user_id, job_snapshot, scheduled_for=None):
    run_id = uuid4()
    conn.execute(
        "INSERT INTO runs (run_id, job_type, job_id, user_id, scheduled_for, job_snapshot) VALUES (%s,%s,%s,%s,%s,%s)",
        (run_id, job_type, job_id, user_id, scheduled_for, json.dumps(job_snapshot))
    )
    return str(run_id)
```

**FastAPI ad-hoc screener endpoint example**

```python
@router.post("/screener/run")
async def run_screener(payload: ScreenerAdhoc, db=Depends(get_db)):
    tickers = payload.tickers or get_tickers_from_config(payload.screener_set_name)
    snapshot = {"tickers": tickers, "filter": payload.filter, "top_n": payload.top_n, "deliver": payload.deliver}
    run_id = insert_run(db, "screener", None, current_user.id, snapshot)
    run_screener_actor.send(run_id)
    return {"run_id": run_id}
```

**Worker claim skeleton**

```python
def claim_run(conn, run_id, worker_id):
    row = conn.execute(
      "UPDATE runs SET status='running', worker_id=%s, started_at=now() WHERE run_id=%s AND status='pending' RETURNING *",
      (worker_id, run_id)
    ).fetchone()
    return row  # None if already claimed
```

---

# Final notes & next steps

* If you want I can generate:

  1. a concrete Alembic migration for `schedules` + `runs` (with job_snapshot JSONB),
  2. the FastAPI endpoints for one-off report and screener (ad-hoc + schedule creation),
  3. small scheduler process that expands YAML sets and inserts `runs` + enqueues dramatiq.

Pick which of these you want me to produce now and I’ll give ready-to-drop code.
