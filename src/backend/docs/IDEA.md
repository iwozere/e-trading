Nice — Postgres + Redis + Dramatiq + APScheduler is a solid stack. Short answer to your specific questions:

* Yes — keep **alerts** and **schedules** in separate tables (they are different domain objects and often have different fields).
* Yes — your **runs** table should record which job produced a run, and you should store **job_type** in that table as well. Do **not** rely on a single `job_id` column alone (which is ambiguous). Use either a polymorphic pair `(job_type, job_id)` or a unified parent `jobs` table — both are valid; I’ll show the polymorphic approach because it’s simple and flexible.

Below I give a pragmatic schema, design rationale, SQL `CREATE TABLE` examples (Postgres), recommended indexes, idempotency constraints, examples of how to snapshot configs for reproducibility, and a small SQLAlchemy model sketch you can drop into your FastAPI app.

# Why `job_type` + `job_id` (polymorphic) is better

* Alerts and schedules are separate tables with separate lifecycles and fields. A run must point back to *which* domain object triggered it. `job_type` tells you whether `job_id` references `alerts`, `schedules`, `strategies`, etc.
* Easier queries: `SELECT * FROM runs WHERE job_type='alert' AND job_id=42` is explicit.
* Keeps DB normalized without complex foreign-key tricks (you avoid cross-table FK constraints).
* If you later add a common `jobs` table (shared metadata), migrating is straightforward.

# Requirements for runs table

Must capture:

* canonical `run_id` (UUID) — unique per execution (idempotency)
* `job_type` (text enum or text)
* `job_id` (bigint/uuid depending on your PKs)
* `user_id` or `owner_id` (for multi-user systems / permissions)
* scheduling metadata: `scheduled_for` (when it was supposed to run), `enqueued_at`/`queued_by`
* lifecycle timestamps: `started_at`, `finished_at`
* `status` (enum: pending, running, success, failed, cancelled)
* `attempt` and `max_attempts` (for retries)
* `worker_id` / `hostname` (who processed)
* `result` / `output` (JSONB), `error` (text) — limit size or move large artifacts to blob store
* `trace_id` / `correlation_id` for observability
* `job_snapshot` (JSONB) — snapshot of the alert/schedule config at enqueue time (reproducibility)
* `priority`, `locked_until` (optional locking for manual claim)
* `duration_ms` (computed)

# Suggested Postgres schema (practical)

```sql
-- ENUM for status
CREATE TYPE run_status AS ENUM ('pending','scheduled','running','success','failed','cancelled');

-- alerts table (example)
CREATE TABLE alerts (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL,
  name TEXT NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT true,
  interval_minutes INT, -- or cron expression
  rule JSONB NOT NULL,
  actions JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- schedules table (example)
CREATE TABLE schedules (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL,
  name TEXT NOT NULL,
  cron TEXT NOT NULL,
  task JSONB NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- runs table (polymorphic)
CREATE TABLE runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_type TEXT NOT NULL,        -- 'alert' | 'schedule' | 'strategy'
  job_id BIGINT NOT NULL,        -- references alerts.id or schedules.id (no FK)
  user_id BIGINT,
  status run_status NOT NULL DEFAULT 'pending',
  attempt INT NOT NULL DEFAULT 0,
  max_attempts INT NOT NULL DEFAULT 3,
  scheduled_for TIMESTAMPTZ,     -- when this run was scheduled for (cron tick)
  enqueued_at TIMESTAMPTZ DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  worker_id TEXT,                -- hostname/process id or dramatiq worker id
  locked_until TIMESTAMPTZ,      -- optional lock expiry if using manual claim
  duration_ms INT,
  priority INT DEFAULT 100,
  trace_id UUID,                 -- optional for tracing
  job_snapshot JSONB,            -- config snapshot for reproducibility
  result JSONB,                  -- output/results summary
  error TEXT,                    -- short error message
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- useful indexes
CREATE INDEX idx_runs_job ON runs (job_type, job_id);
CREATE INDEX idx_runs_status ON runs (status);
CREATE INDEX idx_runs_enqueued_at ON runs (enqueued_at);
CREATE INDEX idx_runs_scheduled_for ON runs (scheduled_for);
CREATE INDEX idx_runs_user ON runs (user_id);
```

Notes:

* I used `gen_random_uuid()` (Postgres `pgcrypto` or `uuid-ossp`) for `run_id`. Ensure the extension is enabled: `CREATE EXTENSION IF NOT EXISTS pgcrypto;`
* `job_snapshot` stores the JSON config of the alert/schedule/strategy as it was when enqueued — crucial for reproducibility and debugging.
* `result` should be a short summary (JSON) only; store large artifacts separately (S3/minio or local FS) and store a link in `result`.

# Idempotency and uniqueness constraints

To prevent double-enqueuing/execution you can add constraints depending on business logic.

Common patterns:

1. **Unique per scheduled tick** — do not allow two runs for the same job at the same scheduled time:

```sql
CREATE UNIQUE INDEX ux_runs_job_scheduled_for ON runs (job_type, job_id, scheduled_for);
```

This ensures the scheduler cannot enqueue the same job twice for the same `scheduled_for`. If you need multiple simultaneous runs, leave this out.

2. **Run-level idempotency** — workers receive `run_id`. Before executing they `UPDATE runs SET status='running', started_at=now(), worker_id=... WHERE run_id = :run_id AND status IN ('pending','scheduled') RETURNING *`. If no row changed, treat as already processed.

3. **Claiming pending runs** (if queueless/DB-polling) — `UPDATE runs SET status='running', worker_id=..., started_at=now() WHERE run_id = :run_id AND status='pending' RETURNING run_id;` Use the returned row presence to know claim success.

# Snapshotting job config

When scheduler enqueues a run, fetch the job's config from `alerts` or `schedules` and store it into `runs.job_snapshot`. This guards against changes after enqueue: you will always know what parameters the run used.

Example insert (pseudo):

```sql
INSERT INTO runs (run_id, job_type, job_id, user_id, scheduled_for, job_snapshot)
VALUES (:run_id, 'alert', :alert_id, :user_id, :scheduled_for, :job_snapshot::jsonb);
```

# Handling retries

* Use `attempt` and `max_attempts`. Worker on failure increments `attempt` and sets `status='pending'` or `status='scheduled'` with `enqueued_at=now() + delay` (requeue). Or use Dramatiq's retry mechanism and still update your `runs` row accordingly.
* Record `error` text and stack (truncate) and include a short `result` summary.

# Archiving & retention

* `runs` can grow large. Add a background job to archive old runs to an archive table or S3/CSV and delete old rows (e.g., keep last 90 days).
* Use partitioning by time (`created_at`) if you expect very high volume.

# Example SQLAlchemy models (minimal)

```python
# models.py
from sqlalchemy import Column, BigInteger, Text, JSON, Boolean, TIMESTAMP, Integer, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
import enum
import uuid
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RunStatus(enum.Enum):
    pending = "pending"
    scheduled = "scheduled"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    name = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True)
    interval_minutes = Column(Integer)
    rule = Column(JSONB)
    actions = Column(JSONB)

class Schedule(Base):
    __tablename__ = "schedules"
    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    name = Column(Text, nullable=False)
    cron = Column(Text, nullable=False)
    task = Column(JSONB)
    enabled = Column(Boolean, default=True)

class Run(Base):
    __tablename__ = "runs"
    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(Text, nullable=False)
    job_id = Column(BigInteger, nullable=False)
    user_id = Column(BigInteger)
    status = Column(Enum(RunStatus), default=RunStatus.pending, nullable=False)
    attempt = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    scheduled_for = Column(TIMESTAMP(timezone=True))
    enqueued_at = Column(TIMESTAMP(timezone=True))
    started_at = Column(TIMESTAMP(timezone=True))
    finished_at = Column(TIMESTAMP(timezone=True))
    worker_id = Column(Text)
    job_snapshot = Column(JSONB)
    result = Column(JSONB)
    error = Column(Text)
```

# Example lifecycle pseudocode (scheduler + worker)

Scheduler (APScheduler task):

1. For each enabled alert/schedule due now:

   * Build `job_snapshot` from DB row (serialize)
   * `INSERT INTO runs (run_id, job_type, job_id, user_id, scheduled_for, job_snapshot, enqueued_at) VALUES (...)`
   * Enqueue dramatiq task with payload `{run_id, job_type, job_id}`.

Worker (dramatiq actor):

1. Receive message with `run_id`.
2. `UPDATE runs SET status='running', started_at=now(), worker_id=:id WHERE run_id=:run_id AND status IN ('pending','scheduled') RETURNING *`
3. If no row returned → already processed or claimed — exit.
4. Execute job using `job_snapshot` (or fetch current job if needed).
5. On success: `UPDATE runs SET status='success', finished_at=now(), duration_ms=..., result=:json WHERE run_id=:run_id`
6. On failure: increment attempt and either requeue or set status 'failed' depending on `attempt < max_attempts`.

# Extra touches & recommendations

* Use a `job_type` enum in Postgres (for clarity) or keep it as TEXT for extensibility.
* Keep a small `jobs` view that unifies alerts/schedules for admin UI (SELECT across both).
* Store user-facing reasons and short codes in `result` so UI can show concise run history.
* Add `correlation_id / trace_id` to link logs, dramatiq message id, and run row for observability.
* Consider a `job_def` table if you later want strict foreign keys or shared metadata (name, owner, visibility).

---

If you want, I can:

* generate the exact SQL `CREATE TABLE` statements (with extensions, types, and indexes) ready for Alembic migration; **or**
* produce the SQLAlchemy models + Alembic revision file skeleton; **or**
* provide the full scheduler → enqueuer → dramatiq worker code snippet that demonstrates the lifecycle with `job_snapshot` writes and idempotent claim logic.

Which would you like next?
