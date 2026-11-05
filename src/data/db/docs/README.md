Great q — you’ve got a couple similarly named pieces floating around, so here’s a crisp map of responsibilities and exactly how they should line up so you don’t have overlapping “repos”.

# Recommended layering (who does what)

1. **Application layer (bots, strategies)**

   * Calls *simple*, intention-level functions (e.g., `telegram_service.add_alert(...)`).
   * Knows nothing about ORM, sessions, or SQL.

2. **Service interface (your `telegram_service.py`)**

   * Orchestrates use-cases: input validation, composing multiple repo calls, shaping DTOs.
   * Gets a repo instance from the service layer (`database_service`) via a context manager.
   * Returns plain dicts / primitives to the app layer.

3. **Service layer / Unit of Work (`database_service.py`)**

   * Creates/owns the SQLAlchemy **Session** and transaction boundary.
   * Hands out repo instances that are **bound to that session**.
   * Commits/rolls back when the `with ...` block exits.

4. **Repository layer (pure data access)**

   * One class per bounded context:

     * `repo_users.py` (User + AuthIdentity)
     * `repo_telegram.py` (Alerts, Schedules, Feedback, Cmd Audit, Settings, VerificationCodes)
     * `repo_trading.py` (Positions, Trades, etc.)
   * Contains **only** data-access logic (queries, inserts/updates/deletes).
   * Receives an existing Session from the UoW; **does not** open/close sessions itself.

5. **ORM models (`model_users.py`, `model_telegram.py`, `model_trading.py`)**

   * SQLAlchemy Declarative; no behavior beyond relationships/constraints.

6. **DB bootstrap (`database.py`)**

   * Engine, SessionLocal, pragmas (WAL, FKs), shared helpers.

# Where your two files fit

* **Your existing `repo_telegram.py`** ← keep this as **the** Telegram repository.
* **The “`telegram_repository.py`” I posted** was a ready-made repo class that:

  * resolves `telegram_user_id → users.id` via `auth_identities`,
  * writes verification codes both to the codes table and the legacy `users.telegram_verification_*` columns,
  * exposes exactly the methods your `telegram_service.py` is calling.

If you already have a `repo_telegram.py` that your service uses, you have two choices:

### Option A (recommended): consolidate into one repo

* **Keep** `src/data/db/repo/repo_telegram.py` as the **only** Telegram repo.
* **Merge** the “extras” you need from my sample (notably:

  * external-ID resolution via `auth_identities`,
  * `set_verification_code()` writing both places,
  * `count_codes_last_hour()` reading `telegram_verification_codes`).
* Delete/retire the old `telegram_repository.py` to avoid confusion.
* Ensure `database_service.get_telegram_repo()` returns **this** class.

### Option B: split by responsibility (advanced)

* Keep a **pure** `repo_telegram.py` (bare CRUD on telegram tables, assumes `user_id` is `users.id`).
* Add a thin **adapter** (call it `telegram_repo_adapter.py`) that:

  * accepts `telegram_user_id` strings,
  * looks up or creates `users` via `repo_users.py`,
  * calls through to `repo_telegram.py`.
* `database_service.get_telegram_repo()` would yield the **adapter**, which composes both repos.

Option A is simpler and matches your current `telegram_service.py` signatures.

# Concrete project map (absolute imports)

```
src/
  data/
    db/
      database.py
      models/
        __init__.py
        model_users.py
        model_telegram.py
        model_trading.py
      repo/
        __init__.py
        repo_users.py
        repo_telegram.py     # <- keep ONE Telegram repo here
        repo_trading.py
    services/
      __init__.py
      database_service.py   # Unit of Work, exposes get_telegram_repo()
      telegram_service.py   # Your app-facing functions (already written)
```

* In `database_service.py`:

  ```python
  from src.data.db.repos.repo_telegram import TelegramRepository
  ```
* In `telegram_service.py` you already do:

  ```python
  from src.data.db.service.database_service import get_database_service
  ```

# Who calls whom (at runtime)

* **Bot/strategy** → `telegram_service.add_alert(user_id, ...)`
* `telegram_service` → `with get_database_service().get_telegram_repo() as repo: repo.add_alert(...)`
* `database_service` → opens Session, yields **repo bound to that Session**, commits on exit
* `repo_telegram` → executes SQLAlchemy queries/inserts on that Session
* `models_*`/`database.py` → used by repo under the hood

# What logic sits where?

* **ID resolution (`telegram_user_id` → `users.id`)**: in **repo\_telegram** (so the service can keep passing strings as today).
* **Business rules / orchestration**: in **telegram\_service** (e.g., verify then set `verified=True`).
* **Cross-aggregate transactions** (e.g., create alert + audit log): the **service** composes multiple repo calls **inside the same UoW** (same `with` block).
* **Validation/DTO shaping**: **service** (your `_alert_to_dict` / `_schedule_to_dict` are perfect).

# Minimal checklist to finish aligning

1. **Pick Option A** (one Telegram repo).

   * Move/merge any methods you need from my example into `repo_telegram.py`:

     * `_resolve_user(...)` using `auth_identities`
     * `set_verification_code(...)` writing both tables/columns
     * `count_codes_last_hour(...)` using `telegram_verification_codes`
2. Ensure `database_service.get_telegram_repo()` yields `repo_telegram.TelegramRepository`.
3. Delete/rename any older `telegram_repository.py` to avoid import collisions.
4. Keep your absolute imports and `pytest.ini` with `pythonpath = src`.



You can use either:
* Single-repo getters:
  ```python
  with get_database_service().get_positions_repo() as repo:
      repo.ensure_open(...)
  ```

* Or the UoW for multi-repo transactions:
  ```python
  with get_database_service().uow() as r:
      pos = r.positions.ensure_open(...)
      r.trades.add({...})
      r.webui_audit.log(user_id=..., action="opened_position", resource_id=pos.id)
  # auto-commit here
  ```

---

### Why this matters

* **One transaction** per service use-case (atomic writes across repos).
* **No hidden sessions**: easier debugging & testing (you can inject a test session).
* **Clean layering**: service code orchestrates; repos only do data access.

---

# Schema and migrations

* **The ORM models describe the prod schema exactly** (verified 2026-11-05 against a
  read-only report of the Pi's DB): index names, `DESC` and partial indexes, server
  defaults, comments, CHECK / UNIQUE / FK names (`conv()` keeps legacy names verbatim
  instead of applying the naming convention in `core/base.py`). Only primary-key
  names are left to the convention; nothing references them.
* **`alembic check` must report "No new upgrade operations detected"** against a DB at
  head. Run it before writing a migration and after changing a model; anything it
  lists is either a model bug or the migration you still have to write.
* Not modeled on purpose (excluded in `migrations/env.py`, `UNMODELED_TABLES`):
  `apscheduler_jobs` (APScheduler's job store). The four `v_*` views are not modeled
  either. The unused legacy P04 table `ss_sentiment_history` was dropped in 016.
* The chain cannot be replayed on an empty DB (the first revision is a diff against
  pre-Alembic tables); tests build the schema with `create_all` + `alembic stamp head`.
* Legacy column quirks the models keep: `job_schedules.state_json` is TEXT
  (`JsonText` stores/reads JSON), several `updated_at` columns are timestamp
  *without* time zone, legacy ids are `GENERATED ALWAYS` identities (never insert an
  explicit id), `job_schedule_runs.job_id` is an integer FK to `job_schedules.id`
  (NULL for ad-hoc API runs, whose label is in `job_snapshot["run_label"]`).
