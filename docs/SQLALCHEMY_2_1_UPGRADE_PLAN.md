# SQLAlchemy 2.0 → 2.1 Upgrade — Implementation Plan

*Prepared 2026-10-13. Target: SQLAlchemy 2.1.4 (latest on PyPI at time of writing).*

> **Status (2026-10-13):** Phases 0–2 done locally. Phase 1 = `96f440c`; Phase 2 = the commit that
> adds this note. Results on 2.1.4: type-check gate 0/0; pytest across `src/data/db/tests`,
> `src/notification/tests` and `src/telegram/tests` failed exactly the same tests as on 2.0.52
> (52 failed / 143 errors, all already failing before the upgrade); `alembic current`
> and the offline upgrade work; all 11 changed queries ran against the dev DB with
> DeprecationWarnings promoted to errors. Phase 3 (Pi rollout) is pending.
>
> The DB-backed repo/service tests can't run on the dev machine: `repos/conftest.py` creates a
> throwaway DB through the server's `postgres` admin database, and the local server doesn't have one.
> Run with `TEST_DB_URL` set to `donotshare.TEST_DB_URL`; the shell's own value points at a missing DB.

## 1. Background

`requirements.txt` declared `SQLAlchemy >=2.0.43` with no upper bound. On 2026-10-13 the CI
type-check gate (`.github/workflows/typecheck.yml`) resolved **2.1.4** in its fresh venv, while the
local `.venv` and prod (`R:\apps\e-trading\.venv`) both run **2.0.52**. The result was
14 mypy and 17 pyright errors in code that hadn't changed. Commit `87ba7b1` capped the dependency at `<2.1`
to restore the gate.

This plan describes how to lift that cap deliberately.

## 2. Impact assessment (measured, not estimated)

A throwaway env was built with SQLAlchemy 2.1.4 overlaid on the project `.venv` (all other packages
identical), and both checkers were run with the project's pinned configuration:

| Checker | Result on 2.1.4 | Matches CI failure |
|---|---|---|
| mypy 2.3.1 | 14 errors in 3 files | yes, identical |
| pyright 1.1.411 | 17 errors in 4 files | yes, identical |

The runtime import of `database.py`, `repo_notification.py` and `database_optimization.py` succeeds on 2.1.4.

### 2.1 Typing break A — `desc()` / `asc()` function form (14 sites, both checkers)

2.1 added an overload to the module-level `desc()` / `asc()`:

```python
def desc(column: str | ColumnElement[_T]) -> UnaryExpression[_T]: ...
def desc(column: _ColumnExpressionOrStrLabelArgument[_T]) -> OrderByList | UnaryExpression[_T]: ...
```

An ORM attribute (`Message.created_at`, an `InstrumentedAttribute`) is not a `ColumnElement`, so it
hits the second overload. `Query.order_by()` does not accept `OrderByList`, so every call has the form
`query.order_by(desc(Model.col))` fails.

**Fix:** use the method form `Model.col.desc()` / `Model.col.asc()`. This returns
`UnaryExpression[_T]` on both 2.0 and 2.1, and the SQL it generates is identical. A probe module type-checked clean under
mypy `--strict` and pyright on **both** 2.0.52 and 2.1.4.

Affected sites (find with `grep -n "desc(\|asc(" <file>`; line numbers shift with edits):

| File | Methods |
|---|---|
| `src/data/db/repos/repo_notification.py` | `MessageRepository.list_messages` (dynamic `order_field`, ×2), `MessageRepository.get_pending_messages`, `MessageRepository.get_failed_messages_for_retry`, `DeliveryStatusRepository.get_delivery_statuses_by_message`, `ChannelConfigRepository.list_channel_configs` |
| `src/notification/service/database_optimization.py` | `OptimizedMessageRepository.list_messages` (dynamic `order_field`, ×2), `OptimizedMessageRepository.get_pending_messages_optimized` (×2), `OptimizedMessageRepository.get_failed_messages_for_retry`, `OptimizedDeliveryStatusRepository.get_delivery_history_optimized` (×2) |
| `src/data/db/services/notification_service.py` | `NotificationService.search_messages` |

For the dynamic `getattr(Message, order_by, Message.created_at)` sites, annotate
`order_field: Any` (or keep the inferred `Any`) and call `order_field.desc()` / `order_field.asc()`.
Afterwards, drop `asc`/`desc` from the `from sqlalchemy import ...` lines where they become unused.

### 2.2 Typing break B — `Row` element typing (3 sites, pyright only)

`src/data/db/repos/repo_telegram.py`, `BroadcastRepo.stats`. In 2.1, `Row` is
generic over its element types (`Row[int | None, int | None]` for two `func.sum(...)` columns). Pyright
no longer narrows `result[0] if result and result[0] else 0` to `int`, so `total_recipients - ...`,
`... / total_recipients` and `total_recipients > 0` fail.

**Fix (version-neutral):**

```python
total_recipients: int = (result[0] or 0) if result else 0
successful_deliveries: int = (result[1] or 0) if result else 0
```

The probe checked this pattern clean on both versions.

### 2.3 Runtime / behaviour changes from the 2.1 migration guide, checked against this codebase

| 2.1 change | Exposure here | Action |
|---|---|---|
| **Python ≥ 3.11 required** | dev 3.13, CI 3.13, prod 3.14 | none |
| **Default PostgreSQL driver is now psycopg 3** (`postgresql://` → `psycopg`, which isn't installed). Verified: `create_engine("postgresql://…")` raises `ModuleNotFoundError: psycopg` on 2.1.4. | Main `DB_URL` and `TEST_DB_URL` already use `postgresql+psycopg2://`, so they're safe. **Exposed:** (a) `src/data/db/migrations/env.py` when `ALEMBIC_DB_URL`/`DATABASE_URL` is a bare `postgresql://` or `postgres://` URL; (b) `src/notification/service/config.py` `DatabaseConfig.url` default `postgresql://localhost/trading`. | Phase 1: normalise to `postgresql+psycopg2://` in both (see §3). |
| **greenlet no longer installed by default** (only needed for `sqlalchemy.ext.asyncio`) | No `create_async_engine`/`AsyncSession` anywhere in `src/`. `scheduler_service.py` uses `asyncpg` directly, not through SQLAlchemy. | none. `greenlet` stays pinned in `requirements-full.txt`, which is harmless. |
| **Autoflush now fires for Core/`text()` executions too** | Production `sessionmaker` in `database.py` sets `autoflush=False`, so there's no effect. The test fixture in `src/data/db/tests/repos/conftest.py` sets `autoflush=True`, so tests mixing pending ORM objects with `text()` may change. | Phase 2: run the DB test suite. |
| `Select.filter_by()` searches all FROM entities | Only `Query.filter_by()` is used (`register_jobs.py`, `dependency_status.py`), and it's unchanged. | none |
| PG named types (ENUM) bound to `MetaData`; `Table.drop()` no longer drops them | No `Enum`/`ENUM` columns in models or migrations. | none |
| `JSON.contains()` deprecated | `Message.channels.contains(...)` is on an `ARRAY(String)` column, so it's unaffected. | none |
| Cursor errors in `before/after_cursor_execute` hooks now invalidate the connection | Only in `src/notification/docs/utilities/query_analyzer.py`, a dev utility. | none |
| `Computed.persisted` default, HSTORE subscripting, BIT type, collation quoting, URL db-name escaping | Not used; the DB name `trading` has no special characters. | none |
| Dataclass/composite/loader-option semantics | No `MappedAsDataclass`, `composite()` or `__declare_first/last__` usage. | none |
| `sessionmaker(..., autocommit=False, future=True)` legacy kwargs | Verified that they're still accepted on 2.1.4 with `-W error::DeprecationWarning`. | none (optional cleanup) |
| Alembic | 1.19.1 requires `SQLAlchemy>=1.4.23`, which is compatible. | none |

**Net:** two typing fixes (17 sites) plus two URL normalisations. Neither the schema nor the queries change.

## 3. Phases

### Phase 0 — Close the drift window (now, before any upgrade work)

1. **Prod still has the floating pin.** `R:\apps\e-trading\requirements.txt` reads
   `SQLAlchemy >=2.0.43`. Any `pip install -r requirements.txt` on the Pi before it pulls `87ba7b1`
   would install 2.1.4 untested. Pull `main` on the Pi at the next deploy, or now if a pip install is planned.
2. Cap `requirements-webui.txt` as well (`sqlalchemy>=2.0.23` → `sqlalchemy>=2.0.43,<2.1`), so no
   install path floats onto 2.1.

### Phase 1 — Version-neutral code fixes (land on 2.0.52)

Every change below is valid on both 2.0 and 2.1, so it ships and runs in prod **before** the bump.
That keeps the bump commit down to just the requirement-file changes.

> Prerequisite: commit or stash the in-flight edits to `repo_notification.py` /
> `database_optimization.py` (Optimized* subclassing work) first, because the same files are touched here.

1. Convert the 14 `desc(X)` / `asc(X)` sites to `X.desc()` / `X.asc()` (§2.1); remove the unused imports.
2. Rewrite the `repo_telegram.py` statistics narrowing (§2.2).
3. URL normalisation:
   - `src/data/db/migrations/env.py` `_get_db_url()`: after the `postgres://` → `postgresql://`
     rewrite, map a driverless `postgresql://` to `postgresql+psycopg2://` for the env-var and ini paths.
   - `src/notification/service/config.py`: default to `postgresql+psycopg2://localhost/trading`.
   - Leave `scheduler_service.py`'s `+psycopg2` stripping alone. That URL goes to `asyncpg.connect()`,
     not to SQLAlchemy.
4. Unit tests (per the project conventions):
   - an order-by test asserting `str(stmt)` contains `ORDER BY messages.created_at DESC` for the
     dynamic `order_by`/`order_desc` path in `MessageRepository`;
   - a `_get_db_url()` test covering bare `postgres://`, bare `postgresql://` and explicit `+psycopg2`.
5. Verify on 2.0.52: `.venv/Scripts/python.exe scripts/typecheck.py` reports 0/0, and the DB test suite passes.
6. Verify on 2.1.4 using the overlay env (§5): mypy 0, pyright 0.
7. Commit: `Make SQLAlchemy usage forward-compatible with 2.1`.

### Phase 2 — Bump the dependency (dev + CI)

1. `requirements.txt`: `SQLAlchemy >=2.1.4,<2.2`. Replace the cap comment with a short note on what
   2.1 changed.
2. `requirements-full.txt`: `SQLAlchemy==2.1.4`.
3. `requirements-webui.txt`: `sqlalchemy>=2.1.4,<2.2`.
4. Upgrade the local `.venv`: `.venv/Scripts/python.exe -m pip install "SQLAlchemy==2.1.4"`.
5. Verify:
   - `scripts/typecheck.py`: 0 mypy / 0 pyright;
   - `.venv/Scripts/python.exe -m pytest src/data/db/tests src/notification/tests` against the local
     test Postgres. Watch especially for autoflush-related differences under the `autoflush=True` fixture;
   - `alembic current` and `alembic upgrade head --sql` (offline) against the dev DB, which should be a no-op diff;
   - smoke-run one notification job and one pipeline job locally, with `-W default::DeprecationWarning`
     to surface any 2.1 deprecations.
6. Commit `Upgrade SQLAlchemy to 2.1.4`, push, and confirm the CI type-check passes.

### Phase 3 — Production rollout (on the Pi)

1. `git pull`, then `.venv/bin/pip install -r requirements.txt`, then confirm with
   `python -c "import sqlalchemy; print(sqlalchemy.__version__)"` that it reports `2.1.4`.
2. `alembic current`: no pending migrations should appear because of the upgrade.
3. Restart `scheduler.service` (and the notification/telegram services that import the DB layer).
4. Watch the first full scheduling cycle. Use `R:\apps\e-trading\logs` from the dev machine and look for
   `ModuleNotFoundError: psycopg`, `SAWarning`, `DeprecationWarning` and `ArgumentError`.
   Manually trigger a notification delivery and one data-pipeline job.
5. **Rollback:** `pip install "SQLAlchemy==2.0.52"`, revert the Phase 2 commit, then restart services.
   The Phase 1 code stays, because it's 2.0-compatible.

## 4. Acceptance criteria

- mypy and pyright report 0 errors on 2.1.4 in CI and locally.
- DB and notification test suites pass on 2.1.4.
- Prod runs a full day of scheduled jobs on 2.1.4 with no new errors or deprecation warnings in the logs.
- No requirements file allows an untested SQLAlchemy minor version (`<2.2` upper bounds).

## 5. Reproducing the 2.1 type-check locally (without touching `.venv`)

```bash
D=<scratch dir>
"$(.venv/Scripts/python.exe -c 'import sys;print(sys._base_executable)')" -m venv "$D/sa21"
echo 'C:\dev\cursor\e-trading\.venv\Lib\site-packages' > "$D/sa21/Lib/site-packages/zz_parent.pth"
"$D/sa21/Scripts/python.exe" -m pip install --no-deps "sqlalchemy==2.1.4"

# mypy: run from the overlay interpreter
"$D/sa21/Scripts/python.exe" -m mypy src

# pyright: --pythonpath is ignored when pyrightconfig.json sets venv, so extend the config instead
printf '{"extends": "C:/dev/cursor/e-trading/pyrightconfig.json", "venvPath": "%s", "venv": "sa21"}' "$D" > "$D/pyright21.json"
.venv/Scripts/python.exe -m pyright -p "$D/pyright21.json"
```

The `.pth` must be a plain path line. Pyright doesn't execute `import site; site.addsitedir(...)` lines,
and without the plain path it resolves only SQLAlchemy (~990 spurious errors).

## 6. Out of scope

- Migrating to psycopg 3 (2.1's new default driver). That's a separate change with its own risk (driver
  semantics, `sslmode` handling, binary wheels on the Pi). The explicit `+psycopg2` URLs keep it optional.
- SQLAlchemy's `[mypy]` extra, which now wants `mypy>=2.4`. The project pins mypy 2.3.1 and doesn't use the extra.
- Removing the legacy `autocommit=False, future=True` sessionmaker kwargs. They still work on 2.1, so this is optional cleanup.

## 7. Effort

| Phase | Estimate |
|---|---|
| 0 | 10 min |
| 1 | ~2 h, including tests |
| 2 | ~1 h, mostly test runs |
| 3 | 30 min, plus a day of observation |
