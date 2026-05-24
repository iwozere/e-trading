# Architecture Review: `src/data/db` & `src/data/downloader`

**Date:** 2026-05-24  
**Reviewer:** AI Architecture Review (Claude Sonnet 4.6)  
**Scope:** `src/data/db/` (models, repos, services, core) · `src/data/downloader/`  
**Status:** ✅ All issues resolved — 2 commits on `main`

---

## Commits

| SHA | Message |
|-----|---------|
| `0be4ebd` | Fix all 5 critical bugs identified in architecture review |
| `5316940` | Fix convention and design issues identified in architecture review |

---

## Critical Bugs

### Bug 1 — `BotInstance.__repr__` references nonexistent attribute

**File:** `src/data/db/models/model_trading.py`  
**Severity:** 🔴 Critical  
**Impact:** `AttributeError` raised on every `repr()` call — silently breaks error logs, test output, and any code that prints a `BotInstance`.

**Root cause:**
```python
# Before — self.type does not exist on the model
return f"<BotInstance(id={self.id}, type='{self.type}', status='{self.status}')>"
```

**Fix:**
```python
return f"<BotInstance(id={self.id}, status='{self.status}')>"
```

---

### Bug 2+3 — `ScheduleRun.job_id` typed as `BigInteger`/`int`, but always used as `str`

**Files:** `src/data/db/models/model_jobs.py`, `src/data/db/schemas/schema_jobs.py`  
**Severity:** 🔴 Critical  
**Impact:** Pydantic validation error at runtime for every `trigger_schedule`, `create_screener_run`, and `create_report_run` call. The jobs service always creates string IDs such as `"manual_1_1716000000"` and `"screener_sp500_1716000000"`, which fail integer coercion.

**Root cause:**
```python
# ORM — wrong column type
job_id: Mapped[int | None] = mapped_column(BigInteger)

# Pydantic — wrong field types
class ScheduleRunCreate(BaseModel):
    job_id: Optional[int] = None

class ScheduleRunResponse(BaseModel):
    job_id: Optional[int]
```

**Fix:** Changed DB column to `String(255)` and both Pydantic schemas to `Optional[str]` throughout. Removed now-unused `BigInteger` import; replaced remaining `BigInteger` on `user_id` with `Integer`.

---

### Bug 4 — Repos call `session.rollback()` inside their own exception handlers

**Files:** `src/data/db/repos/repo_jobs.py` (6 sites), `src/data/db/repos/repo_notification.py` (9 sites), `src/data/db/repos/repo_system_health.py` (2 sites)  
**Severity:** 🔴 Critical  
**Impact:** The Unit-of-Work context manager in `DatabaseService.uow()` is solely responsible for rollback. When a repo rolls back first, the SQLAlchemy session is desynced — subsequent writes in the same UoW may silently succeed in Python but never reach the DB (the session is in a broken state), or the UoW's own rollback-on-exception path triggers a second rollback on an already-closed transaction, causing spurious warnings and potential silent data loss.

**Root cause:**
```python
# Pattern repeated 17 times across 3 repos
except IntegrityError:
    self.session.rollback()   # ← repos must NEVER own rollback
    raise
```

**Fix:** Removed all 17 rogue `self.session.rollback()` calls. Repos now simply re-raise; `DatabaseService.uow()` handles all rollback. The single intentional `NotificationRepository.rollback()` public façade method (used for explicit transaction control by callers) was preserved.

---

### Bug 5 — `BaseDataDownloader.download_multiple_symbols()` does not exist

**File:** `src/data/downloader/base_data_downloader.py`  
**Severity:** 🔴 Critical  
**Impact:** Four subclasses (`polygon_downloader.py`, `coingecko_downloader.py`, `finnhub_downloader.py`, `twelvedata_downloader.py`) call `super().download_multiple_symbols(...)` — which raises `AttributeError` because the method was never defined in the base class. Any batch-download call through these four providers fails unconditionally.

**Root cause:** Method called by subclasses but never implemented in `BaseDataDownloader`.

**Fix:** Added `download_multiple_symbols()` to `BaseDataDownloader`. The base implementation iterates symbols via the caller-supplied `download_func`, logs per-symbol failures with `_logger.exception()` without aborting the batch, and returns a `Dict[str, pd.DataFrame]`. Also removed the unused `import os` and added `Callable, Dict` to the typing imports.

---

## Convention Issues

### Issue 6 — F-string in logger call

**File:** `src/data/db/services/base_service.py`  
**Rule:** CLAUDE.md §3.2 — always use lazy `%s` formatting.  

```python
# Before
self._logger.exception(f"Database error in {func.__name__}")

# After
self._logger.exception("Database error in %s", func.__name__)
```

---

### Issue 7 — Naive `datetime.now()` (no timezone)

**Files:** `src/data/db/services/trading_service.py`, `src/data/db/services/short_squeeze_service.py`, `src/data/db/repos/repo_short_squeeze.py`  
**Rule:** CLAUDE.md §2.3 — date operations must be UTC-aware.  
**Count:** 10+ occurrences across 3 files.

```python
# Before
datetime.now()

# After
datetime.now(timezone.utc)
```

Added `timezone` to the `from datetime import ...` line in each file.

---

### Issue 8 — Deprecated Pydantic v1 `.dict()` 

**File:** `src/data/db/services/jobs_service.py`  
**Impact:** `.dict()` is removed in Pydantic v3 and emits deprecation warnings in Pydantic v2.

```python
# Before
update_data.dict(exclude_unset=True)

# After
update_data.model_dump(exclude_unset=True)
```

Both call sites updated.

---

### Issue 9 — `print()` used instead of logger

**File:** `src/data/downloader/data_downloader_factory.py` — `list_providers()` method  
**Rule:** CLAUDE.md §3 — all modules must use `setup_logger(__name__)`, never `print()`.

```python
# Before
print("Supported Data Providers:")
print("=" * 40)
for spec in _REGISTRY:
    codes = ", ".join([spec.canonical] + list(spec.aliases))
    print(f"  {spec.display_name} ({spec.canonical}) - Codes: {codes}")

# After
_logger.info("Supported Data Providers:")
_logger.info("=" * 40)
for spec in _REGISTRY:
    codes = ", ".join([spec.canonical] + list(spec.aliases))
    _logger.info("%s (%s) - Codes: %s", spec.display_name, spec.canonical, codes)
```

---

## Design Issues

### Issue 11 — Redundant `create_all()` loop over 8 Base classes

**File:** `src/data/db/services/database_service.py`  
**Problem:** `init_databases()` called `create_all()` separately for each of 8 `Base` subclasses. All ORM models share a single `MetaData` instance via `DeclarativeBase`, so 7 of the 8 calls were no-ops and the loop hid that fact.

```python
# Before — 8 separate create_all calls
for base in [Base, UsersBase, NotificationBase, TradingBase, ...]:
    base.metadata.create_all(bind=eng)

# After — one call suffices
_SharedBase.metadata.create_all(bind=eng)
```

---

### Issue 12 — Pydantic schemas co-located with SQLAlchemy ORM models

**Files:** `src/data/db/models/model_jobs.py` → new `src/data/db/schemas/schema_jobs.py`  
**Problem:** `model_jobs.py` contained 9 Pydantic schemas (≈130 lines) alongside the ORM models. Violates SRP: ORM models describe the DB layer; Pydantic schemas describe the API/service contract.

**Fix:** Extracted all 9 schemas into `src/data/db/schemas/schema_jobs.py`:
- `ScheduleCreate`, `ScheduleUpdate`, `ScheduleResponse`
- `ScheduleRunCreate`, `ScheduleRunUpdate`, `ScheduleRunResponse`
- `ReportRequest`, `ScreenerRequest`, `ScreenerSetInfo`

`model_jobs.py` re-exports them all for backward compatibility — existing `from src.data.db.models.model_jobs import ScheduleCreate` imports continue to work unchanged. New `src/data/db/schemas/__init__.py` created (empty per CLAUDE.md §2.2).

---

### Issue 13 — `repo_jobs.py` uses legacy SQLAlchemy 1.x `session.query()` API

**File:** `src/data/db/repos/repo_jobs.py`  
**Problem:** All 11 query sites used the deprecated `session.query()` API inconsistent with every other repo in the codebase (which all use SQLAlchemy 2.x `select()/execute()/scalars()`).

**Affected methods and conversion pattern:**

| Method | Old API | New API |
|--------|---------|---------|
| `get_schedule` | `.query().filter().first()` | `execute(select().where()).scalar_one_or_none()` |
| `get_schedule_by_name` | `.query().filter(and_()).first()` | `execute(select().where(and_())).scalar_one_or_none()` |
| `list_schedules` | `.query().filter()...all()` | `execute(select().where()...).scalars()` |
| `get_pending_schedules` | `.query().filter(and_(or_())).all()` | `execute(select().where(and_(or_()))).scalars()` |
| `get_run` | `.query().filter().first()` | `execute(select().where()).scalar_one_or_none()` |
| `list_runs` | `.query().filter()...all()` | `execute(select().where()...).scalars()` |
| `claim_run` | `.query().filter().with_for_update().first()` | `execute(select().where().with_for_update()).scalar_one_or_none()` |
| `get_pending_runs` | `.query().filter()...all()` | `execute(select().where()...).scalars()` |
| `get_runs_by_job` | `.query().filter(and_()).all()` | `execute(select().where(and_())).scalars()` |
| `get_run_statistics` | `.query().filter().count()` (loop) | `execute(select(func.count()).where()).scalar()` |
| `cleanup_old_runs` | `.query().filter().delete()` | `execute(delete().where())` + `.rowcount` |

Also: consolidated `timedelta` import to top-level (was `from datetime import timedelta` inside two methods), simplified `get_run_statistics` avg-time computation to a list comprehension, and added `delete`, `func` to the SQLAlchemy imports.

---

### Issue 14 — Asyncio event-loop mutation during module import

**File:** `src/data/downloader/data_downloader_factory.py` — `_build_registry()`  
**Problem:** The IBKR import block called `asyncio.set_event_loop(asyncio.new_event_loop())` at module import time. Mutating the running event loop during import is unsafe — it can break any async application that imports the factory after its own loop is running (e.g. a FastAPI app), and it suppresses the `RuntimeError` that would otherwise surface a real misconfiguration.

```python
# Before
try:
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    from src.data.downloader.ibkr_downloader import IBKRDownloader as _IBKR
except Exception:
    _IBKR = None

# After — let the IBKR module handle async initialisation
try:
    from src.data.downloader.ibkr_downloader import IBKRDownloader as _IBKR
except Exception:
    _IBKR = None
```

---

### Issue 15 — `tz_localize(None)` on tz-aware Timestamps

**File:** `src/data/downloader/alpaca_data_downloader.py`  
**Problem:** `tz_localize(None)` raises `TypeError: Already tz-aware, use tz_convert to convert` when called on a Series that already has timezone information. The code guarded with `if df['timestamp'].dt.tz is not None:` — meaning this branch was reached precisely when the Series was tz-aware, guaranteeing the error.

```python
# Before — TypeError when tz is set
if df['timestamp'].dt.tz is not None:
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

# After — tz_convert(None) is the correct idiom for stripping tz
if df['timestamp'].dt.tz is not None:
    df['timestamp'] = df['timestamp'].dt.tz_convert(None)
```

Both occurrences updated.

---

### Issue 16 — Eager module-level singleton causes DB connection at import time

**File:** `src/data/db/services/database_service.py`  
**Problem:** `database_service = get_database_service()` at module level meant a database connection was attempted the moment any code did `from src.data.db.services.database_service import database_service`. This broke unit tests (no DB), environments without a running database, and any import in a context where the DB config is not yet loaded.

```python
# Before — eager construction at import time
database_service = get_database_service()

# After — lazy construction on first access via module __getattr__
def __getattr__(name: str) -> DatabaseService:
    if name == "database_service":
        return get_database_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Python 3.7+ module `__getattr__` is triggered by both attribute access (`module.database_service`) and `from module import database_service`, so all existing import styles continue to work unchanged.

---

### Issue 17 — `heartbeat()` parameter type mismatch

**File:** `src/data/db/repos/repo_trading.py`  
**Problem:** `BotsRepo.heartbeat(bot_id: str)` typed `bot_id` as `str`, but `BotInstance.id` is an `Integer` primary key. Passing an integer `bot_id` worked at runtime (SQLAlchemy coerces it) but broke static analysis and created a misleading type contract.

```python
# Before
def heartbeat(self, bot_id: str) -> None:

# After
def heartbeat(self, bot_id: int) -> None:
```

---

## Summary

| Category | # Issues | Status |
|----------|----------|--------|
| Critical bugs | 5 | ✅ All fixed |
| Convention issues | 4 (§6–9) | ✅ All fixed |
| Design issues | 7 (§11–17) | ✅ All fixed |
| **Total** | **16** | ✅ **All resolved** |

### Files changed

| File | Changes |
|------|---------|
| `src/data/db/models/model_jobs.py` | Bug 2+3: `job_id` type fix; Issue 12: extract Pydantic schemas, re-export |
| `src/data/db/models/model_trading.py` | Bug 1: remove `self.type` from `__repr__` |
| `src/data/db/repos/repo_jobs.py` | Bug 4: remove rollbacks; Issue 13: SQLAlchemy 2.x select() migration |
| `src/data/db/repos/repo_notification.py` | Bug 4: remove 9 rollback calls |
| `src/data/db/repos/repo_system_health.py` | Bug 4: remove 2 rollback calls |
| `src/data/db/repos/repo_short_squeeze.py` | Issue 7: UTC-aware datetimes |
| `src/data/db/repos/repo_trading.py` | Issue 17: `heartbeat(bot_id: int)` |
| `src/data/db/schemas/__init__.py` | Issue 12: new (empty) |
| `src/data/db/schemas/schema_jobs.py` | Issue 12: new — extracted Pydantic schemas |
| `src/data/db/services/base_service.py` | Issue 6: lazy logger formatting |
| `src/data/db/services/database_service.py` | Issue 11: single `create_all()`; Issue 16: lazy singleton |
| `src/data/db/services/jobs_service.py` | Issue 8: `.dict()` → `.model_dump()` |
| `src/data/db/services/short_squeeze_service.py` | Issue 7: UTC-aware datetimes |
| `src/data/db/services/trading_service.py` | Issue 7: UTC-aware datetimes |
| `src/data/downloader/alpaca_data_downloader.py` | Issue 15: `tz_localize` → `tz_convert` |
| `src/data/downloader/base_data_downloader.py` | Bug 5: add `download_multiple_symbols()` |
| `src/data/downloader/data_downloader_factory.py` | Issue 9: `print()` → logger; Issue 14: remove asyncio mutation |
