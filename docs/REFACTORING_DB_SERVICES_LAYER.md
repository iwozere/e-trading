# Database Services Layer Refactoring

## Overview
This document summarizes the refactoring completed to ensure all database access goes through the services layer (`src/data/db/services`) without directly accessing models, repos, or sessions via `session_scope`.

## Date: 2025-01-XX

## Objectives
1. **Centralize database access** through the services layer
2. **Eliminate direct `session_scope` usage** in application code
3. **Use modern UoW (Unit of Work) pattern** consistently via `BaseDBService`
4. **Improve testability** and maintainability

---

## Changes Made

### 1. ShortSqueezeService Refactoring ✅

**File:** `src/data/db/services/short_squeeze_service.py`

#### Updated Method Signatures
Removed the `repos` parameter from all methods decorated with `@with_uow`. The UoW is now accessed via `self.uow` property from `BaseDBService`.

**Before:**
```python
@with_uow
def get_candidates_for_deep_scan(self, repos) -> List[str]:
    return repos.short_squeeze.get_active_candidates_for_deep_scan()
```

**After:**
```python
@with_uow
def get_candidates_for_deep_scan(self) -> List[Dict[str, Any]]:
    return self.uow.short_squeeze.get_active_candidates_for_deep_scan()
```

#### Methods Updated:
- `save_screener_results()`
- `save_deep_scan_results()`
- `get_candidates_for_deep_scan()`
- `add_adhoc_candidate()`
- `remove_adhoc_candidate()`
- `expire_adhoc_candidates()`
- `create_alert()`
- `mark_alert_sent()`
- `get_top_candidates_by_screener_score()`
- `get_top_squeeze_scores()`
- `get_ticker_analysis()`
- `cleanup_old_data()`
- `get_pipeline_statistics()`
- `get_active_adhoc_candidates()`

#### New Methods Added:
- `get_latest_finra_short_interest(ticker: str)`
- `get_finra_data_freshness_report()`
- `store_finra_data(finra_data_list: List[Dict])`
- `get_bulk_finra_short_interest(tickers: List[str])`

---

### 2. ML Pipeline Files Refactoring ✅

#### File: `src/ml/pipeline/p04_short_squeeze/core/daily_deep_scan.py`

**Changes:**
1. Removed `from src.data.db.core.database import session_scope`
2. Updated `_load_active_candidates()` - removed `with session_scope()`
3. Updated `_enhance_candidate_with_finra_data()` - removed `with session_scope()`
4. Updated `_store_results()` - removed `with session_scope()`, converted data to dict format
5. Updated `_get_historical_mentions_async()` - added TODO comment for service layer implementation

**Before:**
```python
with session_scope() as session:
    service = ShortSqueezeService(session)
    candidates = service.get_candidates_for_deep_scan()
```

**After:**
```python
service = ShortSqueezeService()
candidates = service.get_candidates_for_deep_scan()
```

---

#### File: `src/ml/pipeline/p04_short_squeeze/core/weekly_screener.py`

**Changes:**
1. Removed `from src.data.db.core.database import session_scope`
2. Updated `_update_finra_data()` - removed `with session_scope()`
3. Updated `_enhance_with_finra_data()` - removed `with session_scope()`
4. Updated `_store_results()` - removed `with session_scope()`, converted candidates to dict format

---

### 3. Scheduler Refactoring ✅

#### File: `src/scheduler/main.py`

**Changes:**
1. Removed `from src.data.db.core.database import session_scope`
2. Updated `initialize_services()` method:

**Before:**
```python
with session_scope() as session:
    self.jobs_service = JobsService(session)
```

**After:**
```python
self.jobs_service = JobsService()  # Uses modern UoW pattern
```

---

### 4. Remaining Files (Not Yet Updated)

The following files still use `session_scope` and should be refactored in a future iteration:

#### Application Code:
- `src/ml/pipeline/p04_short_squeeze/core/universe_loader.py` (1 usage)
- `src/ml/pipeline/p04_short_squeeze/core/alert_engine.py` (3 usages)

#### Scripts (Lower Priority):
- `cache_edgar/code/repair_finra_data.py` (2 usages)
- `cache_edgar/code/populate_finra_calculated_fields.py` (3 usages)
- `src/ml/pipeline/p04_short_squeeze/scripts/run_short_data.py` (1 usage)

#### Test Files (Keep as-is for now):
- `tests/test_short_squeeze_database_integration.py` (multiple usages)
- `src/ml/pipeline/p04_short_squeeze/tests/*.py` (multiple usages)
- `src/notification/tests/test_delivery_history_integration.py` (1 usage)

**Note:** Test files may continue using `session_scope` for direct database testing purposes.

---

## Architecture Pattern

### Modern UoW Pattern (BaseDBService)

```python
from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error

class MyService(BaseDBService):
    def __init__(self, db_service=None):
        super().__init__(db_service)

    @with_uow
    @handle_db_error
    def my_method(self, param1, param2):
        # Access repositories via self.uow
        data = self.uow.my_repo.query_something(param1)
        self.uow.my_repo.update_something(param2)
        # Commit happens automatically on success
        # Rollback happens automatically on exception
        return data
```

### Usage in Application Code

```python
from src.data.db.services.short_squeeze_service import ShortSqueezeService

# Simply instantiate and call methods
service = ShortSqueezeService()
candidates = service.get_candidates_for_deep_scan()
service.save_deep_scan_results(results, scan_date)
```

---

## Benefits

1. **Single Point of Control**: All DB logic centralized in services
2. **Consistent Transactions**: UoW pattern ensures atomic operations
3. **Better Testability**: Mock services instead of database sessions
4. **Cleaner Code**: No `with session_scope()` boilerplate in business logic
5. **Easier Maintenance**: Changes to DB access patterns in one place

---

## Migration Checklist

- [x] Update `ShortSqueezeService` to use `self.uow` pattern
- [x] Refactor `daily_deep_scan.py`
- [x] Refactor `weekly_screener.py`
- [x] Refactor `scheduler/main.py`
- [ ] Refactor `universe_loader.py`
- [ ] Refactor `alert_engine.py`
- [ ] Refactor cache_edgar scripts (optional)
- [ ] Update documentation
- [ ] Run integration tests

---

## Next Steps

1. **Complete Remaining Files**: Update `universe_loader.py` and `alert_engine.py`
2. **Add Missing Repo Methods**: Some service methods assume repo methods that may need to be implemented
3. **Integration Testing**: Run full pipeline to ensure refactoring works end-to-end
4. **Performance Testing**: Verify no performance regression with new pattern
5. **Documentation**: Update API documentation to reflect new service methods

---

## Breaking Changes

### For Internal Code
- Services no longer accept `session` or `repos` parameters
- Service methods return dict/list instead of ORM models in some cases
- Methods renamed for clarity:
  - `get_candidates_for_deep_scan()` returns dicts, not tickers
  - Added `get_candidates_for_deep_scan_tickers()` for just tickers

### Migration Guide for Existing Code

**Old Pattern:**
```python
from src.data.db.core.database import session_scope
from src.data.db.services.short_squeeze_service import ShortSqueezeService

with session_scope() as session:
    service = ShortSqueezeService(session)
    result = service.do_something()
```

**New Pattern:**
```python
from src.data.db.services.short_squeeze_service import ShortSqueezeService

service = ShortSqueezeService()
result = service.do_something()
```

---

## Questions & Answers

**Q: Can I still use `session_scope` in tests?**
A: Yes, integration tests that need to verify database state directly can continue using `session_scope`.

**Q: What if I need to perform multiple operations in one transaction?**
A: Create a service method that groups those operations. The `@with_uow` decorator ensures they're all in one transaction.

**Q: How do I access raw SQL or complex queries?**
A: Create a repository method for complex queries, then call it from a service method via `self.uow.my_repo`.

**Q: What about performance with multiple service calls?**
A: Each service method call is a separate transaction. For bulk operations, create a single service method that does all the work.

---

## File Count Summary

**Updated:** 4 files
**Remaining:** 5 application files + scripts + tests
**Lines Changed:** ~200 lines

## Status: 🟡 Partially Complete

The core refactoring is complete for the main ML pipeline and scheduler. Remaining files are lower priority and can be updated incrementally.
