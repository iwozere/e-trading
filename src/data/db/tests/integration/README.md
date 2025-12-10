# Database Integration Tests

## ⚠️ IMPORTANT SAFETY CHANGE

**What Changed:**
These integration tests have been **moved and refactored** to prevent connecting to the production database.

**Previous Location (DELETED):**
- `tests/test_database_integration.py` ❌
- `tests/test_live_bot_database.py` ❌
- `tests/test_short_squeeze_database_integration.py` ❌

**New Location:**
- `src/data/db/tests/integration/test_database_integration.py` ✅
- `src/data/db/tests/integration/test_live_bot_database.py` ✅
- `src/data/db/tests/integration/test_short_squeeze_integration.py` ✅

## Why This Change Was Made

The old tests were connecting directly to the **production database** instead of using isolated test databases. This caused:
- **Data loss**: All users were deleted from `usr_users` and `usr_auth_identities` tables
- **Test pollution**: Test data mixed with production data
- **Violation of testing principles**: Tests must not modify production data

## How Tests Now Work

### Isolated Test Database
All integration tests now use pytest fixtures that:
1. Create a temporary test database for each test session
2. Run Alembic migrations on the test database
3. Use transactional rollback after each test
4. **Never touch production data**

### Test Fixtures Used
- `db_session`: Provides an isolated database session
- `repos_bundle`: Provides access to all repositories
- `engine`: Test database engine

## Running the Tests

```powershell
# Set test database URL (required)
$env:TEST_DB_URL = "postgresql+psycopg2://postgres:yourpassword@localhost/e_trading_test"

# Or use individual postgres env vars
$env:POSTGRES_TEST_USER = "postgres"
$env:POSTGRES_TEST_PASSWORD = "yourpassword"
$env:POSTGRES_TEST_HOST = "localhost"
$env:POSTGRES_TEST_PORT = "5432"
$env:POSTGRES_TEST_DB = "e_trading_test"

# Run integration tests
pytest src/data/db/tests/integration/ -v

# Run all database tests
pytest src/data/db/tests/ -v
```

## Safety Guardrails Added

### database.py Protection
Added safety check to `drop_all_tables()` function:
- **Refuses** to drop tables if DB_URL doesn't contain 'test'
- Prevents accidental production database destruction
- Raises `RuntimeError` with clear error message

## Test Database Setup

The test database is automatically:
1. Created with a unique name for each test session
2. Initialized with Alembic migrations
3. Dropped after all tests complete

No manual setup required if you set the `TEST_DB_URL` environment variable.

## Migration from Old Tests

If you have custom tests that used the old pattern:

**Before (DON'T DO THIS):**
```python
from src.data.db.services import database_service as ds
from src.data.db.repos.repo_trading import TradeRepository

def test_something():
    db_service = ds.get_database_service()  # ❌ Uses production DB
    repo = TradeRepository()  # ❌ Uses production DB
```

**After (DO THIS):**
```python
import pytest

def test_something(db_session, repos_bundle):
    # ✅ Uses isolated test DB
    repo = repos_bundle.trades
    # ... test code using repo and db_session
```

## Questions?

See:
- `src/data/db/tests/repos/conftest.py` - Repository test fixtures
- `src/data/db/tests/services/conftest.py` - Service test fixtures
- `src/data/db/tests/integration/conftest.py` - Integration test fixtures
