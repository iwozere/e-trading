# Database Tests

This directory contains unit tests for the database layer, covering repositories, services, and models.

## Test Database Setup

The tests use PostgreSQL instead of SQLite to match the production environment after the migration.

### Prerequisites

1. **PostgreSQL Server**: Ensure PostgreSQL is running locally or accessible
2. **Test Database**: Create a test database (e.g., `etrading_test`)
3. **Environment Variable**: Set `TEST_DATABASE_URL` or use the default

### Configuration

Set the test database URL using the environment variable:

```bash
export TEST_DATABASE_URL="postgresql://username:password@localhost:5432/etrading_test"
```

Or use the default: `postgresql://postgres:password@localhost:5432/etrading_test`

### Database Schema

The tests automatically:
- Drop all existing tables in the test database
- Create fresh tables from the SQLAlchemy models
- Clean up after test completion

### Running Tests

Run all database tests:
```bash
pytest src/data/db/tests/ -v
```

Run specific test files:
```bash
pytest src/data/db/tests/test_users_repo.py -v
pytest src/data/db/tests/test_jobs_service.py -v
```

### Test Coverage

The tests cover:

#### Users Module
- **Renamed Tables**: `usr_users`, `usr_auth_identities`, `usr_verification_codes`
- **Repository Layer**: `test_users_repo.py`
- **Service Layer**: `test_users_service.py`

#### Jobs Module (New)
- **New Tables**: `job_schedules`, `job_runs`
- **Repository Layer**: `test_jobs_repo.py`
- **Service Layer**: `test_jobs_service.py`

#### Existing Modules
- **Telegram**: Tests updated for renamed foreign key references
- **Trading**: Existing tests maintained
- **WebUI**: Tests updated for renamed foreign key references

### Test Structure

```
src/data/db/tests/
├── conftest.py          # Shared fixtures and PostgreSQL setup
├── factories.py         # Test data factories
├── test_users_repo.py   # Users repository tests
├── test_users_service.py # Users service tests
├── test_jobs_repo.py    # Jobs repository tests
├── test_jobs_service.py # Jobs service tests
├── test_telegram_*.py   # Existing telegram tests
├── test_trading_*.py    # Existing trading tests
└── test_webui_*.py      # Existing webui tests
```

### Migration Validation

These tests validate that:
1. **Table Renaming**: Old `users` → `usr_users` works correctly
2. **Foreign Key Updates**: All FKs reference the new table names
3. **New Tables**: `job_schedules` and `job_runs` function properly
4. **Data Integrity**: Relationships and constraints work as expected
5. **Service Layer**: Business logic handles the schema changes transparently

### Troubleshooting

**Connection Issues**:
- Verify PostgreSQL is running
- Check database credentials
- Ensure the test database exists

**Permission Issues**:
- Ensure the database user has CREATE/DROP privileges
- The test user needs to create and drop tables

**Schema Issues**:
- Tests automatically handle schema creation/cleanup
- If tests fail, check that all model imports are working
- Verify foreign key references use the new table names