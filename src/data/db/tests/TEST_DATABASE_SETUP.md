# Test Database Configuration

## Overview

All unit tests in `src/data/db/tests` use a **separate TEST database**, NOT the production database. This ensures that:
- Tests never modify production data
- Tests can run in isolation
- Tests can be run in parallel
- Test data can be freely created and destroyed

## Test Database Configuration

### Environment Variables

The test suite supports multiple ways to configure the test database:

#### Option 1: TEST_DB_URL (Recommended)
```bash
export TEST_DB_URL="postgresql+psycopg2://user:password@localhost:5432/e_trading_test"
```

#### Option 2: ALEMBIC_DB_URL
```bash
export ALEMBIC_DB_URL="postgresql+psycopg2://user:password@localhost:5432/e_trading_test"
```

#### Option 3: Individual PostgreSQL Variables
```bash
export POSTGRES_TEST_USER="test_user"
export POSTGRES_TEST_PASSWORD="test_password"
export POSTGRES_TEST_HOST="localhost"
export POSTGRES_TEST_PORT="5432"
export POSTGRES_TEST_DB="e_trading_test"
```

### Configuration Files

#### 1. Model Tests (`src/data/db/tests/conftest.py`)
- **Database**: Uses `TEST_DB_URL` environment variable
- **Schema**: Creates temporary schema per test session: `test_models_{uuid}`
- **Tables**: Creates tables from SQLAlchemy models
- **Isolation**: Each test runs in a transaction that rolls back

#### 2. Repository Tests (`src/data/db/tests/repos/conftest.py`)
- **Database**: Creates fresh temporary database per test session: `e_trading_test_{uuid}`
- **Schema**: Uses Alembic migrations to set up schema via `alembic upgrade head`
- **Isolation**: Each test runs in a transaction that rolls back
- **Cleanup**: Drops temporary database after all tests complete

#### 3. Service Tests (`src/data/db/tests/services/conftest.py`)
- **Database**: Inherits from repository tests (uses same temporary database)
- **Additional Fixtures**: Provides `mock_database_service` and `repos_bundle`
- **Isolation**: Same as repository tests

## Safety Measures

### ✅ Production Database Protection

All test configurations explicitly **NEVER** import or use:
- `config.donotshare.donotshare.DB_URL` (production database)
- `DATABASE_URL` environment variable (if pointing to production)

### ✅ Explicit Test Database Requirement

Tests will **skip** (not fail) if:
- `TEST_DB_URL` or `ALEMBIC_DB_URL` is not configured
- PostgreSQL driver (`psycopg2`) is not installed
- Database URL does not start with `postgresql`

### ✅ Temporary Schema/Database Cleanup

- Model tests: Drop temporary schema after test session
- Repository/Service tests: Drop temporary database after test session
- Each test: Rolls back transaction to keep test data isolated

## Running Tests

### Prerequisites

1. **Install psycopg2-binary** (if not already installed):
   ```bash
   pip install psycopg2-binary
   ```

2. **Create test database**:
   ```sql
   CREATE DATABASE e_trading_test;
   ```

3. **Set environment variable**:
   ```bash
   # Windows PowerShell
   $env:TEST_DB_URL = "postgresql+psycopg2://user:password@localhost:5432/e_trading_test"

   # Linux/Mac
   export TEST_DB_URL="postgresql+psycopg2://user:password@localhost:5432/e_trading_test"
   ```

### Run All Tests

```bash
# Activate virtual environment first
source .venv/Scripts/activate  # Linux/Mac
.venv\Scripts\activate         # Windows

# Run all database tests
pytest src/data/db/tests -v

# Run specific test layers
pytest src/data/db/tests/models -v        # Model tests
pytest src/data/db/tests/repos -v         # Repository tests
pytest src/data/db/tests/services -v      # Service tests
```

### Run Tests with Coverage

```bash
pytest src/data/db/tests --cov=src/data/db --cov-report=html
```

## Test Database Schema

### Model Tests
Tables are created directly from SQLAlchemy models using `Base.metadata.create_all()`:
- usr_users, usr_auth_identities, usr_verification_codes
- trading_bots, trading_positions, trading_trades, trading_performance_metrics
- job_schedules, job_schedule_runs
- msg_messages, msg_delivery_status, msg_rate_limits, msg_channel_configs, msg_system_health
- webui_audit_logs, webui_strategy_templates, webui_performance_snapshots, webui_system_config
- telegram_settings, telegram_feedbacks, telegram_command_audits, telegram_broadcast_logs
- ss_ad_hoc_candidates, ss_alerts, ss_deep_metrics, ss_snapshot (short squeeze tables)

### Repository/Service Tests
Schema is created via Alembic migrations:
```bash
alembic upgrade head
```

This ensures the test database schema matches what will be deployed to production.

## Troubleshooting

### Test skipped: "TEST_DB_URL not configured"
**Solution**: Set the `TEST_DB_URL` environment variable pointing to your test database.

### Test skipped: "psycopg2 not installed"
**Solution**: Install psycopg2-binary:
```bash
pip install psycopg2-binary
```

### Error: "database does not exist"
**Solution**: Create the test database first:
```sql
CREATE DATABASE e_trading_test;
```

### Error: "permission denied to create database"
**Solution**: Ensure your PostgreSQL user has `CREATEDB` privilege:
```sql
ALTER USER your_user CREATEDB;
```

### Tests are slow or hanging
**Cause**: Repository tests create/drop databases, which can be slow.
**Solution**: Use model tests for quick iteration, repository tests for integration validation.

## Test Database Naming Convention

- **Model tests**: `test_models_{uuid8}` (temporary schema in TEST database)
- **Repo/Service tests**: `e_trading_test_{uuid8}` (temporary database created and dropped)
- **User-provided**: `e_trading_test` (or any name via TEST_DB_URL)

## Verification

To verify all tests use the test database, run:

```bash
python src/data/db/tests/verify_test_db_config.py
```

This script checks that:
1. No test files import production `DB_URL`
2. All conftest.py files use `TEST_DB_URL` or `ALEMBIC_DB_URL`
3. Test database URLs point to PostgreSQL (not SQLite or production)
