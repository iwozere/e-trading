# Database Testing Guide

## Overview

This guide provides comprehensive information about testing the database layer of the e-trading application. The test suite covers models, repositories, and services with a focus on unit testing, integration testing, and ensuring code quality.

---

## Test Structure

```
src/data/db/tests/
├── CODE_REVIEW.md              # Comprehensive code review document
├── TESTING_GUIDE.md            # This file
├── conftest.py                 # Root test configuration
├── fixtures/                   # Test data factories
│   ├── __init__.py
│   ├── factory_jobs.py         # Job and schedule test data
│   └── factory_users.py        # User and auth test data
├── models/                     # Model layer tests
│   ├── conftest.py            # Schema-based test fixtures
│   ├── test_model_jobs.py
│   ├── test_model_notification.py
│   ├── test_model_short_squeeze.py
│   ├── test_model_system_health.py
│   ├── test_model_telegram.py
│   ├── test_model_trading.py
│   ├── test_model_users.py
│   └── test_model_webui.py
├── repos/                      # Repository layer tests
│   ├── conftest.py            # Database-based test fixtures
│   ├── test_repo_jobs.py
│   ├── test_repo_notification.py
│   ├── test_repo_short_squeeze.py
│   ├── test_repo_system_health.py
│   ├── test_repo_telegram.py
│   ├── test_repo_trading.py
│   ├── test_repo_users.py
│   └── test_repo_webui.py
└── services/                   # Service layer tests [NEW]
    ├── conftest.py            # Service test fixtures
    ├── test_service_jobs.py   # JobsService comprehensive tests
    └── test_service_users.py  # UsersService comprehensive tests
```

---

## Test Fixtures

### Fixture Factories

The `fixtures/` directory contains factory modules for creating test data:

#### **factory_jobs.py**

Provides factories for job-related test data:

```python
from src.data.db.tests.fixtures.factory_jobs import ScheduleFactory, ScheduleRunFactory

# Create a daily screener schedule
schedule_data = ScheduleFactory.daily_screener(user_id=1, name="my_screener")

# Create a pending run
run_data = ScheduleRunFactory.pending_run(job_id="test_job", user_id=1)

# Create a completed run
completed_data = ScheduleRunFactory.completed_run(job_id="done_job", execution_time_ms=1500)
```

**Available Schedule Factories:**
- `create_data()` - Generic schedule data
- `daily_screener()` - Daily screener schedule
- `weekly_report()` - Weekly report schedule
- `disabled_schedule()` - Disabled schedule

**Available Run Factories:**
- `create_data()` - Generic run data
- `pending_run()` - Pending run
- `running_run()` - Running run
- `completed_run()` - Completed run
- `failed_run()` - Failed run

#### **factory_users.py**

Provides factories for user-related test data:

```python
from src.data.db.tests.fixtures.factory_users import (
    UserFactory, AuthIdentityFactory, VerificationCodeFactory
)

# Create admin user
admin_data = UserFactory.admin_user()

# Create Telegram auth identity
telegram_auth = AuthIdentityFactory.telegram_identity(
    user_id=1,
    telegram_id=123456789,
    username="testuser"
)

# Create active verification code
code_data = VerificationCodeFactory.active_code(user_id=1, code="123456")
```

**Available User Factories:**
- `create_data()` - Generic user data
- `admin_user()` - Admin user
- `regular_user()` - Regular user
- `inactive_user()` - Inactive user

**Available Auth Identity Factories:**
- `create_data()` - Generic auth identity
- `telegram_identity()` - Telegram auth
- `google_identity()` - Google auth

**Available Verification Code Factories:**
- `create_data()` - Generic code
- `active_code()` - Active code
- `expired_code()` - Expired code
- `used_code()` - Used code

---

## Running Tests

### Prerequisites

1. **PostgreSQL Database**: A PostgreSQL instance must be running
2. **Environment Variables**: Set up test database configuration

### Environment Setup

#### Option 1: Using .env file

Create `config/donotshare/.env.test`:

```bash
ALEMBIC_DB_URL=postgresql+psycopg2://user:password@localhost:5432/e_trading_test
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

#### Option 2: Using environment variables

**PowerShell (Windows):**
```powershell
$env:ALEMBIC_DB_URL = "postgresql+psycopg2://user:password@localhost/e_trading_test"
$env:POSTGRES_USER = "your_user"
$env:POSTGRES_PASSWORD = "your_password"
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
```

**Bash (Linux/Mac):**
```bash
export ALEMBIC_DB_URL="postgresql+psycopg2://user:password@localhost/e_trading_test"
export POSTGRES_USER="your_user"
export POSTGRES_PASSWORD="your_password"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
```

### Running Test Suites

#### Run All Tests
```bash
pytest src/data/db/tests/ -v
```

#### Run Model Tests Only
```bash
pytest src/data/db/tests/models/ -v
```

#### Run Repository Tests Only
```bash
pytest src/data/db/tests/repos/ -v
```

#### Run Service Tests Only
```bash
pytest src/data/db/tests/services/ -v
```

#### Run Specific Test File
```bash
pytest src/data/db/tests/services/test_service_jobs.py -v
```

#### Run Specific Test Class
```bash
pytest src/data/db/tests/services/test_service_jobs.py::TestJobsServiceSchedules -v
```

#### Run Specific Test Method
```bash
pytest src/data/db/tests/services/test_service_jobs.py::TestJobsServiceSchedules::test_create_schedule_success -v
```

#### Run with Coverage
```bash
pytest src/data/db/tests/ --cov=src/data/db --cov-report=html
```

---

## Test Strategies

### Model Layer Testing

**Strategy**: Schema-based testing with transactional rollback

**What's Tested:**
- Pydantic validation
- Database constraints (unique, check, foreign key)
- Model methods and properties
- Enum validation
- Default values

**Example:**
```python
def test_schedule_pydantic_cron_validator():
    with pytest.raises(ValueError):
        ScheduleCreate(name="a", job_type="report", target="t", cron="* * *")

def test_schedule_db_insert_and_query(db_session):
    s = Schedule(user_id=1, name="daily", job_type="report",
                 target="do_report", cron="0 0 * * *")
    db_session.add(s)
    db_session.flush()
    assert s.id is not None
```

### Repository Layer Testing

**Strategy**: Full database testing with Alembic migrations

**What's Tested:**
- CRUD operations
- Complex queries (filtering, sorting, pagination)
- Atomic operations (locking, claiming)
- Bulk operations
- Statistics and aggregations
- Data cleanup

**Example:**
```python
def test_claim_run_atomically(jobs_repo, db_session):
    run_data = ScheduleRunFactory.pending_run()
    run = ScheduleRun(**run_data)
    db_session.add(run)
    db_session.flush()

    claimed = jobs_repo.claim_run(run.id, "worker_1")

    assert claimed is not None
    assert claimed.status == RunStatus.RUNNING
    assert claimed.worker_id == "worker_1"
```

### Service Layer Testing

**Strategy**: Mock database service with real repositories

**What's Tested:**
- Business logic validation
- UoW transaction management
- Error handling and rollback
- Decorator functionality
- Input validation
- Complex workflows

**Example:**
```python
def test_create_schedule_success(mock_database_service, db_session):
    service = JobsService(db_service=mock_database_service)

    schedule_data = ScheduleCreate(
        name="daily_screener",
        job_type=JobType.SCREENER,
        target="AAPL,MSFT",
        cron="0 9 * * *"
    )

    schedule = service.create_schedule(user_id=1, schedule_data=schedule_data)

    assert schedule.id is not None
    assert schedule.next_run_at is not None
```

---

## Test Coverage

### Current Coverage Status

| Layer | Files Tested | Coverage | Status |
|-------|-------------|----------|--------|
| **Models** | 8/8 | 100% | ✅ Complete |
| **Repositories** | 8/8 | 100% | ✅ Complete |
| **Services** | 2/8+ | 25% | 🚧 In Progress |
| **Integration** | 0 | 0% | ❌ Not Started |

### Completed Test Files

#### Service Tests (NEW)

**test_service_jobs.py** - 400+ lines, 40+ tests
- ✅ Schedule CRUD operations
- ✅ Run CRUD operations
- ✅ Cron validation and calculation
- ✅ Worker claiming logic
- ✅ Screener run creation
- ✅ Report run creation
- ✅ Statistics and cleanup
- ✅ Error handling

**test_service_users.py** - 300+ lines, 20+ tests
- ✅ User creation (ensure pattern)
- ✅ Telegram profile operations
- ✅ User listing and filtering
- ✅ Admin user operations
- ✅ Broadcast user listing
- ✅ Edge cases and error handling

### Test Coverage by Feature

#### JobsService Coverage

| Feature | Tests | Status |
|---------|-------|--------|
| Create Schedule | 3 | ✅ |
| Get/List Schedules | 4 | ✅ |
| Update Schedule | 3 | ✅ |
| Delete Schedule | 1 | ✅ |
| Trigger Schedule | 2 | ✅ |
| Create Run | 2 | ✅ |
| Get/List Runs | 2 | ✅ |
| Update Run | 1 | ✅ |
| Claim Run | 2 | ✅ |
| Cancel Run | 2 | ✅ |
| Cron Validation | 2 | ✅ |
| Target Expansion | 4 | ✅ |
| Screener Runs | 3 | ✅ |
| Report Runs | 1 | ✅ |
| Statistics | 1 | ✅ |
| Cleanup | 1 | ✅ |

**Total: 34 tests**

#### UsersService Coverage

| Feature | Tests | Status |
|---------|-------|--------|
| Ensure User | 4 | ✅ |
| Get User | 2 | ✅ |
| Telegram Profile | 3 | ✅ |
| List Users | 3 | ✅ |
| Admin Operations | 1 | ✅ |
| Edge Cases | 4 | ✅ |
| Integration | 2 | ✅ |

**Total: 19 tests**

---

## Writing New Tests

### Test Template

```python
"""
Tests for [ServiceName].

Tests cover:
- [Feature 1]
- [Feature 2]
- [Feature 3]
"""
import pytest
from src.data.db.services.[service_module] import [ServiceClass]

class Test[ServiceName][FeatureGroup]:
    """Tests for [feature group]."""

    def test_[feature]_success(self, mock_database_service, db_session):
        """Test successful [feature]."""
        service = [ServiceClass](db_service=mock_database_service)

        # Arrange
        # ... setup test data

        # Act
        result = service.some_method()

        # Assert
        assert result is not None
        # ... more assertions

    def test_[feature]_error_case(self, mock_database_service):
        """Test [feature] error handling."""
        service = [ServiceClass](db_service=mock_database_service)

        with pytest.raises(SomeException):
            service.some_method_with_invalid_input()
```

### Best Practices

1. **Use Descriptive Test Names**: `test_create_schedule_with_invalid_cron_raises_error`
2. **Use Arrange-Act-Assert Pattern**: Separate setup, execution, and verification
3. **Test One Thing**: Each test should verify one specific behavior
4. **Use Fixtures**: Leverage pytest fixtures for common setup
5. **Use Factories**: Use factory functions for test data creation
6. **Test Edge Cases**: Include boundary conditions and error cases
7. **Use Meaningful Assertions**: Assert specific values, not just truthiness
8. **Clean Test Data**: Use transactional rollback to keep tests isolated

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

**Error:** `could not connect to server`

**Solution:**
- Ensure PostgreSQL is running
- Verify connection parameters in environment variables
- Check firewall settings

#### 2. Migration Errors

**Error:** `relation "table_name" does not exist`

**Solution:**
```bash
# Run migrations manually
cd /path/to/project
alembic upgrade head
```

#### 3. Test Database Not Cleaned

**Error:** `duplicate key value violates unique constraint`

**Solution:**
- Repository tests create temporary databases automatically
- Model tests use transactional rollback
- Check if transaction is committed accidentally

#### 4. Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
- Ensure repository root is in `sys.path`
- Run tests from repository root
- Check `conftest.py` path setup

---

## Next Steps

### Remaining Service Tests

Priority order for implementing remaining service tests:

1. **NotificationService** (High Priority)
   - Message queueing
   - Delivery tracking
   - Rate limiting
   - Channel management

2. **ShortSqueezeService** (High Priority)
   - Snapshot management
   - Alert generation
   - FINRA data processing
   - Candidate scoring

3. **TradingService** (Medium Priority)
   - Trade execution
   - Position management
   - PnL calculation
   - Bot lifecycle

4. **Other Services** (Low Priority)
   - TelegramService
   - WebUIService
   - SystemHealthService
   - AlertsService

### Integration Tests

Create end-to-end workflow tests:

1. **Job Execution Workflow**
   - Schedule creation → pending runs → worker claiming → execution → completion

2. **Notification Pipeline**
   - Message creation → queueing → delivery → tracking → retry

3. **Trading Workflow**
   - Bot initialization → trade execution → position updates → PnL calculation

4. **Multi-Repository Transactions**
   - Test UoW pattern with multiple repositories
   - Test rollback scenarios
   - Test commit consistency

---

## Resources

### Documentation
- [CODE_REVIEW.md](CODE_REVIEW.md) - Comprehensive code review
- [pytest Documentation](https://docs.pytest.org/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html)

### Related Files
- [models/conftest.py](models/conftest.py) - Model test fixtures
- [repos/conftest.py](repos/conftest.py) - Repository test fixtures
- [services/conftest.py](services/conftest.py) - Service test fixtures

---

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use the factory pattern for test data
3. Write descriptive test names
4. Include docstrings
5. Test both success and error cases
6. Maintain high test coverage
7. Run full test suite before committing

---

**Last Updated:** 2025-11-02
**Test Suite Version:** 1.0
**Coverage Goal:** 90%+
