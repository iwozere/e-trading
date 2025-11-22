# Database Imports in EMPS Module

## Summary

The EMPS integration uses `ShortSqueezeService` which follows the **Unit of Work (UoW) pattern** and manages database sessions internally. No need to import `session_scope` directly! ✅

---

## Correct Pattern: Service Layer (UoW)

### ✅ Use Services, Not Direct Session Management

**Correct approach for EMPS:**

```python
from src.data.db.services.short_squeeze_service import ShortSqueezeService

# Service manages sessions internally
service = ShortSqueezeService()
data = service.get_bulk_finra_short_interest(tickers)
```

**Why this is better:**
- ✅ Services handle session management internally via UoW pattern
- ✅ Automatic transaction boundaries with `@with_uow` decorator
- ✅ Consistent error handling
- ✅ Thread-safe
- ✅ No need to manage sessions manually

### ❌ Don't Use session_scope Directly

**Old pattern (don't do this):**
```python
# ❌ WRONG - Don't pass sessions to services
from src.data.db.core.database import session_scope

with session_scope() as session:
    service = ShortSqueezeService(session)  # ❌ Wrong!
    data = service.get_bulk_finra_short_interest(tickers)
```

**Why this is wrong:**
- Services expect `db_service`, not raw sessions
- UoW pattern handles sessions internally
- Breaks the abstraction layer

---

## EMPS Integration Pattern

### How EMPS Uses Database Services

**File:** [emps_p04_integration.py](c:\dev\cursor\e-trading\src\ml\pipeline\p05_emps\emps_p04_integration.py)

```python
# Optional import with graceful fallback
try:
    from src.data.db.services.short_squeeze_service import ShortSqueezeService
    DB_AVAILABLE = True
except ImportError:
    ShortSqueezeService = None
    DB_AVAILABLE = False

# Later in code...
if DB_AVAILABLE:
    service = ShortSqueezeService()  # ✅ No session needed!
    p04_data = service.get_bulk_finra_short_interest(tickers)
```

### How the UoW Pattern Works

**Service class structure:**
```python
class ShortSqueezeService(BaseDBService):
    def __init__(self, db_service=None):
        super().__init__(db_service)
        # db_service manages sessions, NOT raw SQLAlchemy sessions

    @with_uow  # Decorator handles session lifecycle
    @handle_db_error
    def get_bulk_finra_short_interest(self, tickers):
        # self.uow provides access to repositories
        # Session is managed automatically
        ...
```

**What `@with_uow` does:**
1. Opens database session
2. Creates repository bundle
3. Executes method
4. Commits on success
5. Rolls back on error
6. Closes session
7. All automatic! 🎉

---

## Why Import Might Fail

The import itself is correct, but it can fail at **module load time** if:

### 1. **Missing Database Dependencies**
```
ModuleNotFoundError: No module named 'psycopg2'
```

**Cause:** PostgreSQL adapter not installed

**Solution:**
```bash
pip install psycopg2-binary
# or
pip install psycopg2
```

### 2. **Database Connection Issues**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Cause:** Database URL not configured or database not running

**Solution:**
- Check `config/donotshare/donotshare.py` for `DB_URL`
- Ensure database server is running
- Verify connection credentials

---

## EMPS Integration Solution

The EMPS integration file now handles database availability gracefully:

```python
# Optional database imports (only needed for P04 integration)
try:
    from src.data.db.core.database import session_scope
    from src.data.db.services.short_squeeze_service import ShortSqueezeService
    DB_AVAILABLE = True
except ImportError as e:
    logger.warning("Database imports not available: %s", e)
    session_scope = None
    ShortSqueezeService = None
    DB_AVAILABLE = False
```

### Behavior

**With Database Available:**
- ✅ Full P04 integration works
- ✅ Combined EMPS + short interest scoring
- ✅ FINRA data lookup

**Without Database:**
- ✅ EMPS standalone still works
- ⚠️ P04 integration gracefully skipped
- ℹ️ Warning logged but no crash

---

## Testing Database Availability

### Test 1: Check Import
```bash
python -c "from src.data.db.core.database import session_scope; print('✅ OK')"
```

**Expected results:**
- ✅ Success: Database configured properly
- ❌ ImportError: Missing psycopg2 or other dependency
- ❌ OperationalError: Database connection failed

### Test 2: Test EMPS Without Database
```python
from src.ml.pipeline.p05_emps.emps import compute_emps_from_intraday
import pandas as pd

# This will work even without database
df = pd.DataFrame({...})  # Your intraday data
result = compute_emps_from_intraday(df, ticker='TEST')
print(result['emps_score'])  # ✅ Works
```

### Test 3: Test EMPS With P04 Integration
```python
from src.ml.pipeline.p05_emps.emps_p04_integration import create_emps_scanner

scanner = create_emps_scanner(fmp_downloader)

# This will check DB_AVAILABLE internally
results = scanner.scan_with_p04_integration(limit=10)
# ✅ Works with database
# ✅ Also works without (skips P04 part)
```

---

## When You Need Database

### ✅ Database NOT Required For:
- Standalone EMPS scoring
- Universe scanning (EMPS only)
- Using FMP data adapter
- Computing technical indicators

### ⚠️ Database Required For:
- P04 short squeeze integration
- Combined EMPS + short interest scoring
- FINRA data lookup
- Historical candidate tracking

---

## Installation Requirements

### Minimum (EMPS Standalone)
```bash
pip install pandas numpy requests
```

### Full (With P04 Integration)
```bash
pip install pandas numpy requests
pip install sqlalchemy psycopg2-binary
# Configure database in config/donotshare/donotshare.py
```

---

## Troubleshooting

### Issue: "No module named 'psycopg2'"

**Solution 1: Install binary package (recommended)**
```bash
pip install psycopg2-binary
```

**Solution 2: Install from source (requires PostgreSQL dev libs)**
```bash
# On Ubuntu/Debian
sudo apt-get install libpq-dev
pip install psycopg2

# On macOS
brew install postgresql
pip install psycopg2

# On Windows
# Download binary wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/
pip install psycopg2‑X.X.X‑cpXX‑cpXX‑win_amd64.whl
```

### Issue: "Could not connect to database"

**Check 1: Database is running**
```bash
# PostgreSQL
pg_ctl status

# Or check connection
psql -U your_user -d your_database
```

**Check 2: Connection string is correct**
```python
# In config/donotshare/donotshare.py
DB_URL = "postgresql://user:password@localhost:5432/dbname"
```

**Check 3: Use SQLite for testing**
```python
# In config/donotshare/donotshare.py
DB_URL = "sqlite:///./test.db"  # No psycopg2 needed
```

---

## Best Practices

### 1. **Use Try/Except for Optional Features**
```python
try:
    from src.data.db.core.database import session_scope
    # P04 integration code
except ImportError:
    # Graceful fallback
    logger.warning("P04 integration not available")
```

### 2. **Check Availability Before Use**
```python
if DB_AVAILABLE:
    # Use database features
else:
    # Standalone mode
```

### 3. **Provide Clear Error Messages**
```python
if not DB_AVAILABLE and combine_scores:
    raise ValueError(
        "P04 integration requires database. "
        "Install: pip install psycopg2-binary"
    )
```

---

## Summary

| Component | Database Required | Status |
|-----------|-------------------|---------|
| EMPS Core | ❌ No | ✅ Works standalone |
| Data Adapter | ❌ No | ✅ Uses FMP API |
| Universe Scanner | ❌ No | ✅ Uses FMP screener |
| P04 Integration | ✅ Yes | ⚠️ Optional with fallback |
| Combined Scoring | ✅ Yes | ⚠️ Optional with fallback |

**Bottom line:**
- Import path is correct ✅
- EMPS works without database ✅
- P04 integration is optional ✅
- Graceful fallbacks implemented ✅

---

**Last Updated:** 2025-01-21
