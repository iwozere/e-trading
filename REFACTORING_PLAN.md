# Database Refactoring Plan

## Current Issues
1. Mixed database access patterns (SQLAlchemy + raw sqlite3)
2. Code duplication in telegram/db.py
3. Single database for multiple concerns
4. Inconsistent error handling

## Phase 1: Immediate Cleanup (High Priority) ✅ COMPLETED

### 1.1 Remove Raw SQL from telegram/db.py ✅
- **File**: `src/frontend/telegram/db.py` - REMOVED
- **Action**: Removed all sqlite3 imports and raw SQL code
- **Reason**: Already using TelegramRepository with SQLAlchemy
- **New Location**: `src/data/db/telegram_service.py`

### 1.2 Consolidate Database Managers ✅
- **Current**: Multiple database connection patterns - UNIFIED
- **Target**: Single DatabaseManager instance via DatabaseService
- **Files**: All database logic moved to `src/data/db/`

### 1.3 Standardize Error Handling ✅
- **Pattern**: Consistent repository pattern with context managers
- **Files**: All repository classes use unified error handling

## Phase 2: Architecture Improvements (Medium Priority)

### 2.1 Separate Database Concerns
- **Trading Database**: `db/trading.db` (trades, bots, performance)
- **Telegram Database**: `db/telegram.db` (users, alerts, schedules)
- **Benefits**: Better separation of concerns, easier scaling

### 2.2 Create Unified Database Service
```python
class DatabaseService:
    def __init__(self):
        self.trading_db = DatabaseManager("sqlite:///db/trading.db")
        self.telegram_db = DatabaseManager("sqlite:///db/telegram.db")
    
    def get_trading_repo(self) -> TradeRepository:
        return TradeRepository(self.trading_db.get_session())
    
    def get_telegram_repo(self) -> TelegramRepository:
        return TelegramRepository(self.telegram_db.get_session())
```

### 2.3 Add Connection Pooling
- **Library**: SQLAlchemy connection pooling
- **Benefits**: Better performance, resource management

## Phase 3: Advanced Optimizations (Low Priority)

### 3.1 Add Database Migrations
- **Tool**: Alembic
- **Benefits**: Version control for schema changes

### 3.2 Add Query Optimization
- **Indexes**: Review and optimize existing indexes
- **Caching**: Add query result caching where appropriate

### 3.3 Add Database Health Monitoring
- **Metrics**: Connection count, query performance
- **Alerting**: Database issues detection

## Implementation Status

### ✅ COMPLETED
1. **Database Consolidation**: All database logic moved to `src/data/db/`
   - `src/frontend/telegram/db.py` → REMOVED
   - `src/frontend/telegram/db_clean.py` → `src/data/db/telegram_service.py`
   - All imports updated across the codebase

2. **Clean Architecture**: 
   - Frontend layer (`src/frontend/`) now only contains UI logic
   - Data layer (`src/data/db/`) contains all database operations
   - Service layer provides clean interface between frontend and data

3. **Unified Database Service**: 
   - `DatabaseService` provides single entry point
   - Repository pattern with proper session management
   - Context managers for automatic cleanup

### 🔄 TODO
1. **Ticker Management Functions**: Some functions from old db.py need to be implemented
2. **Database Separation**: Consider if separate DBs are needed (currently using single DB)
3. **Migrations and Monitoring**: Future enhancements

## Risk Assessment

- **Low Risk**: Removing duplicate code in telegram/db.py
- **Medium Risk**: Database separation (requires data migration)
- **High Risk**: Major architectural changes

## Testing Strategy

1. **Unit Tests**: Test each repository independently
2. **Integration Tests**: Test database service interactions
3. **Migration Tests**: Verify data integrity during separation