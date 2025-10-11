# Web UI Database Architecture Analysis

## Current State Issues

### 1. Architectural Violations
- `src/web_ui/backend/telegram_routes.py` directly imports `src.data.db.services.telegram_service`
- Web UI has its own models that duplicate functionality from domain models
- Mixed database session usage between web_ui and data layers

### 2. Model Confusion - Three Different Model Layers!
- `src/web_ui/backend/models.py` - Web UI specific models (should be removed)
- `src/data/models/consolidated_models.py` - Unused consolidated attempt (should be removed)
- `src/data/db/models/model_*.py` - **ACTUAL domain models used by services** (correct)

The domain services actually use `src/data/db/models/model_*.py`, making the consolidated models redundant and confusing.

### 3. Database Session Confusion
- `src/web_ui/backend/database.py` creates its own engine and session
- `src/data/db/services/database_service.py` has its own database management
- No clear single source of truth for database operations

## Recommended Architecture

### 1. Domain Service Layer Pattern
The web_ui should only interact with domain services, not directly with repositories or database sessions.

```
src/web_ui/backend/
├── api/                    # FastAPI routes
├── services/              # Application services (orchestration)
└── dto/                   # Data Transfer Objects

src/data/db/
├── services/              # Domain services (business logic)
├── repos/                 # Repository pattern (data access)
├── models/                # Database models
└── core/                  # Database infrastructure
```

### 2. Proper Service Usage
Web UI should create application services that use domain services:

```python
# src/web_ui/backend/services/telegram_app_service.py
from src.data.db.services import telegram_service, users_service

class TelegramAppService:
    def get_user_stats(self) -> UserStatsDTO:
        users = users_service.list_telegram_users_dto()
        # Transform to web UI DTOs
        return UserStatsDTO(...)
```

### 3. Remove Model Duplication
- Remove `src/web_ui/backend/models.py`
- Remove unused `src/data/models/consolidated_models.py`
- Use only domain models from `src/data/db/models/model_*.py`
- Create DTOs for API responses instead of using database models directly

## Implementation Plan

### Phase 1: Remove Direct Database Imports
1. Remove direct import of `telegram_service` in `telegram_routes.py`
2. Create application service layer in `src/web_ui/backend/services/`
3. Update routes to use application services

### Phase 2: Consolidate Models
1. Remove duplicate models from `src/web_ui/backend/models.py`
2. Remove unused `src/data/models/consolidated_models.py`
3. Update imports to use domain models from `src/data/db/models/model_*.py`
4. Create API DTOs for response models

### Phase 3: Unify Database Sessions
1. Remove `src/web_ui/backend/database.py`
2. Use only the database service from `src/data/db/services/database_service.py`
3. Update authentication to use domain services

## Benefits
- Clear separation of concerns
- Single source of truth for data models
- Proper domain boundaries
- Easier testing and maintenance
- Consistent data access patterns