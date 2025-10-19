# Database Core Module

## Overview
This module provides the core database infrastructure for the e-trading platform, including SQLAlchemy base classes, engine configuration, and session management.

## Features
- Unified SQLAlchemy Base class with consistent naming conventions
- Database engine configuration for both SQLite and PostgreSQL
- Session management with context managers
- SQLite-specific optimizations (WAL mode, foreign keys, etc.)
- Utility functions for table creation and management

## Quick Start

### Basic Usage
```python
from src.data.db.core.database import engine, SessionLocal, session_scope
from src.data.db.core.base import Base

# Using session context manager (recommended)
with session_scope() as session:
    # Your database operations here
    session.add(some_object)
    # Automatic commit/rollback/close

# Direct session usage
session = SessionLocal()
try:
    # Your operations
    session.commit()
finally:
    session.close()
```

### Creating Tables
```python
from src.data.db.core.database import create_all_tables

# Create all tables for the shared Base
create_all_tables()

# Or create tables for specific bases (legacy support)
create_all_tables(SomeBase, AnotherBase)
```

## Architecture

### Files Structure
- `base.py` - SQLAlchemy declarative base with naming conventions
- `database.py` - Engine, session management, and utilities

### Key Components
- **Base**: Shared declarative base for all models
- **engine**: Pre-configured SQLAlchemy engine
- **SessionLocal**: Session factory
- **session_scope()**: Context manager for safe session handling

## Configuration
Database configuration is handled through environment variables:
- `DB_URL`: Database connection string (defaults to config value)
- `SQL_ECHO`: Enable SQL logging (0/1, defaults to config value)

## Integration
This module integrates with:
- All model modules in `src.data.db.models.*`
- Repository modules in `src.data.db.repos.*`
- Test infrastructure in `src.data.db.tests.*`

## Recent Changes
- **2025-01-27**: Refactored to eliminate duplication between `base.py` and `database.py`
- Consolidated naming convention definitions
- Added `get_database_url()` function for external modules
- Improved table creation utilities with fallback to shared Base