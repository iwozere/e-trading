# Requirements Document

## Introduction

This feature involves migrating the existing SQLite database (`db/trading.db`) to PostgreSQL while consolidating and optimizing the database schema. The migration includes analyzing current table usage, removing unused tables, merging related tables (specifically `telegram_alerts` and `telegram_schedules` into a unified `schedules` table), and ensuring data integrity throughout the migration process.

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want to analyze the current database schema and table usage, so that I can identify which tables are actively used and which can be safely removed or consolidated.

#### Acceptance Criteria

1. WHEN analyzing the database schema THEN the system SHALL identify all 18 existing tables and their relationships
2. WHEN reviewing table usage THEN the system SHALL categorize tables as: actively used, deprecated, or candidates for consolidation
3. WHEN examining telegram tables THEN the system SHALL confirm that `telegram_alerts` and `telegram_schedules` can be merged into the existing `schedules` table
4. WHEN analyzing table dependencies THEN the system SHALL document all foreign key relationships and constraints
5. IF tables are unused in the codebase THEN the system SHALL mark them for removal

### Requirement 2

**User Story:** As a developer, I want to consolidate the telegram scheduling functionality into a unified schedules table, so that the system has a cleaner and more maintainable database schema.

#### Acceptance Criteria

1. WHEN merging telegram tables THEN the system SHALL combine `telegram_alerts` and `telegram_schedules` functionality into the existing `schedules` table
2. WHEN consolidating data THEN the system SHALL preserve all existing alert and schedule configurations
3. WHEN updating the schema THEN the system SHALL ensure the `schedules` table supports both alert and schedule job types
4. WHEN migrating data THEN the system SHALL map telegram alert configurations to schedule job parameters
5. IF data conflicts exist THEN the system SHALL provide clear resolution strategies

### Requirement 3

**User Story:** As a system administrator, I want to remove unused database tables, so that the database schema is clean and only contains necessary data structures.

#### Acceptance Criteria

1. WHEN identifying unused tables THEN the system SHALL verify no active code references exist
2. WHEN removing tables THEN the system SHALL create backup procedures for data recovery
3. WHEN cleaning schema THEN the system SHALL remove tables: `telegram_alerts`, `telegram_schedules` (after consolidation)
4. WHEN updating codebase THEN the system SHALL remove all references to deleted table models
5. IF tables contain important data THEN the system SHALL migrate data before removal

### Requirement 4

**User Story:** As a database administrator, I want to migrate from SQLite to PostgreSQL, so that the system can benefit from better performance, concurrent access, and advanced database features.

#### Acceptance Criteria

1. WHEN setting up PostgreSQL THEN the system SHALL create a new database with proper configuration
2. WHEN migrating schema THEN the system SHALL convert SQLite schema to PostgreSQL-compatible DDL
3. WHEN transferring data THEN the system SHALL preserve all data integrity and relationships
4. WHEN handling data types THEN the system SHALL properly convert SQLite types to PostgreSQL equivalents
5. WHEN migrating constraints THEN the system SHALL ensure all foreign keys and indexes are properly created

### Requirement 5

**User Story:** As a developer, I want to update the application configuration and connection strings, so that the system seamlessly connects to the new PostgreSQL database.

#### Acceptance Criteria

1. WHEN updating configuration THEN the system SHALL modify database connection settings for PostgreSQL
2. WHEN updating dependencies THEN the system SHALL ensure PostgreSQL drivers are properly configured
3. WHEN testing connections THEN the system SHALL verify successful database connectivity
4. WHEN updating Alembic THEN the system SHALL configure migrations for PostgreSQL
5. IF connection issues occur THEN the system SHALL provide clear error messages and troubleshooting guidance

### Requirement 6

**User Story:** As a quality assurance engineer, I want to validate the migration process, so that I can ensure data integrity and system functionality after the migration.

#### Acceptance Criteria

1. WHEN validating data THEN the system SHALL verify row counts match between source and target databases
2. WHEN testing functionality THEN the system SHALL confirm all CRUD operations work correctly
3. WHEN checking relationships THEN the system SHALL verify all foreign key constraints are properly enforced
4. WHEN running tests THEN the system SHALL execute the full test suite against the new database
5. WHEN comparing performance THEN the system SHALL measure query performance improvements

### Requirement 7

**User Story:** As a system administrator, I want comprehensive migration documentation and rollback procedures, so that I can safely execute the migration and recover if issues occur.

#### Acceptance Criteria

1. WHEN creating documentation THEN the system SHALL provide step-by-step migration instructions
2. WHEN documenting rollback THEN the system SHALL include procedures to revert to SQLite if needed
3. WHEN providing scripts THEN the system SHALL include automated migration and validation scripts
4. WHEN documenting changes THEN the system SHALL list all schema modifications and their rationale
5. IF migration fails THEN the system SHALL provide clear recovery procedures