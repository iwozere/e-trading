# Requirements Document

## Introduction

The database migration from SQLite to PostgreSQL has been completed at the database level, but the application code (models, repositories, services, and tests) needs to be aligned with the new PostgreSQL schema. This involves updating SQLAlchemy models to match the actual database structure, fixing missing models for new tables, updating repository code, and ensuring all tests pass with the new schema.

## Requirements

### Requirement 1

**User Story:** As a developer, I want the SQLAlchemy models to accurately reflect the PostgreSQL database schema, so that the application can properly interact with the database without errors.

#### Acceptance Criteria

1. WHEN examining model definitions THEN all table names SHALL match the PostgreSQL schema exactly
2. WHEN checking column definitions THEN all data types SHALL be PostgreSQL-compatible
3. WHEN validating constraints THEN all foreign keys, indexes, and check constraints SHALL match the database
4. WHEN reviewing missing models THEN models SHALL exist for all database tables
5. IF model attributes conflict with database columns THEN models SHALL be updated to match the database

### Requirement 2

**User Story:** As a developer, I want missing models for new database tables to be created, so that the application can access all database functionality.

#### Acceptance Criteria

1. WHEN analyzing the database schema THEN models SHALL be created for missing tables
2. WHEN creating new models THEN they SHALL follow the existing code patterns and conventions
3. WHEN adding trading models THEN they SHALL include proper relationships and constraints
4. WHEN implementing new models THEN they SHALL use PostgreSQL-specific features where appropriate
5. IF tables have complex relationships THEN models SHALL properly define foreign key relationships

### Requirement 3

**User Story:** As a developer, I want the job scheduling models to be corrected to match the actual database structure, so that job scheduling functionality works properly.

#### Acceptance Criteria

1. WHEN examining job models THEN column names SHALL match the database exactly (job_id as int8, not string)
2. WHEN checking run model THEN the primary key SHALL be run_id with UUID type
3. WHEN validating constraints THEN unique constraints SHALL match database definitions
4. WHEN reviewing indexes THEN all database indexes SHALL be reflected in the model
5. IF there are data type mismatches THEN models SHALL be corrected to match PostgreSQL types

### Requirement 4

**User Story:** As a developer, I want the telegram models to be updated to match the new database structure, so that telegram functionality continues to work correctly.

#### Acceptance Criteria

1. WHEN checking telegram models THEN missing fields SHALL be added to match database columns
2. WHEN validating telegram_feedbacks THEN missing columns (type, message, created, status) SHALL be added
3. WHEN examining telegram_settings THEN the model SHALL use the correct primary key definition
4. WHEN reviewing constraints THEN all database constraints SHALL be reflected in models
5. IF models have incorrect field definitions THEN they SHALL be corrected to match the database

### Requirement 5

**User Story:** As a developer, I want the user models to be corrected to match the PostgreSQL schema, so that user authentication and management works properly.

#### Acceptance Criteria

1. WHEN examining user models THEN the AuthIdentity model SHALL use correct attribute names
2. WHEN checking verification codes THEN the model SHALL include all database columns (provider, created_at)
3. WHEN validating constraints THEN unique constraints and indexes SHALL match the database
4. WHEN reviewing foreign keys THEN all relationships SHALL be properly defined
5. IF attribute names conflict with SQL keywords THEN proper column mapping SHALL be used

### Requirement 6

**User Story:** As a developer, I want repository and service code to be updated for the corrected models, so that data access operations work correctly with the new schema.

#### Acceptance Criteria

1. WHEN updating repositories THEN they SHALL work with the corrected model definitions
2. WHEN checking field access THEN repositories SHALL use correct attribute names
3. WHEN validating queries THEN they SHALL reference correct table and column names
4. WHEN testing operations THEN all CRUD operations SHALL work with the new models
5. IF repositories use deprecated model attributes THEN they SHALL be updated to use correct ones

### Requirement 7

**User Story:** As a developer, I want test factories and test cases to be updated for the new models, so that the test suite passes and provides reliable validation.

#### Acceptance Criteria

1. WHEN running tests THEN all model tests SHALL pass with the corrected definitions
2. WHEN using test factories THEN they SHALL create valid data for the new models
3. WHEN checking test imports THEN they SHALL reference correct model classes
4. WHEN validating test data THEN it SHALL be compatible with PostgreSQL constraints
5. IF tests reference old model attributes THEN they SHALL be updated to use correct ones

### Requirement 8

**User Story:** As a developer, I want comprehensive validation that the code alignment is complete, so that I can be confident the application works correctly with PostgreSQL.

#### Acceptance Criteria

1. WHEN running the full test suite THEN all tests SHALL pass without database-related errors
2. WHEN performing CRUD operations THEN they SHALL work correctly with all models
3. WHEN checking foreign key relationships THEN they SHALL be properly enforced
4. WHEN validating data integrity THEN constraints SHALL work as expected
5. IF any alignment issues remain THEN they SHALL be identified and documented for resolution