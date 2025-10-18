# Implementation Plan

- [x] 1. Fix job models to match PostgreSQL schema





- [x] 1.1 Correct Run model data types and constraints


  - Change job_id from String(255) to BigInteger to match int8 in database
  - Change user_id to BigInteger to match int8 in database  
  - Change job_type from String(50) to Text to match database
  - Change status from String(20) to Text to match database
  - Add missing worker_id column as String(255)
  - Fix unique constraint name to match database: "ux_runs_job_scheduled_for"
  - _Requirements: 1.1, 1.2, 3.1, 3.2_

- [x] 1.2 Update Schedule model to match database exactly


  - Verify all column types match PostgreSQL schema
  - Ensure constraint names match database definitions
  - Check index definitions are correct
  - _Requirements: 1.1, 1.2, 3.3, 3.4_

- [x] 1.3 Write unit tests for corrected job models


  - Test model creation with correct data types
  - Validate constraint enforcement
  - Test foreign key relationships
  - _Requirements: 3.1, 3.2_

- [ ] 1.4 Fix JSONB type issues in job models
  - Change JSON type to JSONB in Schedule and Run models to match PostgreSQL
  - Update test expectations to check for JSONB instead of JSON
  - Ensure JSONB functionality works correctly with PostgreSQL
  - _Requirements: 3.1, 3.3_

- [ ] 2. Fix telegram service integration issues
- [ ] 2.1 Add missing telegram_verification repository
  - Create TelegramVerificationRepo class in repos bundle
  - Implement required methods for verification operations
  - Update database service to include telegram_verification repo
  - Fix telegram service to use the new repository
  - _Requirements: 6.1, 6.2_

- [ ] 2.2 Fix telegram service jobs migration format issues
  - Fix alert format conversion from jobs system to expected format
  - Update schedule format conversion to handle missing attributes
  - Ensure backward compatibility with existing API expectations
  - Test all telegram service functions work correctly
  - _Requirements: 6.1, 6.4_

- [ ] 2.3 Fix TelegramBase import issues in tests
  - Add missing TelegramBase import to telegram repo tests
  - Update test setup to use correct base classes
  - Ensure all telegram tests can run without import errors
  - _Requirements: 7.1, 7.3_

- [ ] 2.4 Fix TelegramFeedback model with missing columns
  - Add type column as String(50)
  - Add message column as Text  
  - Add created column as DateTime(timezone=True)
  - Add status column as String(20)
  - Ensure foreign key relationship is correct
  - _Requirements: 1.1, 1.4, 4.2_

- [ ] 2.5 Verify and fix other telegram models
  - Check TelegramBroadcastLog matches database schema
  - Verify TelegramCommandAudit has all required indexes
  - Ensure TelegramSetting uses correct primary key definition
  - Update any missing constraints or indexes
  - _Requirements: 1.1, 1.3, 4.1, 4.4_

- [ ] 2.6 Write unit tests for updated telegram models
  - Test TelegramFeedback with all columns
  - Validate constraint enforcement
  - Test model relationships
  - _Requirements: 4.2, 4.4_

- [ ] 3. Correct user models to match PostgreSQL schema
- [ ] 3.1 Fix AuthIdentity model column mapping
  - Ensure identity_metadata attribute maps to "metadata" database column
  - Verify all constraint names match database exactly
  - Check index definitions are correct
  - Update foreign key relationship definitions
  - _Requirements: 1.1, 1.5, 5.1, 5.4_

- [ ] 3.2 Update VerificationCode model with missing columns
  - Add provider column as String(20) with default 'telegram'
  - Add created_at column as DateTime(timezone=True) with default now()
  - Ensure foreign key relationship is properly defined
  - Add missing index definitions
  - _Requirements: 1.1, 1.4, 5.2, 5.4_

- [ ] 3.3 Write unit tests for corrected user models
  - Test AuthIdentity with correct column mapping
  - Test VerificationCode with all columns
  - Validate foreign key relationships
  - _Requirements: 5.1, 5.2_

- [ ] 4. Verify and complete trading models
- [ ] 4.1 Review trading models against database schema
  - Check BotInstance model matches trading_bot_instances table
  - Verify Trade model matches trading_trades table exactly
  - Ensure Position model matches trading_positions table
  - Validate PerformanceMetric model matches trading_performance_metrics table
  - _Requirements: 1.1, 1.2, 2.3_

- [ ] 4.2 Fix any trading model discrepancies found
  - Correct data types to match PostgreSQL schema
  - Update constraint names and definitions
  - Fix index definitions
  - Ensure all foreign key relationships are properly defined
  - _Requirements: 1.1, 1.3, 2.3, 2.4_

- [ ] 4.3 Write unit tests for trading models
  - Test all trading model creation and relationships
  - Validate constraint enforcement
  - Test foreign key cascading behavior
  - _Requirements: 2.3, 2.4_

- [ ] 5. Update repository code for corrected models
- [ ] 5.1 Fix jobs repository for updated Run model
  - Update job_id parameter handling for BigInteger type
  - Fix any attribute access for corrected field names
  - Update query logic for new data types
  - Test all CRUD operations work correctly
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 5.2 Update users repository for corrected AuthIdentity model
  - Fix access to identity_metadata attribute (maps to metadata column)
  - Update any queries that reference the metadata column directly
  - Test all user authentication operations
  - Verify telegram profile operations work correctly
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 5.3 Update telegram repository for corrected models
  - Fix TelegramFeedback operations for new columns
  - Update any queries that reference corrected field names
  - Test all telegram-related CRUD operations
  - Ensure repository methods work with updated models
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 5.4 Write integration tests for updated repositories
  - Test all repository methods with corrected models
  - Validate CRUD operations work correctly
  - Test error handling with new constraints
  - _Requirements: 6.4, 6.5_

- [ ] 6. Update test factories for corrected models
- [ ] 6.1 Fix job model factories
  - Update make_run factory for BigInteger job_id and user_id
  - Add worker_id parameter to run factory
  - Ensure factory creates valid data for PostgreSQL constraints
  - Test factory-generated data works with database
  - _Requirements: 7.2, 7.4_

- [ ] 6.2 Update telegram model factories
  - Fix make_feedback factory for new TelegramFeedback columns
  - Add type, message, created, status parameters
  - Ensure all telegram factories create valid PostgreSQL data
  - Test factory compatibility with database constraints
  - _Requirements: 7.2, 7.4_

- [ ] 6.3 Update user model factories
  - Fix AuthIdentity factory for identity_metadata attribute
  - Update VerificationCode factory for new columns (provider, created_at)
  - Ensure factories create valid foreign key relationships
  - Test factory-generated data with PostgreSQL
  - _Requirements: 7.2, 7.4_

- [ ] 6.4 Write tests for updated factories
  - Test all factories create valid model instances
  - Validate factory data is PostgreSQL-compatible
  - Test factory relationships and constraints
  - _Requirements: 7.2, 7.3_

- [ ] 7. Update existing test cases for model changes
- [ ] 7.1 Fix model import tests
  - Update test_models_basic.py for corrected model definitions
  - Fix any test assertions that check old model attributes
  - Update table name checks for corrected models
  - Ensure all model import tests pass
  - _Requirements: 7.1, 7.3_

- [ ] 7.2 Update repository test cases
  - Fix test_jobs_repo.py for BigInteger job_id changes
  - Update test_users_repo.py for AuthIdentity attribute changes
  - Fix test_telegram_repo.py for TelegramFeedback model changes
  - Ensure all repository tests pass with corrected models
  - _Requirements: 7.1, 7.3, 7.5_

- [ ] 7.3 Update service test cases
  - Fix any service tests that use old model attributes
  - Update test data creation for corrected models
  - Ensure service tests work with PostgreSQL constraints
  - Validate all service functionality with updated models
  - _Requirements: 7.1, 7.3, 7.5_

- [ ] 7.4 Write comprehensive integration tests
  - Test complete workflows with corrected models
  - Validate end-to-end functionality works correctly
  - Test error handling and constraint enforcement
  - _Requirements: 7.1, 7.5_

- [ ] 8. Validate complete code alignment with database
- [ ] 8.1 Run comprehensive model validation tests
  - Test all models can be created and imported without errors
  - Validate all table names match PostgreSQL schema exactly
  - Check all column definitions match database structure
  - Verify all constraints and indexes are properly defined
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 8.2 Test all CRUD operations with PostgreSQL
  - Test create operations for all models
  - Test read operations and queries work correctly
  - Test update operations respect constraints
  - Test delete operations and foreign key cascading
  - _Requirements: 8.2, 8.4, 8.5_

- [ ] 8.3 Validate foreign key relationships and constraints
  - Test all foreign key relationships work correctly
  - Validate constraint enforcement (unique, check, not null)
  - Test cascade behavior for delete operations
  - Ensure referential integrity is maintained
  - _Requirements: 8.3, 8.4, 8.5_

- [ ] 8.4 Run complete test suite validation
  - Execute all unit tests and ensure they pass
  - Run all integration tests successfully
  - Validate all repository and service tests pass
  - Check that no database-related errors occur
  - _Requirements: 8.1, 8.5_

- [ ] 8.5 Performance and reliability validation
  - Test query performance with corrected models
  - Validate database connection stability
  - Test error handling and recovery
  - Measure application startup time and reliability
  - _Requirements: 8.2, 8.4_