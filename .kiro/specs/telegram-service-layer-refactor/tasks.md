# Implementation Plan

- [x] 1. Refactor business logic layer to use service dependencies
  - Create dependency injection pattern for telegram business logic
  - Modify business_logic.py to accept service instances as constructor parameters
  - Update all database operations to use telegram_service methods instead of direct SQL
  - _Requirements: 1.1, 1.3, 4.1, 4.2_

- [x] 1.1 Create service-aware business logic class
  - Define TelegramBusinessLogic class with service dependencies
  - Implement constructor that accepts telegram_service and indicator_service instances
  - Add proper type hints and documentation for service interfaces
  - _Requirements: 1.1, 4.1, 4.2_

- [x] 1.2 Replace direct database calls in business_logic.py
  - Remove sqlite3 imports and raw SQL queries
  - Replace direct database operations with telegram_service method calls
  - Update user status checks to use get_user_status() service method
  - Update language setting operations to use service layer methods
  - _Requirements: 1.1, 1.2, 1.4, 5.1, 5.2_

- [x] 1.3 Update command handlers to use service layer
  - Modify handle_register, handle_verify, handle_info functions to use telegram_service
  - Update handle_language function to use service layer for language updates
  - Replace direct user limit operations with set_user_limit service calls
  - _Requirements: 1.1, 1.3, 5.1, 5.2, 5.3_

- [x] 2. Replace indicator calculations with IndicatorService
  - Remove direct talib usage from telegram modules
  - Replace indicator_calculator.py functionality with IndicatorService calls
  - Update screener modules to use centralized indicator service
  - _Requirements: 2.1, 2.2, 2.5, 3.1, 3.2_

- [x] 2.1 Create indicator service integration layer
  - Add IndicatorService instance to business logic dependencies
  - Create helper methods to convert telegram requests to IndicatorService requests
  - Implement error handling for indicator service failures
  - _Requirements: 2.1, 2.4, 6.1, 6.2_

- [x] 2.2 Replace indicator_calculator.py usage
  - Update alert_logic_evaluator.py to use IndicatorService instead of local calculator
  - Remove direct talib imports from screener modules
  - Replace manual RSI, MACD, SMA, Bollinger Bands calculations with service calls
  - _Requirements: 2.1, 2.2, 2.5_
- [x] 2.3 Update screener modules for service integration
  - Modify enhanced_screener.py to use IndicatorService for technical analysis
  - Update fundamental_screener.py to use indicators service for fundamental data
  - Replace direct indicator calculations in screener business logic
  - _Requirements: 2.1, 2.3, 3.1, 3.2_

- [x] 3. Update bot initialization and dependency injection
  - Modify bot.py to create and inject service instances
  - Update command handlers to pass service dependencies to business logic
  - Ensure proper service lifecycle management
  - _Requirements: 1.1, 2.1, 4.1, 4.2_

- [x] 3.1 Initialize services in bot startup
  - Create telegram_service and indicator_service instances in bot.py
  - Add proper service configuration and initialization
  - Implement service health checks and error handling
  - _Requirements: 1.1, 2.1, 6.1, 6.3_

- [x] 3.2 Update command handler dependency injection
  - Modify all command handlers to receive service instances
  - Update audit_command_wrapper to pass services to business logic
  - Ensure consistent service usage across all command handlers
  - _Requirements: 1.1, 2.1, 4.1, 4.2_

- [x] 4. Remove direct database access patterns
  - Eliminate remaining sqlite3 imports from telegram modules
  - Remove raw SQL queries and direct database connections
  - Update test files to use service mocks instead of direct database access
  - _Requirements: 1.2, 1.4, 7.1, 7.2_

- [x] 4.1 Clean up direct database imports
  - Remove sqlite3 imports from business_logic.py and other telegram files
  - Remove direct model imports that bypass service layer
  - Update import statements to use only service layer interfaces
  - _Requirements: 1.2, 1.4_

- [x] 4.2 Update rearm_alert_system.py for service usage
  - Replace direct database update calls with service layer methods
  - Remove raw SQL operations and use telegram_service methods
  - Update alert status management to use service layer
  - _Requirements: 1.1, 1.3, 1.4_

- [x] 5. Update configuration and settings management
  - Ensure all user settings operations use telegram_service
  - Update admin operations to use service layer methods
  - Replace direct configuration database access with service calls
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5.1 Refactor user settings management
  - Update language preference changes to use service layer
  - Modify user verification operations to use telegram_service methods
  - Replace direct user limit modifications with service calls
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 5.2 Update admin operations for service usage
  - Modify admin command handlers to use telegram_service for user management
  - Update user approval operations to use service layer methods
  - Replace direct admin database operations with service calls
  - _Requirements: 5.4, 5.5_

- [x] 6. Implement comprehensive error handling
  - Add proper error handling for service layer failures
  - Implement graceful degradation for indicator service errors
  - Update logging to use service layer error reporting
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6.1 Add service layer error handling
  - Implement try-catch blocks for telegram_service operations
  - Add error handling for IndicatorService failures with fallback behavior
  - Create user-friendly error messages for service failures
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 6.2 Update logging for service layer usage
  - Ensure all service operations are properly logged
  - Add context information for service layer errors
  - Update audit logging to use service layer methods
  - _Requirements: 6.4, 6.5_

- [x] 7. Update tests for service layer integration
  - Modify unit tests to use service mocks instead of direct database
  - Add integration tests for service layer usage
  - Update test fixtures to work with service layer patterns
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 7.1 Create service layer mocks for testing
  - Implement mock classes for telegram_service and IndicatorService
  - Create test fixtures that use service mocks instead of database setup
  - Update existing tests to use dependency injection with mocks
  - _Requirements: 7.1, 7.2_

- [x] 7.2 Update unit tests for business logic
  - Modify business logic tests to inject mock services
  - Test business logic in isolation from database and external services
  - Add tests for error handling and service failure scenarios
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 7.3 Add integration tests for service usage
  - Create integration tests that verify service layer interactions
  - Test end-to-end command processing through service layers
  - Validate that service layer contracts are properly implemented
  - _Requirements: 7.4, 7.5_

- [x] 8. Clean up and remove deprecated code
  - Remove unused indicator calculation files
  - Delete deprecated direct database access code
  - Update documentation and examples to reflect service layer usage
  - _Requirements: 2.2, 1.2, 4.4_

- [x] 8.1 Remove indicator_calculator.py and related files
  - Delete src/telegram/screener/indicator_calculator.py
  - Remove any remaining direct talib usage in telegram modules
  - Update imports and references to use IndicatorService
  - _Requirements: 2.2, 2.3_

- [x] 8.2 Clean up deprecated database access code
  - Remove unused sqlite3 connection code
  - Delete raw SQL query functions that have been replaced
  - Clean up any remaining direct model imports
  - _Requirements: 1.2, 1.4_

- [x] 8.3 Update documentation and examples
  - Update code comments to reflect service layer usage
  - Add documentation for service layer integration patterns
  - Create examples showing proper service usage in telegram bot context
  - _Requirements: 4.4, 4.5_