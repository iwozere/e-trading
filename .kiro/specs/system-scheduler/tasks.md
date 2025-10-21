# Implementation Plan

- [x] 1. Set up core alert services infrastructure
  - Create directory structure for common alert services
  - Implement cron parsing service with 5/6 field support
  - Create JSON schema validation service for alert configurations
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 1.1 Create cron parser service
  - Implement CronParser class in src/common/alerts/cron_parser.py
  - Add support for both 5-field and 6-field cron expressions using croniter
  - Include timezone-aware datetime handling and validation
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 1.2 Create alert schema validator
  - Implement AlertSchemaValidator class in src/common/alerts/schema_validator.py
  - Create JSON schemas for alert and schedule task_params validation
  - Add schema loading, caching, and validation error reporting
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 2. Implement centralized alert evaluation service
  - Create AlertEvaluator service with rule evaluation and rearm logic
  - Integrate with existing market data and indicator services
  - Implement alert state management and persistence
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 2.1 Create alert evaluator core
  - Implement AlertEvaluator class in src/common/alerts/alert_evaluator.py
  - Add alert configuration parsing and validation
  - Implement rule tree evaluation with logical operators (and/or/not)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.2 Implement rearm logic
  - Port and refactor rearm logic from existing telegram services
  - Add hysteresis, cooldown, and persistence bar support
  - Implement crossing detection with side tracking
  - _Requirements: 2.4, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 2.3 Add market data and indicator integration
  - Integrate with DataManager for OHLCV data retrieval
  - Integrate with IndicatorService for technical indicator calculations
  - Handle data unavailability and provider failover
  - _Requirements: 6.2, 6.3, 2.1, 2.2_

- [x] 2.4 Implement alert state persistence
  - Add state_json field management for alert context
  - Implement state loading, updating, and error handling
  - Handle missing or corrupted state with default initialization
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 3. Create main scheduler service
  - Implement SchedulerService with APScheduler integration
  - Add job loading, registration, and execution management
  - Implement service lifecycle and error handling
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 3.1 Implement scheduler service core
  - Create SchedulerService class in src/scheduler/scheduler_service.py
  - Initialize APScheduler with PostgreSQL job store and ThreadPoolExecutor
  - Add service start/stop lifecycle management
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3.2 Add job loading and registration
  - Load enabled schedules from database using JobsService
  - Calculate next_run_at timestamps using CronParser
  - Register jobs with APScheduler for execution
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3.3 Implement job execution callbacks
  - Create job execution callback handlers
  - Update job_schedule_runs records with execution status
  - Handle job timeouts and worker thread exceptions
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 3.4 Add error handling and recovery
  - Implement retry logic with exponential backoff
  - Handle database connection failures during startup
  - Add comprehensive logging for troubleshooting
  - _Requirements: 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 3.5 Write integration tests for scheduler service
  - Test job loading and APScheduler registration
  - Test job execution and callback handling
  - Test error recovery and retry mechanisms
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 4. Complete notification integration
  - Integrate with existing notification service client
  - Update scheduler service to use NotificationServiceClient
  - Handle notification failures gracefully
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Update scheduler service notification integration
  - Modify SchedulerService to use existing NotificationServiceClient
  - Remove placeholder notification client code
  - Ensure proper error handling for notification failures
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.2 Add notification formatting enhancements
  - Enhance notification message formatting in SchedulerService._send_notification
  - Include more detailed market data and alert context
  - Format messages for optimal readability across channels
  - _Requirements: 4.1, 4.3, 4.4_

- [x] 4.3 Write tests for notification integration
  - Test notification service client integration with mock responses
  - Test notification formatting with various alert types
  - Test error handling and retry scenarios
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Create main scheduler application
  - Implement main application entry point
  - Add configuration management and dependency injection
  - Create service orchestration and startup logic
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 5.1 Create main application module
  - Implement main scheduler application in src/scheduler/main.py
  - Add configuration loading from environment and config files
  - Initialize all required services with proper dependency injection
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 5.2 Add service orchestration
  - Wire together SchedulerService, AlertEvaluator, and supporting services
  - Implement graceful startup and shutdown procedures
  - Add health check and status reporting endpoints
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 5.3 Create configuration management
  - Add configuration classes for scheduler settings
  - Support environment variable and file-based configuration
  - Include database connection, APScheduler, and service settings
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 5.4 Write integration tests for main application
  - Test complete service initialization and startup
  - Test configuration loading and validation
  - Test graceful shutdown and cleanup
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 6. Create deployment configuration
  - Create systemd service file for scheduler service
  - Add configuration templates and environment setup
  - Create startup and monitoring scripts
  - _Requirements: Service deployment and operational requirements_

- [x] 7. Add missing unit tests for core alert services








  - Create comprehensive unit tests for CronParser
  - Create comprehensive unit tests for AlertSchemaValidator  
  - Create comprehensive unit tests for AlertEvaluator components
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.1, 3.2, 3.3, 3.4, 3.5, 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 7.1 Create unit tests for CronParser


  - Create tests/test_cron_parser.py with comprehensive test coverage
  - Test 5-field and 6-field cron expression parsing
  - Test timezone handling and next run calculations
  - Test validation and error handling for invalid expressions
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 7.2 Create unit tests for AlertSchemaValidator


  - Create tests/test_alert_schema_validator.py with comprehensive test coverage
  - Test schema loading and caching mechanisms
  - Test validation with valid and invalid configurations
  - Test error message formatting and warnings
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 7.3 Create unit tests for AlertEvaluator






  - Create tests/test_alert_evaluator.py with comprehensive test coverage
  - Test rule evaluation with various logical operators (and/or/not)
  - Test rearm logic with different configurations (hysteresis, cooldown, persistence)
  - Test state persistence and recovery scenarios
  - Test market data and indicator integration
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 8. Update documentation and operational guides





  - Update HLA documentation with new scheduler architecture
  - Add monitoring and operational documentation
  - _Requirements: All requirements covered through comprehensive documentation_

- [x] 8.1 Update HLA documentation


  - Update docs/HLA/background-services.md with new scheduler implementation
  - Add architecture diagrams showing service interactions
  - Document configuration options and operational procedures
  - _Requirements: All requirements for comprehensive system documentation_

- [x] 8.2 Add operational documentation


  - Create troubleshooting guide for common issues
  - Add monitoring and alerting recommendations
  - Document backup and recovery procedures
  - _Requirements: Operational excellence and maintainability_