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

- [x] 1.3 Write unit tests for core services
  - Create tests for CronParser with various cron expressions
  - Create tests for AlertSchemaValidator with valid/invalid configurations
  - Test edge cases and error conditions
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.1, 3.2, 3.3, 3.4, 3.5_

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

- [x] 2.5 Write unit tests for alert evaluator
  - Test rule evaluation with various logical combinations
  - Test rearm logic with different configurations
  - Test state persistence and recovery scenarios
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2, 8.3, 8.4, 8.5_

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

- [-] 4. Implement notification integration



  - Add TelegramBot FastAPI endpoint integration
  - Create notification formatting and delivery logic
  - Handle notification failures gracefully
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 4.1 Create notification client


  - Implement TelegramNotificationClient in src/scheduler/notification_client.py
  - Add FastAPI endpoint calling with proper error handling
  - Include alert context and market conditions in notifications
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 4.2 Add notification formatting
  - Create notification message templates for different alert types
  - Include relevant market data and alert details
  - Format messages for optimal readability
  - _Requirements: 4.1, 4.3, 4.4_

- [ ] 4.3 Implement delivery error handling
  - Add retry logic for failed notification deliveries
  - Log notification failures without stopping job execution
  - Handle TelegramBot service unavailability gracefully
  - _Requirements: 4.2, 4.5_

- [ ] 4.4 Write tests for notification integration
  - Test FastAPI endpoint calling with mock responses
  - Test notification formatting with various alert types
  - Test error handling and retry scenarios
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 5. Create main scheduler application
  - Implement main application entry point
  - Add configuration management and dependency injection
  - Create service orchestration and startup logic
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 5.1 Create main application module
  - Implement main scheduler application in src/scheduler/main.py
  - Add configuration loading from environment and config files
  - Initialize all required services with proper dependency injection
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 5.2 Add service orchestration
  - Wire together SchedulerService, AlertEvaluator, and supporting services
  - Implement graceful startup and shutdown procedures
  - Add health check and status reporting endpoints
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 5.3 Create configuration management
  - Add configuration classes for scheduler settings
  - Support environment variable and file-based configuration
  - Include database connection, APScheduler, and service settings
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 5.4 Write integration tests for main application
  - Test complete service initialization and startup
  - Test configuration loading and validation
  - Test graceful shutdown and cleanup
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 6. Update documentation and create deployment artifacts
  - Update HLA documentation with new scheduler architecture
  - Create deployment configuration and service files
  - Add monitoring and operational documentation
  - _Requirements: All requirements covered through comprehensive documentation_

- [ ] 6.1 Update HLA documentation
  - Update docs/HLA/background-services.md with new scheduler implementation
  - Add architecture diagrams showing service interactions
  - Document configuration options and operational procedures
  - _Requirements: All requirements for comprehensive system documentation_

- [ ] 6.2 Create deployment configuration
  - Create systemd service file for scheduler service
  - Add configuration templates and environment setup
  - Create startup and monitoring scripts
  - _Requirements: Service deployment and operational requirements_

- [ ] 6.3 Add operational documentation
  - Create troubleshooting guide for common issues
  - Add monitoring and alerting recommendations
  - Document backup and recovery procedures
  - _Requirements: Operational excellence and maintainability_