# Requirements Document

## Introduction

The telegram bot module currently has architectural violations where it directly accesses database models and performs indicator calculations instead of using the proper service layers. This creates tight coupling, makes testing difficult, and violates the separation of concerns principle. This refactoring will ensure all database operations go through `src/data/db/services` and all indicator calculations go through `src/indicators` module.

## Requirements

### Requirement 1: Database Access Layer Compliance

**User Story:** As a developer, I want all telegram bot database operations to go through the service layer, so that the bot is decoupled from direct database access and follows proper architectural patterns.

#### Acceptance Criteria

1. WHEN the telegram bot needs to perform any database operation THEN it SHALL use services from `src/data/db/services` module
2. WHEN the telegram bot code is examined THEN it SHALL NOT contain direct imports from `src/data/db/models`
3. WHEN the telegram bot needs user operations THEN it SHALL use `telegram_service.py` functions
4. WHEN the telegram bot needs to execute raw SQL THEN it SHALL be refactored to use appropriate service methods
5. WHEN database schema changes occur THEN the telegram bot SHALL NOT require modifications due to service layer abstraction

### Requirement 2: Indicator Calculation Service Integration

**User Story:** As a developer, I want all indicator calculations in the telegram bot to use the centralized indicators service, so that calculations are consistent and maintainable across the application.

#### Acceptance Criteria

1. WHEN the telegram bot needs to calculate technical indicators THEN it SHALL use `src/indicators/service.py`
2. WHEN the telegram bot code is examined THEN it SHALL NOT contain direct imports of `talib` or manual indicator calculations
3. WHEN new indicators are added to the system THEN the telegram bot SHALL automatically support them through the service layer
4. WHEN indicator calculations fail THEN the telegram bot SHALL handle errors gracefully through the service interface
5. WHEN the telegram bot needs RSI, MACD, SMA, or Bollinger Bands THEN it SHALL request them from `IndicatorService`

### Requirement 3: Screener Module Service Integration

**User Story:** As a user, I want the screener functionality to use proper service layers, so that screening operations are consistent and reliable.

#### Acceptance Criteria

1. WHEN screener calculations are performed THEN they SHALL use `IndicatorService` for technical analysis
2. WHEN screener results are stored THEN they SHALL use appropriate database services
3. WHEN fundamental data is needed THEN it SHALL be requested through the indicators service
4. WHEN screener configurations are saved THEN they SHALL use service layer methods
5. WHEN screener reports are generated THEN they SHALL use service layer data access

### Requirement 4: Business Logic Layer Cleanup

**User Story:** As a developer, I want the business logic to be cleanly separated from data access concerns, so that the code is maintainable and testable.

#### Acceptance Criteria

1. WHEN business logic functions need data THEN they SHALL request it through service interfaces
2. WHEN business logic is tested THEN it SHALL be possible to mock service dependencies
3. WHEN business logic changes THEN it SHALL NOT require database schema knowledge
4. WHEN new features are added THEN they SHALL follow the established service layer pattern
5. WHEN errors occur in business logic THEN they SHALL be handled at the appropriate abstraction level

### Requirement 5: Configuration and Settings Management

**User Story:** As a user, I want my telegram bot settings and configurations to be managed through proper service layers, so that data integrity is maintained.

#### Acceptance Criteria

1. WHEN user settings are updated THEN they SHALL use `telegram_service` methods
2. WHEN language preferences are changed THEN they SHALL go through the service layer
3. WHEN user verification status is modified THEN it SHALL use service layer functions
4. WHEN admin operations are performed THEN they SHALL use appropriate service methods
5. WHEN user limits are set THEN they SHALL be persisted through the service layer

### Requirement 6: Error Handling and Logging Consistency

**User Story:** As a developer, I want consistent error handling across all telegram bot operations, so that issues can be diagnosed and resolved efficiently.

#### Acceptance Criteria

1. WHEN service layer operations fail THEN errors SHALL be properly logged and handled
2. WHEN database operations encounter issues THEN they SHALL be handled through service layer error handling
3. WHEN indicator calculations fail THEN errors SHALL be propagated appropriately
4. WHEN user operations fail THEN meaningful error messages SHALL be provided
5. WHEN system errors occur THEN they SHALL be logged with appropriate context

### Requirement 7: Testing and Mockability

**User Story:** As a developer, I want the refactored code to be easily testable with proper dependency injection, so that unit tests can be written effectively.

#### Acceptance Criteria

1. WHEN unit tests are written THEN service dependencies SHALL be easily mockable
2. WHEN integration tests are needed THEN service boundaries SHALL be clearly defined
3. WHEN testing database operations THEN they SHALL go through service layer mocks
4. WHEN testing indicator calculations THEN they SHALL use service layer interfaces
5. WHEN testing business logic THEN external dependencies SHALL be injectable