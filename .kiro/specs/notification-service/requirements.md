# Notification Service Requirements - Database-Centric Architecture

## Introduction

The Notification Service is being refactored to a **database-centric architecture** where it operates as a pure message delivery engine without REST endpoints. All client interactions will be handled through the Main API Service, with communication between services occurring exclusively through the database. This approach consolidates all REST operations in a single location while maintaining the notification service as an autonomous delivery engine.

## Glossary

- **Notification Service**: Autonomous message delivery engine that processes messages from database and delivers to external channels
- **Main API Service**: Centralized REST API service that handles all client interactions and database operations
- **Message Queue**: Database-backed queue for storing pending notifications (no in-memory queues)
- **Channel Plugin**: Modular component responsible for delivering messages to specific channels (Telegram, Email, etc.)
- **Database-Centric Communication**: All inter-service communication occurs through database operations, no HTTP calls
- **Unified Analytics Service**: Consolidated analytics module in Main API that handles both notification and trading analytics
- **Delivery Engine**: Core processing component that polls database and delivers messages to external APIs
- **Health Heartbeat**: Database-only health reporting mechanism without REST endpoints

## Requirements

### Requirement 1

**User Story:** As a system architect, I want the notification service to be a pure delivery engine without REST endpoints, so that all client interactions are consolidated in the Main API Service and communication occurs through the database.

#### Acceptance Criteria

1. THE Notification Service SHALL NOT expose any REST API endpoints
2. THE Notification Service SHALL poll the database for pending messages
3. THE Notification Service SHALL communicate status updates only through database writes
4. THE Notification Service SHALL report health status only to the database
5. THE Main API Service SHALL handle all client REST interactions for notifications

### Requirement 2

**User Story:** As a developer, I want to send notifications through the Main API Service, so that I have a single endpoint for all system interactions and don't need to discover multiple services.

#### Acceptance Criteria

1. THE Main API Service SHALL provide REST endpoints for creating notifications
2. THE Main API Service SHALL write notification messages directly to the database
3. THE Main API Service SHALL provide REST endpoints for querying notification status
4. THE Main API Service SHALL provide REST endpoints for notification analytics
5. THE Notification Service SHALL automatically process messages written to the database

### Requirement 3

**User Story:** As a system administrator, I want all analytics consolidated in the Main API Service, so that I can access both notification and trading analytics through a unified interface.

#### Acceptance Criteria

1. THE Main API Service SHALL provide unified analytics endpoints for notifications
2. THE Main API Service SHALL provide unified analytics endpoints for trading data
3. THE Main API Service SHALL query notification data directly from the database
4. THE Main API Service SHALL support analytics patterns reusable across domains
5. THE Notification Service SHALL NOT provide any analytics endpoints

### Requirement 4

**User Story:** As a system operator, I want health monitoring consolidated in the Main API Service, so that I can view overall system health through a single interface.

#### Acceptance Criteria

1. THE Main API Service SHALL provide consolidated health endpoints
2. THE Main API Service SHALL aggregate health data from all subsystems
3. THE Notification Service SHALL report health status only to the database
4. THE Main API Service SHALL query health data directly from the database
5. THE Main API Service SHALL provide channel health status through REST endpoints

### Requirement 5

**User Story:** As a notification service, I want to focus solely on message delivery, so that I can optimize for reliability and performance without REST API overhead.

#### Acceptance Criteria

1. THE Notification Service SHALL poll the database for PENDING messages
2. THE Notification Service SHALL deliver messages through channel plugins
3. THE Notification Service SHALL update message status in the database
4. THE Notification Service SHALL record delivery results in the database
5. THE Notification Service SHALL implement retry logic and error handling

### Requirement 6

**User Story:** As a system administrator, I want administrative operations in the Main API Service, so that I can manage all system operations through a single authenticated interface.

#### Acceptance Criteria

1. THE Main API Service SHALL provide administrative endpoints with proper authentication
2. THE Main API Service SHALL handle notification cleanup operations
3. THE Main API Service SHALL provide processor statistics through database queries
4. THE Main API Service SHALL support administrative operations for both notifications and trading
5. THE Notification Service SHALL NOT provide any administrative endpoints

### Requirement 7

**User Story:** As a developer, I want channel plugins to remain in the notification service, so that delivery logic stays close to the processing engine while configuration is managed through the Main API.

#### Acceptance Criteria

1. THE Notification Service SHALL maintain channel plugin architecture
2. THE Notification Service SHALL load channel configurations from the database
3. THE Notification Service SHALL report channel health to the database
4. THE Main API Service SHALL provide REST endpoints for channel configuration
5. THE Main API Service SHALL provide REST endpoints for channel health status

### Requirement 8

**User Story:** As a system integrator, I want seamless migration from the current architecture, so that existing functionality continues to work while moving to the new database-centric approach.

#### Acceptance Criteria

1. THE Main API Service SHALL provide backward-compatible notification endpoints
2. THE database schema SHALL remain compatible with existing data
3. THE migration SHALL preserve all existing message and delivery history
4. THE Notification Service SHALL continue processing existing pending messages
5. THE migration SHALL be completed without service downtime

### Requirement 9

**User Story:** As a system architect, I want rate limiting and priority handling in the notification service, so that delivery policies are enforced at the processing level while configuration is managed through the Main API.

#### Acceptance Criteria

1. THE Notification Service SHALL implement per-user rate limiting from database configuration
2. THE Notification Service SHALL process messages based on priority levels
3. THE Notification Service SHALL bypass rate limits for critical messages
4. THE Main API Service SHALL provide REST endpoints for rate limit configuration
5. THE Main API Service SHALL provide REST endpoints for priority configuration

### Requirement 10

**User Story:** As a system operator, I want comprehensive delivery tracking through the Main API Service, so that I can monitor notification performance and troubleshoot issues through a unified interface.

#### Acceptance Criteria

1. THE Notification Service SHALL record all delivery attempts in the database
2. THE Notification Service SHALL track response times and external message IDs
3. THE Main API Service SHALL provide REST endpoints for delivery history
4. THE Main API Service SHALL provide REST endpoints for delivery statistics
5. THE Main API Service SHALL support delivery analytics and reporting

### Requirement 11

**User Story:** As a system architect, I want the notification service to be stateless, so that it can be scaled horizontally without coordination between instances.

#### Acceptance Criteria

1. THE Notification Service SHALL NOT maintain any in-memory state
2. THE Notification Service SHALL coordinate through database locks for message processing
3. THE Notification Service SHALL support multiple concurrent instances
4. THE Notification Service SHALL handle instance failures gracefully
5. THE Notification Service SHALL resume processing after restarts without data loss

### Requirement 12

**User Story:** As a developer, I want unified error handling patterns, so that both notification and trading errors are handled consistently through the Main API Service.

#### Acceptance Criteria

1. THE Main API Service SHALL provide consistent error response formats
2. THE Main API Service SHALL handle both notification and trading errors uniformly
3. THE Notification Service SHALL record errors in the database with structured format
4. THE Main API Service SHALL provide error analytics across all domains
5. THE Main API Service SHALL support error notification and alerting