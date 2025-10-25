# Design Document

## Purpose

This document outlines the design for a comprehensive web-based user interface that manages both the enhanced multi-strategy trading system and Telegram bot operations. The web UI provides dual functionality through a unified interface: trading strategy management and Telegram bot administration, implemented with modern web technologies and following clean architecture principles.

## Architecture

### High-Level Architecture

The web UI follows a modern client-server architecture with dual-purpose functionality:

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Browser (Client)                     │
├─────────────────────────────────────────────────────────────┤
│  React Frontend Application (TypeScript + Material-UI)     │
│  ├── Trading Strategy Management                           │
│  │   ├── Strategy Dashboard & Monitoring                   │
│  │   ├── Configuration Forms & Wizards                     │
│  │   └── Performance Analytics Charts                      │
│  ├── Telegram Bot Management                               │
│  │   ├── User Management & Verification                    │
│  │   ├── Alert & Schedule Management                       │
│  │   ├── Broadcast Center                                  │
│  │   └── Audit Logs & Analytics                           │
│  └── System Administration                                 │
│      ├── Authentication & Role Management                  │
│      ├── System Monitoring & Health                        │
│      └── Configuration Management                          │
└─────────────────────────────────────────────────────────────┘
                              │
                    HTTP REST API + WebSocket
                              │
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Backend Server                      │
├─────────────────────────────────────────────────────────────┤
│  ├── Authentication & Authorization (JWT + RBAC)           │
│  ├── Trading Strategy API Endpoints                        │
│  ├── Telegram Bot Management API                           │
│  ├── System Monitoring & Metrics                           │
│  ├── WebSocket Manager (Real-time Updates)                 │
│  └── Application Services Layer                            │
│      ├── WebUI App Service                                 │
│      ├── Telegram App Service                              │
│      ├── Strategy Management Service                       │
│      └── System Monitoring Service                         │
└─────────────────────────────────────────────────────────────┘
                              │
                    Domain Services Integration
                              │
┌─────────────────────────────────────────────────────────────┐
│                Domain Services Layer                       │
├─────────────────────────────────────────────────────────────┤
│  ├── Enhanced Trading System                               │
│  │   ├── StrategyManager                          │
│  │   ├── RaspberryPiTradingService                        │
│  │   └── Configuration Management                         │
│  ├── Database Services                                     │
│  │   ├── Users Service                                     │
│  │   ├── Telegram Service                                  │
│  │   └── WebUI Service                                     │
│  └── Infrastructure Services                               │
│      ├── Database Service (SQLite)                         │
│      ├── Notification Logger                               │
│      └── System Monitoring                                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Design

#### 1. Frontend Architecture (React)

**Technology Stack:**
- React 18 with TypeScript
- Material-UI (MUI) v5 for component library and theming
- React Query (@tanstack/react-query) for data fetching and caching
- React Router DOM v6 for navigation
- Zustand for state management
- React Hook Form with Zod validation
- Recharts for performance analytics
- Socket.IO client for real-time communication
- React Hot Toast for notifications
- Vite for build tooling and development server

**Key Modules:**

1. **Authentication & Layout**
   - Login: JWT-based authentication with role support
   - Layout: Responsive sidebar navigation with role-based menu items
   - ProtectedRoute: Route guards with role-based access control
   - AuthStore: Zustand store for authentication state

2. **Trading Strategy Management**
   - Dashboard: System overview with key metrics and alerts
   - Strategies: Strategy list with status indicators and actions
   - StrategyForm: Create/edit strategy configurations with validation
   - Monitoring: Real-time strategy monitoring and performance
   - Analytics: Performance charts and trade analysis

3. **Telegram Bot Management**
   - TelegramDashboard: Overview of bot statistics and health
   - UserManagement: Telegram user verification and approval
   - AlertManagement: CRUD operations for user alerts
   - ScheduleManagement: Scheduled message management
   - BroadcastCenter: Mass messaging to approved users
   - AuditLogs: Comprehensive command and action logging

4. **System Administration**
   - Administration: System configuration and user management
   - SystemMetrics: Real-time system health monitoring
   - ConfigurationManager: Global system settings
   - LogViewer: Application and system log viewing

5. **Shared Components**
   - API Client: Axios-based HTTP client with authentication
   - WebSocket Context: Real-time communication management
   - Error Boundaries: Graceful error handling
   - Loading States: Consistent loading indicators
   - Form Components: Reusable form controls with validation

#### 2. Backend Architecture (FastAPI)

**Technology Stack:**
- FastAPI with Python 3.9+ and async/await support
- SQLAlchemy ORM with SQLite database
- Pydantic v2 for data validation and serialization
- JWT (PyJWT) for authentication and authorization
- Uvicorn ASGI server for production deployment
- CORS middleware for cross-origin requests
- HTTPBearer security for API authentication

**Application Services Layer:**
- WebUIAppService: Core web UI operations and database management
- TelegramAppService: Telegram bot management and user operations
- StrategyManagementService: Trading strategy lifecycle management
- SystemMonitoringService: System health and performance monitoring

**API Structure:**

1. **Authentication API**
   ```
   POST   /auth/login                  # User login with JWT token
   POST   /auth/refresh                # Refresh JWT token
   POST   /auth/logout                 # User logout
   GET    /api/test-auth               # Test authentication endpoint
   ```

2. **Strategy Management API**
   ```
   GET    /api/strategies              # List all strategies with status
   POST   /api/strategies              # Create new strategy
   GET    /api/strategies/{id}         # Get strategy details
   PUT    /api/strategies/{id}         # Update strategy configuration
   DELETE /api/strategies/{id}         # Delete strategy
   POST   /api/strategies/{id}/start   # Start strategy with confirmation
   POST   /api/strategies/{id}/stop    # Stop strategy gracefully
   POST   /api/strategies/{id}/restart # Restart strategy
   PUT    /api/strategies/{id}/parameters # Update runtime parameters
   ```

3. **Configuration Management API**
   ```
   GET    /api/config/templates        # Get strategy templates
   POST   /api/config/validate         # Validate strategy configuration
   GET    /api/system/status           # Get overall system status
   ```

4. **System Monitoring API**
   ```
   GET    /api/monitoring/metrics      # Get comprehensive system metrics
   GET    /api/monitoring/alerts       # Get system alerts
   POST   /api/monitoring/alerts/{id}/acknowledge # Acknowledge alert
   GET    /api/monitoring/history      # Get performance history
   ```

5. **Telegram Bot Management API**
   ```
   GET    /api/telegram/stats          # Get Telegram bot statistics
   GET    /api/telegram/users          # List Telegram users with filters
   POST   /api/telegram/users/{id}/verify # Verify Telegram user
   POST   /api/telegram/users/{id}/approve # Approve Telegram user
   POST   /api/telegram/users/{id}/reset-email # Reset user email
   GET    /api/telegram/alerts         # List Telegram alerts
   POST   /api/telegram/alerts/{id}/toggle # Toggle alert status
   DELETE /api/telegram/alerts/{id}    # Delete alert
   GET    /api/telegram/schedules      # List scheduled messages
   POST   /api/telegram/broadcast      # Send broadcast message
   GET    /api/telegram/broadcast/history # Get broadcast history
   GET    /api/telegram/audit          # Get audit logs with filtering
   ```

#### 3. Real-Time Communication

**WebSocket Implementation:**
- WebSocketManager: Centralized WebSocket connection management
- Real-time event broadcasting for strategy updates
- Connection authentication and user-specific channels
- Automatic reconnection and error handling

**WebSocket Events:**
- `strategy_status_update`: Strategy status and performance changes
- `trade_executed`: New trade notifications with details
- `system_alert`: System alerts and error notifications
- `performance_update`: Real-time performance metrics
- `system_metrics`: CPU, memory, temperature updates
- `telegram_event`: Telegram bot activity notifications

#### 4. Database Schema Integration

**Existing Database Models (from src.data.db.models):**
1. **model_users.User**: Web UI and Telegram user authentication
   - JWT-compatible authentication with role-based access
   - Integration with Telegram user profiles
   - Password hashing and verification

2. **model_webui.WebUIAuditLog**: Web UI action auditing
   - User action logging for security and compliance
   - Resource-specific audit trails
   - IP address and user agent tracking

3. **model_webui.WebUIConfig**: System configuration storage
   - Key-value configuration management
   - Versioned configuration changes
   - JSON-serialized complex configurations

4. **model_webui.WebUIStrategyTemplate**: Strategy templates
   - Reusable strategy configurations
   - User-created and system templates
   - Template sharing and versioning

5. **model_webui.WebUIPerformanceSnapshot**: Performance tracking
   - Periodic strategy performance data
   - Historical performance analysis
   - Real-time metrics aggregation

**Telegram Integration Models:**
- Telegram user profiles and verification status
- Alert and schedule management
- Broadcast history and audit logs
- Command execution tracking

### Security Design

#### Authentication and Authorization

1. **JWT-based Authentication**
   - Secure token-based authentication
   - Configurable token expiration
   - Refresh token mechanism

2. **Role-Based Access Control (RBAC)**
   - Admin: Full system access
   - Trader: Strategy management and monitoring
   - Viewer: Read-only access to monitoring

3. **Operation-Level Security**
   - Live trading operations require elevated privileges
   - Critical operations require confirmation
   - All actions logged for audit

#### API Security

1. **Input Validation**
   - Pydantic models for request validation
   - SQL injection prevention
   - XSS protection

2. **Rate Limiting**
   - API endpoint rate limiting
   - WebSocket connection limits
   - Brute force protection

### Performance Considerations

#### Frontend Optimization

1. **Code Splitting**
   - Lazy loading of route components
   - Dynamic imports for heavy libraries
   - Bundle size optimization

2. **Data Management**
   - React Query for efficient caching
   - Optimistic updates for better UX
   - Pagination for large datasets

3. **Real-Time Updates**
   - Efficient WebSocket message handling
   - Selective component re-rendering
   - Debounced updates for high-frequency data

#### Backend Optimization

1. **Database Optimization**
   - Efficient queries with proper indexing
   - Connection pooling
   - Query result caching

2. **API Performance**
   - Async/await for non-blocking operations
   - Background tasks for heavy operations
   - Response compression

### Deployment Architecture

#### Development Environment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vite Dev      │    │   FastAPI Dev   │    │   Trading       │
│   Server        │    │   Server        │    │   Service       │
│   (Port 5002)   │    │   (Port 5003)   │    │   (Background)  │
│   - HMR         │    │   - Auto-reload │    │   - Optional    │
│   - Proxy API   │    │   - Debug Mode  │    │   - Mock Mode   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Development Startup Options:**
- `bin/web_ui/start_webui_dev.bat` (Windows)
- `python src/web_ui/run_web_ui.py --dev` (Linux/macOS)
- Manual two-step startup for debugging
- Environment validation and dependency checking

#### Production Environment (Raspberry Pi)
```
┌─────────────────────────────────────────────────────────────┐
│                    Systemd Service                         │
│                  (trading-webui.service)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 WebUI Runner Process                       │
│                (src/web_ui/run_web_ui.py)                  │
├─────────────────────────────────────────────────────────────┤
│  ├── Frontend Build Serving (Static Files)                 │
│  ├── FastAPI Backend (Port 5003)                          │
│  ├── Database Management (SQLite)                         │
│  └── Process Management & Monitoring                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     Database Layer          │    │   Enhanced Trading Service  │
│   - SQLite Database         │    │   - Strategy Management     │
│   - User Management         │    │   - Real-time Monitoring    │
│   - Audit Logging           │    │   - Configuration Files     │
│   - Configuration Storage   │    │   - Performance Tracking    │
└─────────────────────────────┘    └─────────────────────────────┘
```

**Production Features:**
- Systemd service integration with auto-restart
- Production-optimized frontend builds
- Comprehensive logging and monitoring
- Graceful shutdown and error recovery
- Resource optimization for Raspberry Pi

## Implementation Status

- **Phase 1**: Backend API development ✅ **COMPLETED**
  - FastAPI application with comprehensive REST API
  - JWT authentication and role-based access control
  - Strategy management and system monitoring endpoints
  - Telegram bot management API integration

- **Phase 2**: Frontend React application ✅ **COMPLETED**
  - React 18 with TypeScript and Material-UI
  - Comprehensive trading strategy management interface
  - Full Telegram bot administration capabilities
  - Responsive design with mobile support

- **Phase 3**: Real-time communication ⚠️ **PARTIALLY IMPLEMENTED**
  - WebSocket infrastructure prepared
  - Real-time updates temporarily disabled in frontend
  - Backend WebSocket manager implemented

- **Phase 4**: Security and authentication ✅ **COMPLETED**
  - JWT-based authentication system
  - Role-based access control (Admin, Trader, Viewer)
  - Comprehensive audit logging
  - Secure API endpoints with proper validation

- **Phase 5**: Performance optimization ✅ **COMPLETED**
  - Vite build system for optimal frontend performance
  - React Query for efficient data caching
  - Lazy loading and code splitting
  - Production-optimized builds

- **Phase 6**: Production deployment ✅ **COMPLETED**
  - Cross-platform startup scripts (Windows/Linux)
  - Systemd service integration for Raspberry Pi
  - Production runner with process management
  - Comprehensive documentation and troubleshooting guides

## Integration Points

### Enhanced Trading System Integration

The web UI integrates with the existing enhanced trading system through:

1. **Service Communication**: Direct API calls to StrategyManager
2. **Configuration Management**: Read/write access to JSON configuration files
3. **Real-Time Monitoring**: WebSocket integration with strategy status updates
4. **System Control**: Integration with systemd service management
5. **Optional Operation**: Can run independently when trading system is unavailable

### Telegram Bot System Integration

The web UI provides comprehensive management of the Telegram bot system:

1. **User Management**: Direct integration with Telegram user database
2. **Alert Management**: CRUD operations for user alerts and schedules
3. **Broadcast System**: Mass messaging capabilities to approved users
4. **Audit Integration**: Comprehensive logging of all bot interactions
5. **Statistics Dashboard**: Real-time bot usage and performance metrics

### Database Service Integration

Clean architecture through domain services:

1. **Users Service**: Telegram user profile management
2. **Telegram Service**: Bot operations and command handling
3. **WebUI Service**: Web-specific configurations and audit logs
4. **Database Service**: Centralized database connection management

### Data Flow

1. **Trading Configuration Flow**: 
   Web UI → FastAPI → StrategyManagementService → StrategyManager → JSON Config

2. **Telegram Management Flow**: 
   Web UI → FastAPI → TelegramAppService → Domain Services → Database

3. **Monitoring Flow**: 
   Trading Service → SystemMonitoringService → FastAPI → WebSocket → Web UI

4. **Authentication Flow**: 
   Web UI → FastAPI → JWT Validation → Database → Role-based Access Control

## Error Handling

### Frontend Error Handling

1. **Network Errors**: Retry mechanisms with exponential backoff
2. **Validation Errors**: Real-time form validation with clear error messages
3. **WebSocket Disconnections**: Automatic reconnection with status indicators
4. **Component Errors**: Error boundaries with graceful fallbacks

### Backend Error Handling

1. **API Errors**: Structured error responses with error codes
2. **Service Integration Errors**: Graceful degradation when trading service is unavailable
3. **Database Errors**: Transaction rollback and error logging
4. **Authentication Errors**: Secure error messages without information leakage

## Testing Strategy

### Testing Architecture

The web UI requires comprehensive testing at multiple levels to ensure reliability and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Testing Pyramid                         │
├─────────────────────────────────────────────────────────────┤
│  E2E Tests (Cypress)                                       │
│  ├── Critical user workflows                               │
│  ├── Cross-browser compatibility                           │
│  └── Production-like environment testing                   │
├─────────────────────────────────────────────────────────────┤
│  Integration Tests                                          │
│  ├── API integration with test database                    │
│  ├── Service layer integration                             │
│  ├── WebSocket communication testing                       │
│  └── Authentication workflow testing                       │
├─────────────────────────────────────────────────────────────┤
│  Unit Tests (Largest Layer - 80%+ Coverage)                │
│  ├── Backend: pytest + FastAPI TestClient                  │
│  ├── Frontend: Jest + React Testing Library                │
│  ├── Services: Mock dependencies and test logic            │
│  └── Components: Isolated component behavior               │
└─────────────────────────────────────────────────────────────┘
```

### Backend Testing Framework

**Technology Stack:**
- **pytest**: Primary testing framework with fixtures and parametrization
- **pytest-asyncio**: Async test support for FastAPI endpoints
- **httpx**: HTTP client for API testing
- **pytest-mock**: Mocking framework for dependencies
- **coverage.py**: Code coverage measurement and reporting
- **factory-boy**: Test data generation and fixtures

**Testing Structure:**
```
src/web_ui/backend/tests/
├── conftest.py                 # Shared fixtures and configuration
├── test_auth.py               # Authentication and authorization tests
├── test_api_endpoints.py      # API endpoint tests
├── test_services/             # Application service tests
│   ├── test_webui_app_service.py
│   ├── test_telegram_app_service.py
│   └── test_strategy_service.py
├── test_models.py             # Database model tests
├── test_integration/          # Integration tests
│   ├── test_auth_workflow.py
│   ├── test_strategy_management.py
│   └── test_telegram_operations.py
└── fixtures/                  # Test data and fixtures
```

### Frontend Testing Framework

**Technology Stack:**
- **Jest**: JavaScript testing framework with mocking capabilities
- **React Testing Library**: Component testing with user-centric approach
- **MSW (Mock Service Worker)**: API mocking for integration tests
- **@testing-library/jest-dom**: Custom Jest matchers for DOM testing
- **@testing-library/user-event**: User interaction simulation
- **jest-environment-jsdom**: DOM environment for React testing

**Testing Structure:**
```
src/web_ui/frontend/src/tests/
├── setup.ts                   # Test configuration and global setup
├── __mocks__/                 # Mock implementations
│   ├── api.ts                # API client mocks
│   └── websocket.ts          # WebSocket mocks
├── components/               # Component unit tests
│   ├── Auth/
│   ├── Telegram/
│   ├── Strategies/
│   └── Layout/
├── pages/                    # Page component tests
├── stores/                   # State management tests
├── hooks/                    # Custom hook tests
├── utils/                    # Utility function tests
└── integration/              # Frontend integration tests
```

### Testing Requirements by Component

#### 1. Authentication Testing
- **Unit Tests**: JWT token handling, role validation, password hashing
- **Integration Tests**: Complete login/logout workflows
- **Security Tests**: Authorization bypass attempts, token manipulation

#### 2. Strategy Management Testing
- **Unit Tests**: CRUD operations, validation logic, state management
- **Integration Tests**: End-to-end strategy lifecycle management
- **Mock Tests**: Trading system integration without actual trading

#### 3. Telegram Bot Management Testing
- **Unit Tests**: User management operations, alert CRUD, broadcast functionality
- **Integration Tests**: Database operations and service interactions
- **Mock Tests**: Telegram API interactions and message delivery

#### 4. Real-Time Communication Testing
- **Unit Tests**: WebSocket connection management, event handling
- **Integration Tests**: Real-time data synchronization
- **Load Tests**: Multiple concurrent WebSocket connections

### Test Coverage Requirements

**Minimum Coverage Targets:**
- **Backend API Endpoints**: 90% coverage
- **Application Services**: 85% coverage
- **Frontend Components**: 80% coverage
- **Authentication Logic**: 95% coverage
- **Database Operations**: 85% coverage
- **Overall Project**: 80% coverage

**Coverage Exclusions:**
- Configuration files and constants
- Third-party library integrations
- Development-only utilities
- Generated code and migrations

### Continuous Integration Testing

**Automated Testing Pipeline:**
1. **Pre-commit Hooks**: Linting, formatting, basic unit tests
2. **Pull Request Testing**: Full test suite execution
3. **Coverage Reporting**: Automated coverage analysis and reporting
4. **Performance Testing**: Automated performance regression detection
5. **Security Testing**: Automated vulnerability scanning

**Test Execution Strategy:**
- **Fast Feedback**: Unit tests run on every commit
- **Comprehensive Testing**: Full suite on pull requests
- **Nightly Testing**: Extended integration and performance tests
- **Release Testing**: Complete E2E testing before deployment

## Monitoring and Observability

### Application Monitoring

1. **Performance Metrics**: Response times, error rates, throughput
2. **User Analytics**: Feature usage, user behavior tracking
3. **System Health**: Database connections, service availability
4. **Real-Time Alerts**: Critical error notifications

### Logging Strategy

1. **Structured Logging**: JSON-formatted logs with correlation IDs
2. **Log Levels**: Appropriate log levels for different environments
3. **Log Aggregation**: Centralized logging for analysis
4. **Log Retention**: Configurable retention policies