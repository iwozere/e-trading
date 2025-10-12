# Design Document

## Overview

The backend test fixes will systematically address 79 failing tests across the web UI backend test suite. The design focuses on fixing core infrastructure issues including missing attributes, incorrect imports, API mismatches, and incomplete implementations while maintaining backward compatibility.

## Architecture

### Test Infrastructure Layer
- **pytest Configuration**: Ensure proper async support and fixture management
- **Database Session Management**: Fix context manager protocol implementation
- **Mock Strategy**: Replace inappropriate Mock returns with proper data types

### Model Layer Fixes
- **User Model Enhancement**: Add missing telegram_user_id attribute
- **Field Consistency**: Standardize field naming across modules
- **Relationship Validation**: Ensure foreign key relationships work correctly

### Service Layer Integration
- **Database Service Functions**: Complete missing authentication service methods
- **Async Function Handling**: Proper async/await pattern implementation
- **Session Context Management**: Fix __enter__ and __exit__ methods

### WebSocket Infrastructure
- **Connection Management**: Implement missing WebSocketConnection methods
- **Manager Aliasing**: Properly alias ConnectionManager to WebSocketManager
- **Message Handling**: Support all required message types and channels

## Components and Interfaces

### 1. User Model Enhancement
```python
class User:
    # Existing fields
    id: int
    email: str
    username: str
    
    # New required field
    telegram_user_id: Optional[int] = None
    
    # Methods remain unchanged
```

### 2. WebSocket Connection Interface
```python
class WebSocketConnection:
    def add_subscription(self, channel: str) -> None
    def remove_subscription(self, channel: str) -> None
    def has_subscription(self, channel: str) -> bool
    def update_last_ping(self) -> None
```

### 3. Database Session Context Manager
```python
class DatabaseSession:
    def __enter__(self) -> Session
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
```

### 4. Authentication Service Interface
```python
class AuthService:
    def get_user_by_email(self, email: str) -> Optional[User]
    def get_user_by_telegram_id(self, telegram_id: int) -> Optional[User]
    def validate_credentials(self, credentials: dict) -> bool
```

## Data Models

### User Model Schema
- **Primary Key**: `id` (integer)
- **Email**: `email` (string, unique)
- **Username**: `username` (string)
- **Telegram ID**: `telegram_user_id` (optional integer)
- **Created At**: `created_at` (datetime)
- **Updated At**: `updated_at` (datetime)

### WebSocket Connection Schema
- **Connection ID**: `connection_id` (string)
- **User ID**: `user_id` (integer, foreign key)
- **Subscriptions**: `subscriptions` (list of strings)
- **Last Ping**: `last_ping` (datetime)

## Error Handling

### Test Error Categories
1. **AttributeError**: Missing model attributes or methods
2. **ImportError**: Missing or incorrect module imports
3. **TypeError**: Incorrect data types returned from mocks
4. **AssertionError**: Test expectations not met

### Error Resolution Strategy
1. **Identify Root Cause**: Analyze failing test to understand missing component
2. **Implement Missing Component**: Add required attribute, method, or import
3. **Validate Fix**: Ensure fix doesn't break existing functionality
4. **Test Integration**: Verify fix works with related components

### Backward Compatibility
- All existing API endpoints remain unchanged
- Database migrations handle new fields gracefully
- Existing user data is preserved during model updates

## Testing Strategy

### Test Execution Phases
1. **Individual Test Fixes**: Address each failing test systematically
2. **Integration Validation**: Ensure fixes work together
3. **Regression Testing**: Verify no new failures introduced
4. **Performance Validation**: Ensure fixes don't impact performance

### Test Categories to Fix
- **Authentication Tests**: User model and auth service integration
- **WebSocket Tests**: Connection management and message handling
- **Service Tests**: Business logic and database integration
- **Route Tests**: API endpoint functionality
- **Model Tests**: Data validation and relationships

### Success Metrics
- **Test Pass Rate**: 100% (161/161 tests passing)
- **No Regression**: Existing functionality remains intact
- **Performance**: No significant performance degradation
- **Code Coverage**: Maintain or improve current coverage levels

### Mock Strategy Improvements
- Replace `Mock()` returns with proper data structures
- Use realistic test data that matches production schemas
- Implement proper async mock handling for async functions
- Ensure mock configurations match actual service interfaces

### Database Test Strategy
- Use proper transaction rollback for test isolation
- Implement proper session context management
- Ensure test database schema matches production
- Handle async database operations correctly in tests