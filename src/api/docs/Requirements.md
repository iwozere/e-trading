# Requirements

## Python Dependencies
- `fastapi` >= 0.100.0
- `pydantic` >= 2.0.0
- `sqlalchemy` >= 2.0.0
- `psutil` >= 5.9.0
- `websockets` >= 11.0.0

## External Dependencies
- `src.trading` - For strategy control and lifecycle
- `src.notification` - For notification service logs
- `src.data` - For database persistence
- `src.common` - For shared utilities
- `src.model` - For shared schemas and models
- `src.config` - For configuration management

## External Services
- SQLite database
- External trading system runner

## Security Requirements
- JWT token authentication
- Password hashing using bcrypt
- Route guards and permissions (admin, trader, viewer)

## Performance Requirements
- Dynamic parameter updates under 1s
- Concurrent WebSockets support for real-time monitoring updates
