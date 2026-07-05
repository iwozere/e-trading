# Web API

## Overview
Web API module providing REST and WebSocket endpoints for managing strategies, users, authentication, system monitoring, and notifications.

## Features
- JWT user authentication and permissions management
- Strategy CRUD operations and parameter hot-reloading
- Real-time system monitoring metrics (CPU, Memory, Disk)
- Notification log history and channel settings

## Quick Start
To run the Web API service locally:

```bash
uvicorn src.api.main:app --reload
```

## Integration
This module integrates with:
- `src.trading` - For strategy control and lifecycle
- `src.notification` - For notification service logs
- `src.data` - For database persistence

## Configuration
Basic configuration is managed via environment variables defined in `.env` and `src.api.config`.

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
