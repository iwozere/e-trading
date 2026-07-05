# Design

## Purpose
The API module acts as the service layer and entry point for Web UI operations. It bridges the gap between web clients (frontend) and the trading system execution layer.

## Architecture
The API module utilizes FastAPI to expose RESTful endpoints and WebSockets.

### High-Level Architecture
- **Router Layer** (`main.py`): Parses HTTP requests/WebSocket frames, validates payloads using Pydantic, and delegates logic to services.
- **Service Layer** (`services/`): Processes business rules (such as user authentication, configuration validation, system stats aggregation).
- **Security Layer** (`auth.py`): Performs JWT token encoding/decoding and password verification.

### Component Design
- `StrategyManagementService`: Manages strategy configuration validation, instance CRUD operations, and hot-reloading parameters on active instances.
- `SystemMonitoringService`: Leverages `psutil` to collect system resources (CPU, Memory, Disk) and trigger alerts.
- `WebUIAppService`: Handles web user registration, login, and user roles assignment.

## Data Flow
1. Client requests a strategy parameter update.
2. `main.py` route `/api/strategies/{id}/parameters` authenticates user and validates body.
3. `StrategyManagementService.update_strategy_parameters` updates configuration in the strategy instance.
4. If running, the service dynamically retrieves and calls `update_strategy_parameters` on `StrategyManager`.
5. Success/failure response returned to Client.

## Design Decisions
- **Service Pattern**: Offload controller actions to reusable service classes to separate presentation and domain logic.
- **Dynamic Attributes**: Use `getattr`/`hasattr` on manager instances to dynamically call enhanced runner methods without breaking compatibility with mock or basic runners.
