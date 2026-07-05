# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] JWT authentication and password verification.
- [x] Strategy instance creation, updates, and deletion.
- [x] Dynamic parameter hot-reloading for running strategies.
- [x] Robust service mocking and isolated unit testing for `StrategyManagementService`.

### 🔄 IN PROGRESS
- [ ] Refactoring of legacy routes and clean separation of DTOs.
- [ ] Complete unit test coverage for `SystemMonitoringService` and `WebUIAppService` mock mismatches.

### 🚀 PLANNED ENHANCEMENTS
- [ ] Real-time log streaming using WebSockets.
- [ ] Automated database migrations integration.

## Technical Debt
- [ ] Address mock discrepancies in `TestWebUIAppService` and `TestSystemMonitoringService` tests.

## Testing Requirements
- [x] Run pytest on all service classes.
- [x] Type checking with Pyright.
