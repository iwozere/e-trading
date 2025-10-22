# Web UI Testing Guide

This document provides comprehensive instructions for running unit tests for the Web UI module, including both backend Python tests and frontend TypeScript tests.

## ⚠️ **Current Status**

- ✅ **Backend Tests**: Fully functional and ready to run
- ⚠️ **Frontend Tests**: Basic framework created, TypeScript configuration needs refinement
- ✅ **Test Infrastructure**: Complete with runners, coverage, and documentation

## Overview

The Web UI module includes comprehensive unit test coverage for:

### Backend Tests (Python/pytest)
- **Authentication & Authorization**: JWT token management, role-based access control
- **API Endpoints**: All REST API endpoints with request/response validation
- **Application Services**: Business logic and database operations
- **WebSocket Manager**: Real-time communication functionality
- **Telegram Routes**: Telegram bot management endpoints

### Frontend Tests (TypeScript/Vitest)
- **Authentication Components**: Login form, auth store, protected routes
- **Dashboard Components**: System status, metrics display, strategy overview
- **Telegram Management**: User management, alert management, broadcast center
- **Shared Components**: Layout, navigation, form utilities
- **State Management**: Zustand stores and React Query integration

## Quick Start

### Prerequisites

**Backend Tests:**
- Python 3.8+
- pip (Python package manager)

**Frontend Tests:**
- Node.js 16+
- npm (Node package manager)

### Install Dependencies

**Backend:**
```bash
cd src/web_ui/backend
pip install -r tests/requirements-test.txt
```

**Frontend:**
```bash
cd src/web_ui/frontend
npm install
```

### Run All Tests

**Windows:**
```cmd
cd src/web_ui
run_tests.bat
```

**Linux/macOS:**
```bash
cd src/web_ui
./run_tests.sh
```

**Python (Cross-platform):**
```bash
cd src/web_ui
python run_all_tests.py
```

## Detailed Usage

### Command Line Options

All test runners support the following options:

- `--verbose` / `-v`: Enable verbose output
- `--no-coverage`: Disable coverage reporting
- `--backend-only`: Run only backend Python tests
- `--frontend-only`: Run only frontend TypeScript tests
- `--sequential`: Run tests sequentially instead of parallel

### Examples

```bash
# Run all tests with verbose output
./run_tests.sh --verbose

# Run only backend tests
./run_tests.sh --backend-only

# Run only frontend tests without coverage
./run_tests.sh --frontend-only --no-coverage

# Run tests sequentially (useful for debugging)
./run_tests.sh --sequential --verbose
```

## Backend Testing Details

### Test Structure

```
src/web_ui/backend/tests/
├── conftest.py              # Test configuration and fixtures
├── test_auth.py             # Authentication and JWT tests
├── test_auth_routes.py      # Authentication endpoint tests
├── test_main_api.py         # Main API endpoint tests
├── test_services.py         # Application service tests
├── test_telegram_routes.py  # Telegram management tests
├── test_websocket_manager.py # WebSocket functionality tests
└── requirements-test.txt    # Test dependencies
```

### Running Backend Tests Only

```bash
cd src/web_ui/backend
python -m pytest tests/ -v --cov=src.api --cov-report=html
```

### Backend Test Categories

**Authentication Tests:**
```bash
python tests/test_runner.py auth
```

**API Endpoint Tests:**
```bash
python tests/test_runner.py api
```

**Service Layer Tests:**
```bash
python tests/test_runner.py services
```

**Telegram Tests:**
```bash
python tests/test_runner.py telegram
```

### Coverage Requirements

- **Minimum Coverage**: 80%
- **Target Coverage**: 85%+
- **Critical Paths**: 90%+ (authentication, API endpoints)

## Frontend Testing Details

### Test Structure

```
src/web_ui/frontend/tests/
├── components/
│   ├── Login.test.tsx       # Login component tests
│   ├── Dashboard.test.tsx   # Dashboard component tests
│   └── ...                  # Other component tests
├── stores/
│   └── authStore.test.ts    # Authentication store tests
├── utils/
│   └── test-utils.tsx       # Test utilities and helpers
├── setup.ts                 # Global test setup
└── vitest.config.ts         # Vitest configuration
```

### Running Frontend Tests Only

```bash
cd src/web_ui/frontend
npm run test
```

### Frontend Test Categories

**Component Tests:**
```bash
npm run test -- --run components/
```

**Store Tests:**
```bash
npm run test -- --run stores/
```

**Integration Tests:**
```bash
npm run test -- --run integration/
```

### Coverage Requirements

- **Minimum Coverage**: 80%
- **Target Coverage**: 85%+
- **Critical Components**: 90%+ (authentication, dashboard)

## Test Configuration

### Backend Configuration (pytest.ini)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

### Frontend Configuration (vitest.config.ts)

```typescript
export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json'],
      threshold: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        }
      }
    }
  }
});
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Web UI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: |
        cd src/web_ui
        pip install -r backend/tests/requirements-test.txt
        cd frontend && npm install
    
    - name: Run tests
      run: |
        cd src/web_ui
        python run_all_tests.py --junit test-results.xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: src/web_ui/test-results.xml
```

## Troubleshooting

### Common Issues

**Backend Tests:**

1. **Import Errors**: Ensure project root is in Python path
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Database Connection**: Tests use mocked database services
   ```python
   # Tests should not require actual database connection
   ```

3. **Missing Dependencies**: Install test requirements
   ```bash
   pip install -r backend/tests/requirements-test.txt
   ```

**Frontend Tests:**

1. **Node Modules**: Ensure dependencies are installed
   ```bash
   cd frontend && npm install
   ```

2. **Environment Variables**: Tests use mocked environment
   ```typescript
   // Tests should not require actual API endpoints
   ```

3. **Browser APIs**: Mocked in test setup
   ```typescript
   // setup.ts handles browser API mocking
   ```

### Performance Issues

**Slow Tests:**
- Use `--sequential` flag to avoid resource conflicts
- Check for network timeouts in mocked API calls
- Verify test isolation (no shared state)

**Memory Issues:**
- Run tests in smaller batches
- Clear test data between runs
- Check for memory leaks in component tests

### Debugging Tests

**Backend:**
```bash
# Run specific test with debugging
python -m pytest tests/test_auth.py::TestTokenManagement::test_create_access_token -v -s

# Run with pdb debugger
python -m pytest tests/test_auth.py --pdb
```

**Frontend:**
```bash
# Run specific test
npm run test -- Login.test.tsx

# Run with debugging
npm run test -- --inspect-brk Login.test.tsx
```

## Coverage Reports

### Viewing Coverage

**Backend HTML Report:**
```bash
open backend/htmlcov/index.html
```

**Frontend HTML Report:**
```bash
open frontend/coverage/index.html
```

### Coverage Thresholds

The test suite enforces minimum coverage thresholds:

- **Lines**: 80%
- **Functions**: 80%
- **Branches**: 80%
- **Statements**: 80%

Tests will fail if coverage falls below these thresholds.

## Best Practices

### Writing Tests

1. **Test Structure**: Follow AAA pattern (Arrange, Act, Assert)
2. **Test Isolation**: Each test should be independent
3. **Mocking**: Mock external dependencies and APIs
4. **Descriptive Names**: Use clear, descriptive test names
5. **Edge Cases**: Test both happy path and error conditions

### Test Maintenance

1. **Regular Updates**: Keep tests updated with code changes
2. **Refactoring**: Refactor tests when refactoring code
3. **Documentation**: Document complex test scenarios
4. **Performance**: Monitor test execution time
5. **Coverage**: Maintain high coverage without sacrificing quality

## Support

For issues with the test suite:

1. Check this documentation first
2. Review test logs for specific error messages
3. Verify all dependencies are installed correctly
4. Check that the development environment is properly configured
5. Consult the main project documentation for setup issues

## Test Results Summary

When tests complete successfully, you should see:

```
================================================================================
                              ALL TESTS PASSED!
================================================================================
Total Duration: 45.23s
Tests Passed: 2/2
Overall Status: ✓ PASSED

Detailed Results:
--------------------------------------------------------------------------------
Backend Tests: ✓ PASSED (23.45s) (Coverage: 87.3%)
Frontend Tests: ✓ PASSED (21.78s) (Coverage: 84.1%)

Coverage Summary:
----------------------------------------
Backend Tests: ✓ 87.3%
Frontend Tests: ✓ 84.1%
================================================================================
```

This indicates that all tests have passed and coverage requirements are met.