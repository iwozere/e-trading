# Web UI Backend Tests

Comprehensive unit test suite for the Trading Web UI backend.

## Test Structure

- `conftest.py` - Shared fixtures and test configuration
- `test_auth.py` - Authentication and authorization tests
- `test_auth_routes.py` - Authentication endpoint tests
- `test_main_api.py` - Main API endpoint tests
- `test_services.py` - Application service layer tests
- `test_telegram_routes.py` - Telegram management API tests
- `test_runner.py` - Test execution script

## Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python test_runner.py

# Run specific test categories
python test_runner.py auth      # Authentication tests
python test_runner.py api       # API endpoint tests
python test_runner.py services  # Service layer tests
python test_runner.py telegram  # Telegram tests

# Run with pytest directly
pytest -v --cov=src.api
```

## Coverage Target

Minimum 80% code coverage for all components.