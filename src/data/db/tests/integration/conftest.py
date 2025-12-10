"""
Integration test fixtures - reuses repo and service test fixtures.

This conftest provides access to all test fixtures from parent conftest files,
ensuring integration tests use isolated test databases rather than production.
"""

# Import all fixtures from parent test directories
pytest_plugins = [
    "src.data.db.tests.repos.conftest",
    "src.data.db.tests.services.conftest",
]
