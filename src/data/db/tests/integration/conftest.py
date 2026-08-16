"""
Integration test fixtures - reuses repo and service test fixtures.

This conftest provides access to all test fixtures from parent conftest files,
ensuring integration tests use isolated test databases rather than production.
"""

# Import all fixtures from parent test directories. `pytest_plugins` is only
# legal in the rootdir's conftest.py as of pytest 8+ (this repo has no
# rootdir conftest.py, so that mechanism isn't available here); explicit
# imports register these as fixtures in this module instead. Named
# individually (not `import *`) because pytest.fixture names starting with
# "_" are private-by-convention and a wildcard import silently skips them,
# which would break the engine -> _test_db_url -> _db_admin_engine chain
# and drop the autouse _apply_migrations fixture. services.conftest already
# re-exports the repos.conftest fixtures the same way, so importing from it
# alone pulls in both layers.
from src.data.db.tests.services.conftest import (  # noqa: F401
    _apply_migrations,
    _db_admin_engine,
    _test_db_url,
    bots_repo,
    db_session,
    engine,
    jobs_repo,
    mock_database_service,
    notification_repo,
    repos_bundle,
    short_squeeze_repo,
    users_repo,
)
