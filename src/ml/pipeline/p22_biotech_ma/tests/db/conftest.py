"""
P22 real-Postgres test fixtures — isolated to this `tests/db/` subdirectory.

The repo-layer conftest's `_apply_migrations` fixture is session-scoped
`autouse=True`, which forces a live DB connection for every test collected
under the same conftest — this file therefore lives in its own `tests/db/`
subdirectory rather than `tests/` directly, so P22's non-DB unit tests
(clients, raw zone, model shapes) are never forced to connect to Postgres.

Requires ALEMBIC_DB_URL (or ETRADING_TEST_DB_URL) to point at a dedicated
test database — never production. See
src/data/db/tests/repos/conftest.py's module docstring for the safety rules
and setup instructions.
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-import (not `pytest_plugins`, which needs a rootdir conftest.py this
# repo doesn't have) — named individually so autouse/private fixtures aren't
# silently dropped by a wildcard import. Same pattern as
# src/data/db/tests/services/conftest.py.
from src.data.db.tests.repos.conftest import (  # noqa: F401
    _apply_migrations,
    _db_admin_engine,
    _test_db_url,
    db_session,
    engine,
)
