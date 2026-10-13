"""
Tests for normalize_db_url.

SQLAlchemy 2.1 resolves a bare ``postgresql://`` URL to psycopg (v3), which is
not installed; normalize_db_url pins such URLs to psycopg2.
"""

import pytest
from sqlalchemy.engine import make_url

from src.data.db.core.db_url import normalize_db_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@host:5432/db", "postgresql+psycopg2://u:p@host:5432/db"),
        ("postgres://u:p@host/db", "postgresql+psycopg2://u:p@host/db"),
        ("postgresql://localhost/trading", "postgresql+psycopg2://localhost/trading"),
    ],
)
def test_normalize_db_url_pins_bare_postgres_to_psycopg2(url: str, expected: str) -> None:
    assert normalize_db_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg2://u:p@host/db",
        "postgresql+asyncpg://u:p@host/db",
        "sqlite:///:memory:",
        "sqlite:///C:/tmp/x.db",
        "",
    ],
)
def test_normalize_db_url_leaves_explicit_driver_and_other_dialects_unchanged(url: str) -> None:
    assert normalize_db_url(url) == url


def test_normalize_db_url_resolves_to_psycopg2_dbapi() -> None:
    """The normalised URL must load the psycopg2 dialect, not SQLAlchemy 2.1's psycopg default."""
    pytest.importorskip("psycopg2")
    dialect_cls = make_url(normalize_db_url("postgresql://u:p@host/db")).get_dialect()
    assert dialect_cls.driver == "psycopg2"
