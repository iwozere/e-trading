"""
Database URL normalisation.

SQLAlchemy 2.1 changed the default PostgreSQL DBAPI for a bare ``postgresql://``
URL from psycopg2 to psycopg (v3). This project depends on psycopg2 only, so a
URL without an explicit driver would fail at engine creation with
``ModuleNotFoundError: No module named 'psycopg'``. Pinning the driver here keeps
behaviour identical on SQLAlchemy 2.0 and 2.1.
"""

_PSYCOPG2_SCHEME = "postgresql+psycopg2://"
_BARE_SCHEMES = ("postgres://", "postgresql://")


def normalize_db_url(url: str) -> str:
    """
    Pin bare PostgreSQL URLs to the psycopg2 driver.

    ``postgres://`` (Heroku-style) and ``postgresql://`` become
    ``postgresql+psycopg2://``. URLs that already name a driver
    (e.g. ``postgresql+psycopg2://``, ``postgresql+asyncpg://``) and
    non-PostgreSQL URLs (e.g. ``sqlite://``) are returned unchanged.

    Args:
        url: A SQLAlchemy database URL.

    Returns:
        The URL with an explicit psycopg2 driver if it was a bare PostgreSQL URL.
    """
    for scheme in _BARE_SCHEMES:
        if url.startswith(scheme):
            return _PSYCOPG2_SCHEME + url[len(scheme):]
    return url
