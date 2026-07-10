"""
Tests for the psycopg2 numpy scalar adapters registered in database.py.

Regression test for the P20 ingest_eod failure: under numpy 2.x,
repr(np.float64(x)) is "np.float64(x)", which psycopg2's default
float-subclass adapter injected verbatim into SQL, causing Postgres
to fail with InvalidSchemaName: schema "np" does not exist.
"""

import pytest

np = pytest.importorskip("numpy")
psycopg2_ext = pytest.importorskip("psycopg2.extensions")

import src.data.db.core.database  # noqa: F401  (registers the adapters on import)


def _quoted(value) -> bytes:
    adapted = psycopg2_ext.adapt(value)
    return adapted.getquoted()


def test_float64_binds_as_plain_numeric_literal():
    assert _quoted(np.float64(313.390015)) == b"313.390015"


def test_float32_binds_as_plain_numeric_literal():
    assert float(_quoted(np.float32(1.5))) == 1.5


def test_float64_nan_binds_as_pg_nan():
    assert b"NaN" in _quoted(np.float64("nan"))


def test_int_scalars_bind_as_integer_literals():
    assert int(_quoted(np.int64(42))) == 42
    assert int(_quoted(np.int32(-7))) == -7
    assert int(_quoted(np.int16(0))) == 0


def test_bool_binds_as_boolean_literal():
    assert _quoted(np.bool_(True)) == b"true"
    assert _quoted(np.bool_(False)) == b"false"
