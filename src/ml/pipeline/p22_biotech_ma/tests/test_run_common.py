"""Tests for jobs/run_common.record_fetch_failures (spec §7.2) — fake unit of work, no DB."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy.exc import OperationalError

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import FetchFailureRecord
from src.ml.pipeline.p22_biotech_ma.jobs import run_common

_FAILURE = FetchFailureRecord("openfda_drugsfda", "Example Inc", "https://api.fda.gov", "status 403")


def test_record_fetch_failures_writes_each_row(monkeypatch):
    repo = MagicMock()

    @contextmanager
    def uow():
        yield SimpleNamespace(p22=repo)

    monkeypatch.setattr(run_common, "DatabaseService", lambda: SimpleNamespace(uow=uow))
    assert run_common.record_fetch_failures([_FAILURE, _FAILURE]) == 2
    repo.log_fetch_failure.assert_called_with(
        source="openfda_drugsfda", entity="Example Inc", url="https://api.fda.gov", error_message="status 403"
    )


def test_record_fetch_failures_skips_db_when_empty(monkeypatch):
    monkeypatch.setattr(run_common, "DatabaseService", MagicMock(side_effect=AssertionError("no DB expected")))
    assert run_common.record_fetch_failures([]) == 0


def test_record_fetch_failures_survives_db_error(monkeypatch):
    @contextmanager
    def uow():
        raise OperationalError("insert", {}, Exception("down"))
        yield  # pragma: no cover

    monkeypatch.setattr(run_common, "DatabaseService", lambda: SimpleNamespace(uow=uow))
    assert run_common.record_fetch_failures([_FAILURE]) == 0
