"""Tests for jobs/run_process_events_ingest.py's date-range backfill. No live DB or network —
EDGAR, the DB service and run logging are patched."""

import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EftsUnavailableError
from src.ml.pipeline.p22_biotech_ma.jobs import run_process_events_ingest as job

_MODULE = "src.ml.pipeline.p22_biotech_ma.jobs.run_process_events_ingest"


def test_business_days_skips_weekends():
    # 2026-10-30 is a Friday; 10-31/11-01 the weekend.
    assert job.business_days(date(2026, 10, 30), date(2026, 11, 3)) == [
        date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3),
    ]


def test_business_days_rejects_reversed_range():
    with pytest.raises(ValueError):
        job.business_days(date(2026, 11, 3), date(2026, 11, 2))


def _db_service() -> MagicMock:
    @contextmanager
    def uow():
        yield MagicMock()

    service = MagicMock()
    service.uow.side_effect = uow
    return service


def test_backfill_scans_each_business_day_and_records_failed_days():
    summaries = {
        date(2026, 10, 30): {"filings_scanned": 3, "candidates_written": 1, "rejection_breakdown": {"item_not_relevant": 5}},
        date(2026, 11, 3): {"filings_scanned": 2, "candidates_written": 0, "rejection_breakdown": {"item_not_relevant": 4}},
    }

    def fake_scan(repo, edgar, as_of_date=None):
        del repo, edgar
        assert as_of_date is not None
        if as_of_date == date(2026, 11, 2):
            raise EftsUnavailableError("down")
        return summaries[as_of_date]

    with patch(f"{_MODULE}.setup_run_logging"), patch(f"{_MODULE}.EdgarDownloader"), \
            patch(f"{_MODULE}.DatabaseService", return_value=_db_service()), \
            patch(f"{_MODULE}.run_process_events", side_effect=fake_scan) as scan:
        result = job.run(date(2026, 10, 30), date(2026, 11, 3))

    assert [c.kwargs["as_of_date"] for c in scan.call_args_list] == [date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)]
    assert result == {
        "days_scanned": 2,
        "days_failed": ["2026-11-02"],
        "filings_scanned": 5,
        "candidates_written": 1,
        "rejection_breakdown": {"item_not_relevant": 9},
    }


def test_no_dates_runs_the_single_daily_scan():
    daily = {"filings_scanned": 1, "candidates_written": 0, "rejection_breakdown": {}}
    with patch(f"{_MODULE}.setup_run_logging"), patch(f"{_MODULE}.EdgarDownloader"), \
            patch(f"{_MODULE}.DatabaseService", return_value=_db_service()), \
            patch(f"{_MODULE}.run_process_events", return_value=daily) as scan:
        assert job.run() == daily
    scan.assert_called_once()
    assert "as_of_date" not in scan.call_args.kwargs
