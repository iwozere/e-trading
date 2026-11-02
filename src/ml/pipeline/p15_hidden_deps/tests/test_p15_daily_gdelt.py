"""Tests for the P15 daily bundle's GDELT gap-healing window."""

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p15_hidden_deps import p15_daily


class _RecordingGdelt:
    """Stand-in for GdeltDownloader that records the requested ranges."""

    calls: List[Tuple[str, datetime, datetime]] = []

    def download_gkg_range(self, start: datetime, end: datetime) -> Dict[str, Any]:
        _RecordingGdelt.calls.append(("gkg", start, end))
        return {"total": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    def download_events_range(self, start: datetime, end: datetime) -> Dict[str, Any]:
        _RecordingGdelt.calls.append(("events", start, end))
        return {"total": 0, "downloaded": 0, "skipped": 0, "errors": 0}


def test_rescan_window_ignores_watermark_and_caps_length() -> None:
    start, end = p15_daily._rescan_window(date(2015, 2, 18), date(2026, 11, 2), cap_days=60)
    assert end == date(2026, 11, 2)
    assert start == date(2026, 9, 4)
    assert (end - start).days == 59


def test_rescan_window_respects_source_start() -> None:
    start, end = p15_daily._rescan_window(date(2026, 10, 30), date(2026, 11, 2), cap_days=60)
    assert (start, end) == (date(2026, 10, 30), date(2026, 11, 2))


def test_rescan_window_never_starts_after_yesterday() -> None:
    start, end = p15_daily._rescan_window(date(2027, 1, 1), date(2026, 11, 2))
    assert start == end == date(2026, 11, 2)


def test_gdelt_jobs_rescan_interior_hole(tmp_path: Path) -> None:
    # Reproduces 2026-10-31: 10-30 never landed, but P20 then wrote 10-31 and 11-01.
    gkg_dir = tmp_path / "gdelt" / "gkg"
    events_dir = tmp_path / "gdelt" / "events"
    gkg_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    for day in ("20261029", "20261031", "20261101"):
        (gkg_dir / f"{day}.gkg.csv.gz").write_bytes(b"")
        (events_dir / f"{day}.events.csv.gz").write_bytes(b"")

    _RecordingGdelt.calls = []
    yesterday = date(2026, 11, 2)
    with patch.object(p15_daily, "_cache_root", str(tmp_path)), patch.object(p15_daily, "GdeltDownloader", _RecordingGdelt):
        p15_daily._job_gdelt_gkg(yesterday)
        p15_daily._job_gdelt_events(yesterday)

    assert len(_RecordingGdelt.calls) == 2
    for _kind, start, end in _RecordingGdelt.calls:
        assert start.date() <= date(2026, 10, 30), "hole before the watermark must be re-requested"
        assert end.date() == yesterday
