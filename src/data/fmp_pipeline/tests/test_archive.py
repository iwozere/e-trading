"""Tests for the append-only raw archive and its reader (spec §6.6, plan WP-5)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pytest

from src.data.fmp_pipeline.archive.archive import GB, RawArchive, build_record
from src.data.fmp_pipeline.archive.reader import (
    ArchiveCorrupt,
    good_length,
    iter_members,
    iter_records,
    latest_successful,
    request_key,
)
from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.errors import ArchiveUnavailable, InsufficientDiskSpace
from src.data.fmp_pipeline.fsutil import ensure_free_space
from src.data.fmp_pipeline.manifest.manifest import Task, task_id

PRICE = "historical-price-eod/non-split-adjusted"


def _task(symbol: Optional[str] = "AAPL", endpoint: str = PRICE, **params: Any) -> Task:
    params = params or {"from": "2025-01-01", "to": "2025-03-31"}
    return Task(task_id("d", endpoint, symbol, params), "d", endpoint, symbol, params, 10, None, 1)


def _result(body: bytes = b'[{"date":"2025-01-02","adjClose":1.5}]', status: int = 200, classification: str = "ok",
            fetched_at: str = "2026-10-14T10:00:00+00:00") -> FetchResult:
    return FetchResult(PRICE, {}, status, classification, body, len(body) // 3, len(body), 1, 1, fetched_at)


@pytest.fixture
def archive(tmp_path: Path) -> RawArchive:
    return RawArchive(tmp_path / "raw", "2026-10")


def test_layout_and_line_format(archive: RawArchive):
    task, result = _task(), _result()
    sha = archive(task, result)
    path = archive.path_for(PRICE, "AAPL")
    assert path == archive.dump_dir / "historical-price-eod" / "non-split-adjusted" / "AAPL.jsonl.gz"
    assert sha == hashlib.sha256(result.body).hexdigest()
    [record] = list(iter_records(path))
    assert record == {
        "task_id": task.task_id,
        "request": {"endpoint": PRICE, "params": {"symbol": "AAPL", "from": "2025-01-01", "to": "2025-03-31"}},
        "fetched_at": "2026-10-14T10:00:00+00:00", "http_status": 200, "classification": "ok", "rows": 1,
        "sha256": sha, "body": [{"date": "2025-01-02", "adjClose": 1.5}],
    }


def test_appends_are_concatenated_gzip_members_readable_by_gzip(archive: RawArchive):
    for i in range(5):
        archive(_task(**{"from": f"202{i}-01-01"}), _result(fetched_at=f"2026-10-14T10:00:0{i}+00:00"))
    path = archive.path_for(PRICE, "AAPL")
    members = list(iter_members(path))
    assert len(members) == 5 and members[-1].end == path.stat().st_size
    assert [m.start for m in members[1:]] == [m.end for m in members[:-1]]
    lines = gzip.decompress(path.read_bytes()).splitlines()  # the whole file is valid gzip
    assert [json.loads(x)["request"]["params"]["from"] for x in lines] == [f"202{i}-01-01" for i in range(5)]
    assert archive.appended == 5 and archive.bytes_written == path.stat().st_size


def test_members_are_deterministic(tmp_path: Path):
    a, b = RawArchive(tmp_path / "a", "d"), RawArchive(tmp_path / "b", "d")
    a(_task(), _result())
    b(_task(), _result())
    assert a.path_for(PRICE, "AAPL").read_bytes() == b.path_for(PRICE, "AAPL").read_bytes()


def test_never_overwrites_a_refetch_adds_a_line(archive: RawArchive):
    task = _task()
    archive(task, _result(b'[{"v":1}]', fetched_at="2026-10-14T10:00:00+00:00"))
    archive(task, _result(b'[{"v":2}]', fetched_at="2026-10-15T10:00:00+00:00"))
    records = list(iter_records(archive.path_for(PRICE, "AAPL")))
    assert [r["body"] for r in records] == [[{"v": 1}], [{"v": 2}]]


def test_non_json_and_empty_bodies(archive: RawArchive):
    archive(_task("R1"), _result(b"Restricted Endpoint: upgrade", 402, "plan_restricted"))
    archive(_task("NET"), _result(b"", 0, "network_error"))
    [restricted] = iter_records(archive.path_for(PRICE, "R1"))
    assert restricted["body"] is None and restricted["body_text"] == "Restricted Endpoint: upgrade"
    [net] = iter_records(archive.path_for(PRICE, "NET"))
    assert net["body"] is None and "body_text" not in net and net["http_status"] == 0


def test_symbolless_and_awkward_symbols(archive: RawArchive):
    archive(_task(None, "delisted-companies", page=0, limit=100), _result())
    archive(_task("CON"), _result())
    archive(_task("A/B"), _result())
    files = sorted(p.relative_to(archive.dump_dir).as_posix() for p in archive.dump_dir.rglob("*.jsonl.gz"))
    assert files == ["delisted-companies/_.jsonl.gz", f"{PRICE}/A%2FB.jsonl.gz", f"{PRICE}/_CON.jsonl.gz"]
    [record] = iter_records(archive.path_for("delisted-companies", None))
    assert record["request"]["params"] == {"page": 0, "limit": 100}


def test_api_key_never_written(archive: RawArchive, tmp_path: Path):
    task = Task("t", "d", PRICE, "AAPL", {"apikey": "SECRET", "ApiKey": "SECRET"}, 1, None, 1)
    archive(task, _result())
    [record] = iter_records(archive.path_for(PRICE, "AAPL"))
    assert record["request"]["params"] == {"symbol": "AAPL"}
    raw = b"".join(gzip.decompress(p.read_bytes()) for p in tmp_path.rglob("*.gz"))
    assert b"SECRET" not in raw


def test_build_record_preserves_unicode_and_numbers():
    record = build_record(_task(), _result('[{"name":"Nestlé","v":12345678901234567890,"f":0.1}]'.encode()))
    assert record["body"] == [{"name": "Nestlé", "v": 12345678901234567890, "f": 0.1}]


def test_bad_dump_id_and_endpoint_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        RawArchive(tmp_path, "../x")
    with pytest.raises(ValueError):
        RawArchive(tmp_path, "d").path_for("../etc", "A")


# --- torn tails -------------------------------------------------------------------------------------------------


def _write_two(archive: RawArchive) -> tuple[Path, int]:
    archive(_task(**{"from": "2020-01-01"}), _result())
    archive(_task(**{"from": "2021-01-01"}), _result())
    path = archive.path_for(PRICE, "AAPL")
    return path, path.stat().st_size


@pytest.mark.parametrize("cut", [1, 10, 25])
def test_reader_detects_a_partial_last_member(archive: RawArchive, cut: int):
    path, size = _write_two(archive)
    member = gzip.compress(b'{"x":1}\n', mtime=0)
    with path.open("ab") as fh:
        fh.write(member[:cut])  # a write cut off by a kill
    with pytest.raises(ArchiveCorrupt) as err:
        list(iter_records(path))
    assert err.value.good_end == size
    assert len(list(iter_records(path, strict=False))) == 2
    assert good_length(path) == size


def test_reader_detects_garbage_and_bad_crc(archive: RawArchive):
    path, size = _write_two(archive)
    with path.open("ab") as fh:
        fh.write(b"\x00" * 7)
    assert good_length(path) == size
    data = bytearray(path.read_bytes()[:size])
    data[-6] ^= 0xFF  # corrupt the CRC32 of the last member
    path.write_bytes(bytes(data))
    first_end = next(iter_members(path)).end
    assert good_length(path) == first_end


def test_reader_handles_members_spanning_read_chunks(archive: RawArchive):
    body = json.dumps([{"date": f"2020-01-{i % 28 + 1:02d}", "v": i} for i in range(3000)]).encode()
    for i in range(3):
        archive(_task(**{"from": f"201{i}-01-01"}), _result(body))
    path = archive.path_for(PRICE, "AAPL")
    assert [m.end for m in iter_members(path, chunk_size=17)] == [m.end for m in iter_members(path)]
    assert len(list(iter_records(path))) == 3


def test_first_append_repairs_a_torn_tail(tmp_path: Path):
    path, size = _write_two(RawArchive(tmp_path, "d"))
    with path.open("ab") as fh:
        fh.write(b"\x1f\x8b\x08\x00 torn")
    fresh = RawArchive(tmp_path, "d")  # a new process after the crash
    fresh(_task(**{"from": "2022-01-01"}), _result())
    assert fresh.repaired == [path]
    records = list(iter_records(path))  # strict: the file is clean again
    assert [r["request"]["params"]["from"] for r in records] == ["2020-01-01", "2021-01-01", "2022-01-01"]
    assert next(reversed(list(iter_members(path)))).start == size


def test_known_length_skips_rescanning(archive: RawArchive, monkeypatch: pytest.MonkeyPatch):
    import src.data.fmp_pipeline.archive.archive as archive_module

    calls: list[Path] = []

    def counting_good_length(path: Path) -> int:
        calls.append(path)
        return path.stat().st_size

    monkeypatch.setattr(archive_module, "good_length", counting_good_length)
    path, _ = _write_two(archive)
    archive(_task(**{"from": "2022-01-01"}), _result())
    assert calls == []  # every file was created by this process
    with path.open("ab") as fh:
        fh.write(b"external change")
    archive(_task(**{"from": "2023-01-01"}), _result())
    assert calls == [path]


# --- unavailability ---------------------------------------------------------------------------------------------


def test_missing_root_raises_archive_unavailable(tmp_path: Path):
    archive = RawArchive(tmp_path / "raw", "d")
    (tmp_path / "raw" / "d").rmdir()
    (tmp_path / "raw").rmdir()  # the drive went away
    with pytest.raises(ArchiveUnavailable):
        archive(_task(), _result())
    assert not (tmp_path / "raw").exists()  # nothing recreated behind the owner's back


def test_low_free_space_raises_and_is_checked_periodically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.data.fmp_pipeline.archive.archive as archive_module

    free = [10 * GB]
    checks: list[int] = []

    class _Usage:
        def __init__(self) -> None:
            checks.append(1)
            self.free = free[0]

    monkeypatch.setattr(archive_module.shutil, "disk_usage", lambda _: _Usage())
    now = [0.0]
    archive = RawArchive(tmp_path, "d", min_free_bytes=GB, space_check_s=30, clock=lambda: now[0])
    archive(_task("A"), _result())
    free[0] = GB // 2
    archive(_task("B"), _result())  # within 30 s: not re-checked
    assert len(checks) == 1
    now[0] = 31.0
    with pytest.raises(ArchiveUnavailable, match="free"):
        archive(_task("C"), _result())


def test_write_error_rolls_back_and_raises(archive: RawArchive, monkeypatch: pytest.MonkeyPatch):
    path, size = _write_two(archive)
    import src.data.fmp_pipeline.archive.archive as archive_module

    def failing_fsync(_: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(archive_module.os, "fsync", failing_fsync)
    with pytest.raises(ArchiveUnavailable, match="No space"):
        archive(_task(**{"from": "2022-01-01"}), _result())
    assert path.stat().st_size == size  # the partial member was rolled back
    monkeypatch.undo()
    archive(_task(**{"from": "2022-01-01"}), _result())
    assert len(list(iter_records(path))) == 3


# --- newest successful response ---------------------------------------------------------------------------------


def test_latest_successful_picks_newest_ok_line(archive: RawArchive):
    task = _task()
    archive(task, _result(b'[{"v":1}]', fetched_at="2026-10-14T10:00:00+00:00"))
    archive(task, _result(b'[{"v":2}]', fetched_at="2026-10-15T10:00:00+00:00"))
    archive(task, _result(b"oops", 503, "server_error", fetched_at="2026-10-16T10:00:00+00:00"))
    archive(task, _result(b'[{"v":3}]', fetched_at="2026-10-13T10:00:00+00:00"))  # older, appended later
    other = _task(**{"from": "2024-01-01"})
    archive(other, _result(b"[]", classification="ok_empty"))
    latest = latest_successful(iter_records(archive.path_for(PRICE, "AAPL")))
    assert len(latest) == 2
    assert latest[request_key(build_record(task, _result()))]["body"] == [{"v": 2}]
    assert latest[request_key(build_record(other, _result()))]["classification"] == "ok_empty"


def test_request_key_ignores_param_order():
    a = build_record(_task(**{"from": "2020-01-01", "to": "2020-12-31"}), _result())
    b = build_record(_task(**{"to": "2020-12-31", "from": "2020-01-01"}), _result())
    assert request_key(a) == request_key(b)


# --- free disk check --------------------------------------------------------------------------------------------


def test_ensure_free_space(tmp_path: Path):
    assert ensure_free_space(tmp_path / "not" / "yet" / "created", 1) > 0
    with pytest.raises(InsufficientDiskSpace, match="margin"):
        ensure_free_space(tmp_path, 10**18)
