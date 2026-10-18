"""Unit tests for the WP-2 building blocks: rate limiter, circuit breaker, parameter checks, Retry-After, fsutil."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data.fmp_pipeline import fsutil
from src.data.fmp_pipeline.client.circuit_breaker import CircuitBreaker
from src.data.fmp_pipeline.client.fmp_client import FmpClient, parse_retry_after, validate_params
from src.data.fmp_pipeline.client.rate_limiter import AsyncRateLimiter
from src.data.fmp_pipeline.errors import DumpLocked
from src.data.fmp_pipeline.fsutil import exclusive_lock, existing_parent, free_bytes


class FakeClock:
    """Manual monotonic clock whose sleep advances time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.asyncio
async def test_rate_limiter_spaces_grants_without_burst():
    clock = FakeClock()
    limiter = AsyncRateLimiter(600, clock=clock, sleep=clock.sleep)
    grants = []
    for _ in range(5):
        await limiter.acquire()
        grants.append(clock.now)
    assert grants == pytest.approx([1000.0, 1000.1, 1000.2, 1000.3, 1000.4])


@pytest.mark.asyncio
async def test_rate_limiter_never_exceeds_limit_with_concurrent_workers():
    clock = FakeClock()
    limiter = AsyncRateLimiter(60, clock=clock, sleep=clock.sleep)  # one per second

    granted: list[float] = []

    async def worker() -> None:
        for _ in range(10):
            await limiter.acquire()
            granted.append(clock.now)

    await asyncio.gather(*(worker() for _ in range(8)))
    granted.sort()
    assert len(granted) == 80
    # Any 60-second window holds at most 60 grants.
    for i, start in enumerate(granted):
        assert sum(1 for t in granted[i:] if t < start + 60) <= 60


@pytest.mark.asyncio
async def test_rate_limiter_idle_time_is_not_banked():
    clock = FakeClock()
    limiter = AsyncRateLimiter(60, clock=clock, sleep=clock.sleep)
    await limiter.acquire()
    clock.now += 100  # idle
    await limiter.acquire()
    await limiter.acquire()
    assert clock.sleeps == [pytest.approx(1.0)]


def test_rate_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        AsyncRateLimiter(0)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_pauses_and_resets():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, pause_s=60, clock=clock, sleep=clock.sleep)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.is_open
    breaker.record_failure()
    assert breaker.is_open and breaker.times_opened == 1 and breaker.consecutive_failures == 0
    await breaker.wait_if_open()
    assert clock.sleeps == [60] and not breaker.is_open
    await breaker.wait_if_open()
    assert clock.sleeps == [60]


def test_circuit_breaker_flags_repeated_all_429_openings():
    breaker = CircuitBreaker(threshold=2, stop_after_opens=3, clock=FakeClock())
    for _ in range(4):
        breaker.record_failure(rate_limited=True)
    assert breaker.rate_limited_opens == 2 and not breaker.rate_limit_exhausted
    breaker.record_success()  # one good response clears the run of openings
    assert breaker.rate_limited_opens == 0
    for _ in range(6):
        breaker.record_failure(rate_limited=True)
    assert breaker.rate_limit_exhausted


def test_circuit_breaker_ignores_openings_with_5xx_or_network_errors():
    breaker = CircuitBreaker(threshold=2, stop_after_opens=2, clock=FakeClock())
    for rate_limited in (True, True, True, False, True, True):
        breaker.record_failure(rate_limited=rate_limited)
    assert breaker.times_opened == 3 and breaker.rate_limited_opens == 1 and not breaker.rate_limit_exhausted
    for _ in range(10):
        breaker.record_failure()
    assert breaker.rate_limited_opens == 0 and not breaker.rate_limit_exhausted


def test_circuit_breaker_stop_can_be_disabled():
    breaker = CircuitBreaker(threshold=1, stop_after_opens=0, clock=FakeClock())
    for _ in range(10):
        breaker.record_failure(rate_limited=True)
    assert breaker.rate_limited_opens == 10 and not breaker.rate_limit_exhausted


@pytest.mark.parametrize("params", [
    {"from": "2025-01-01", "to": "2025-12-31"}, {"date": "2024-02-29"}, {"symbol": "AAPL"}, {},
])
def test_validate_params_accepts(params: dict):
    validate_params(params)


@pytest.mark.parametrize("params,match", [
    ({"from": "not-a-date"}, "YYYY-MM-DD"),
    ({"to": "2025-13-01"}, "YYYY-MM-DD"),
    ({"date": "2025-1-1"}, "YYYY-MM-DD"),
    ({"from": "2025-02-30"}, "YYYY-MM-DD"),
    ({"from": "2025-06-02", "to": "2025-06-01"}, "after"),
])
def test_validate_params_rejects(params: dict, match: str):
    with pytest.raises(ValueError, match=match):
        validate_params(params)


def test_parse_retry_after():
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("99999") == 300.0
    assert parse_retry_after("Wed, 14 Oct 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Wed, 14 Oct 2026 11:00:00 GMT", now=now) == 0.0
    assert parse_retry_after("soon") is None


def test_client_requires_key():
    with pytest.raises(ValueError):
        FmpClient("")


def test_atomic_write_bytes_replaces_and_leaves_no_temp(tmp_path: Path):
    target = tmp_path / "sub" / "file.bin"
    fsutil.atomic_write_bytes(target, b"one")
    fsutil.atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]


def test_replace_with_retry_survives_transient_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_bytes(b"x")
    real_replace = fsutil.os.replace
    failures = iter([True, True, False])
    delays: list[float] = []

    def flaky(a: Path, b: Path) -> None:
        if next(failures):
            raise PermissionError("locked by indexer")
        real_replace(a, b)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    fsutil.replace_with_retry(src, dst, sleep=delays.append)
    assert dst.read_bytes() == b"x" and delays == [0.05, 0.1]


def test_replace_with_retry_gives_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def locked(a: Path, b: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(fsutil.os, "replace", locked)
    with pytest.raises(PermissionError):
        fsutil.replace_with_retry(tmp_path / "a", tmp_path / "b", attempts=3, sleep=lambda s: None)


def test_atomic_write_cleans_temp_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def locked(a: Path, b: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(fsutil, "replace_with_retry", locked)
    with pytest.raises(PermissionError):
        fsutil.atomic_write_bytes(tmp_path / "f", b"data")
    assert list(tmp_path.iterdir()) == []


def test_exclusive_lock_rejects_a_second_holder_and_releases(tmp_path: Path):
    lock = tmp_path / "locks" / "d.lock"
    with exclusive_lock(lock):
        with pytest.raises(DumpLocked, match="held by another fmp process"):
            with exclusive_lock(lock):
                pass  # pragma: no cover - never entered
    with exclusive_lock(lock):  # released on exit, reusable
        assert lock.exists()


def test_free_bytes_walks_up_to_an_existing_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert free_bytes(tmp_path / "not" / "yet" / "there") == free_bytes(tmp_path)
    assert existing_parent(tmp_path / "a" / "b") == tmp_path

    def broken(_: Path) -> None:
        raise OSError("drive gone")

    monkeypatch.setattr(fsutil.shutil, "disk_usage", broken)
    assert free_bytes(tmp_path) is None
