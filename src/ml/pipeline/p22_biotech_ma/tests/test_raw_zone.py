"""Tests for ingest/raw_zone.py — dedup, partitioning, immutability."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone


def test_write_creates_partitioned_gzip_file(tmp_path):
    result = raw_zone.write(
        source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 1}, root=tmp_path
    )
    assert result.was_new is True
    assert result.path.exists()
    assert result.path.parent == tmp_path / "test_source" / "2024-03-01"
    assert result.path.name == f"{result.content_hash}.json.gz"


def test_write_is_idempotent_for_identical_payload(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3]}
    first = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)
    second = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)

    assert first.was_new is True
    assert second.was_new is False
    assert first.content_hash == second.content_hash
    assert first.path == second.path


def test_different_payloads_get_different_hashes(tmp_path):
    r1 = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 1}, root=tmp_path)
    r2 = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 2}, root=tmp_path)
    assert r1.content_hash != r2.content_hash
    assert r1.path != r2.path


def test_read_roundtrip(tmp_path):
    payload = {"nested": {"x": [1, 2, 3]}, "s": "hello"}
    result = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)
    assert raw_zone.read(result.path) == payload


def test_manifest_records_known_from(tmp_path):
    result = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 1}, root=tmp_path)
    manifest_path = result.path.parent / f"{result.content_hash}.manifest.json"
    assert manifest_path.exists()
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "test_source"
    assert manifest["entity"] == "E1"
    assert manifest["as_of_date"] == "2024-03-01"


def test_read_latest_partition_returns_only_most_recent_date(tmp_path):
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=["old"], root=tmp_path)
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 2), payload=["new1"], root=tmp_path)
    raw_zone.write(source="test_source", entity="E2", as_of_date=date(2024, 3, 2), payload=["new2"], root=tmp_path)

    payloads = raw_zone.read_latest_partition("test_source", root=tmp_path)
    assert sorted(payloads, key=str) == [["new1"], ["new2"]]


def test_read_latest_partition_missing_source_returns_empty(tmp_path):
    assert raw_zone.read_latest_partition("no_such_source", root=tmp_path) == []


def test_read_latest_partition_with_manifest_pairs_payload_and_manifest(tmp_path):
    write_result = raw_zone.write(
        source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=["a", "b"], root=tmp_path
    )

    pairs = raw_zone.read_latest_partition_with_manifest("test_source", root=tmp_path)

    assert len(pairs) == 1
    payload, manifest = pairs[0]
    assert payload == ["a", "b"]
    assert manifest["entity"] == "E1"
    assert manifest["known_from"] == write_result.known_from.isoformat()


def test_read_latest_partition_with_manifest_missing_source_returns_empty(tmp_path):
    assert raw_zone.read_latest_partition_with_manifest("no_such_source", root=tmp_path) == []


def test_has_any_landed_true_after_write(tmp_path):
    raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 3, 1), payload={"x": 1}, root=tmp_path)
    assert raw_zone.has_any_landed("test_source", "MRNA", root=tmp_path) is True


def test_has_any_landed_false_for_unlanded_entity(tmp_path):
    raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 3, 1), payload={"x": 1}, root=tmp_path)
    assert raw_zone.has_any_landed("test_source", "OTHER", root=tmp_path) is False


def test_has_any_landed_false_for_missing_source(tmp_path):
    assert raw_zone.has_any_landed("no_such_source", "MRNA", root=tmp_path) is False


def test_has_any_landed_true_across_older_date_partition_not_just_latest(tmp_path):
    """Unlike read_latest_partition, this must find a landing under ANY date partition, not just
    the most recent — a resumable job needs to know "was this ever landed", not "was it landed today"."""
    raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 1, 1), payload={"x": 1}, root=tmp_path)
    raw_zone.write(source="test_source", entity="OTHER", as_of_date=date(2024, 6, 1), payload={"y": 2}, root=tmp_path)
    assert raw_zone.has_any_landed("test_source", "MRNA", root=tmp_path) is True
