"""
Tests for `src.data.pipeline.raw_zone` — ported from P22's original
`src/ml/pipeline/p22_biotech_ma/tests/test_raw_zone.py` (which now exercises
the backward-compatible shim at that path). Behavior must be identical after
the move; these tests exercise the promoted, pipeline-agnostic module
directly.
"""

from __future__ import annotations

from datetime import date

from src.data.pipeline import raw_zone


def test_write_creates_gzipped_json_and_manifest(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3]}
    result = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)

    assert result.was_new is True
    assert result.path.exists()
    assert result.path.name == f"{result.content_hash}.json.gz"

    manifest_path = result.path.with_suffix("").with_suffix(".manifest.json")
    assert manifest_path.exists()


def test_write_is_idempotent_by_content_hash(tmp_path):
    payload = {"a": 1}
    first = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)
    second = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)

    assert first.content_hash == second.content_hash
    assert first.was_new is True
    assert second.was_new is False


def test_different_payloads_get_different_hashes(tmp_path):
    r1 = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 1}, root=tmp_path)
    r2 = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 2}, root=tmp_path)
    assert r1.content_hash != r2.content_hash


def test_read_round_trips_payload(tmp_path):
    payload = {"nested": {"x": 1}, "list": [1, 2, 3]}
    result = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=payload, root=tmp_path)
    assert raw_zone.read(result.path) == payload


def test_write_never_overwrites_existing_file(tmp_path):
    result = raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 1}, root=tmp_path)
    original_mtime = result.path.stat().st_mtime
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload={"a": 1}, root=tmp_path)
    assert result.path.stat().st_mtime == original_mtime


def test_read_latest_partition_returns_most_recent_date(tmp_path):
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=["yesterday"], root=tmp_path)
    raw_zone.write(source="test_source", entity="E2", as_of_date=date(2024, 3, 2), payload=["today"], root=tmp_path)

    payloads = raw_zone.read_latest_partition("test_source", root=tmp_path)
    assert payloads == [["today"]]


def test_read_latest_partition_missing_source_returns_empty(tmp_path):
    assert raw_zone.read_latest_partition("no_such_source", root=tmp_path) == []


def test_read_latest_partition_with_manifest_pairs_payload_and_manifest(tmp_path):
    write_result = raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 3, 1), payload={"x": 1}, root=tmp_path)
    pairs = raw_zone.read_latest_partition_with_manifest("test_source", root=tmp_path)

    assert len(pairs) == 1
    payload, manifest = pairs[0]
    assert payload == {"x": 1}
    assert manifest["entity"] == "MRNA"
    assert manifest["content_hash"] == write_result.content_hash
    assert "known_from" in manifest


def test_read_latest_partition_with_manifest_missing_source_returns_empty(tmp_path):
    assert raw_zone.read_latest_partition_with_manifest("no_such_source", root=tmp_path) == []


def test_has_any_landed_true_for_known_entity(tmp_path):
    raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 3, 1), payload={"x": 1}, root=tmp_path)
    assert raw_zone.has_any_landed("test_source", "MRNA", root=tmp_path) is True


def test_has_any_landed_false_for_unknown_entity(tmp_path):
    raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 3, 1), payload={"x": 1}, root=tmp_path)
    assert raw_zone.has_any_landed("test_source", "OTHER", root=tmp_path) is False


def test_has_any_landed_false_for_unknown_source(tmp_path):
    assert raw_zone.has_any_landed("no_such_source", "MRNA", root=tmp_path) is False


def test_has_any_landed_checks_all_date_partitions(tmp_path):
    raw_zone.write(source="test_source", entity="MRNA", as_of_date=date(2024, 1, 1), payload={"x": 1}, root=tmp_path)
    raw_zone.write(source="test_source", entity="OTHER", as_of_date=date(2024, 6, 1), payload={"y": 2}, root=tmp_path)
    assert raw_zone.has_any_landed("test_source", "MRNA", root=tmp_path) is True


def test_read_partition_before_excludes_same_or_later_date(tmp_path):
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=["yesterday"], root=tmp_path)
    raw_zone.write(source="test_source", entity="E2", as_of_date=date(2024, 3, 2), payload=["today-so-far"], root=tmp_path)

    payloads = raw_zone.read_partition_before("test_source", date(2024, 3, 2), root=tmp_path)
    assert payloads == [["yesterday"]]


def test_read_partition_before_picks_most_recent_prior_partition(tmp_path):
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 1, 1), payload=["oldest"], root=tmp_path)
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 2, 1), payload=["newer"], root=tmp_path)

    payloads = raw_zone.read_partition_before("test_source", date(2024, 3, 1), root=tmp_path)
    assert payloads == [["newer"]]


def test_read_partition_before_missing_source_returns_empty(tmp_path):
    assert raw_zone.read_partition_before("no_such_source", date(2024, 3, 1), root=tmp_path) == []


def test_read_partition_before_no_prior_partition_returns_empty(tmp_path):
    raw_zone.write(source="test_source", entity="E1", as_of_date=date(2024, 3, 1), payload=["only"], root=tmp_path)
    assert raw_zone.read_partition_before("test_source", date(2024, 3, 1), root=tmp_path) == []


def test_default_root_omitted_falls_back_to_data_cache_dir():
    assert raw_zone.DEFAULT_RAW_ZONE_ROOT.name == "raw"
