"""Tests for ``fmp backup``, ``fmp publish`` and ``fmp sync`` (spec §13, §14): pack, verify, install."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.builders.builder import read_current
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.increments.store import IncrementStore
from src.data.fmp_pipeline.transfer.commands import backup, publish, ready_publications, sync
from src.data.fmp_pipeline.transfer.packing import READY, pack_tree, read_manifest, unpack, verify_packed, write_manifest
from src.data.fmp_pipeline.validators.checksums import write_checksums
from src.data.fmp_pipeline.tests.vintage_fixtures import make_vintage


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("writer")
    make_vintage(root)
    IncrementStore(root).write_meta({"increment_seq": 3})
    return root


@pytest.fixture
def writer(template: Path, tmp_path: Path) -> Path:
    shutil.copytree(template, tmp_path / "writer")
    return tmp_path / "writer"


def test_pack_verify_unpack_round_trip(tmp_path: Path):
    tree = tmp_path / "tree"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / "a" / "b" / "x.bin").write_bytes(b"x" * 10)
    (tree / "c").mkdir()
    (tree / "c" / "y.bin").write_bytes(b"y")
    (tree / "top.txt").write_bytes(b"t")
    write_checksums(tree, tmp_path / "SUMS")
    out = tmp_path / "out"
    pack = pack_tree(tree, out, "t")
    write_manifest(out, "d", [pack])
    assert sorted(t.name for t in pack.tars) == ["t-_root.tar", "t-a.tar", "t-c.tar"] and pack.files == 3
    assert verify_packed(out, "t", tmp_path / "SUMS") == []
    assert unpack(out, "t", tmp_path / "back") == 3 and (tmp_path / "back" / "a" / "b" / "x.bin").read_bytes() == b"x" * 10
    (tree / "c" / "z.bin").write_bytes(b"z")  # the original grew: the pack lacks it
    write_checksums(tree, tmp_path / "SUMS2")
    assert verify_packed(out, "t", tmp_path / "SUMS2") == ["missing from the pack: c/z.bin"]
    with (out / "t-c.tar").open("ab") as fh:
        fh.write(b"junk")
    damaged = verify_packed(out, "t", tmp_path / "SUMS")
    assert damaged == ["tar checksum mismatch: t-c.tar", "missing from the pack: c/y.bin"]
    assert verify_packed(out, "nope", tmp_path / "SUMS") == ["MANIFEST.json has no nope pack"]
    assert verify_packed(tmp_path, "t", tmp_path / "SUMS")[0].endswith("MANIFEST.json missing")


def test_unpack_refuses_unsafe_member_paths(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "evil.txt").write_bytes(b"e")
    with tarfile.open(out / "t-x.tar", "w") as tar:
        tar.add(tmp_path / "evil.txt", arcname="../evil.txt")
    (out / "MANIFEST.json").write_text('{"packs": [{"tree": "t", "tars": [{"name": "t-x.tar"}]}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe member"):
        unpack(out, "t", tmp_path / "target")


def test_backup_is_verified_and_accepted_by_verify(writer: Path, tmp_path: Path):
    result = backup(writer, "v", tmp_path / "nas")
    assert result.errors == [] and result.files > 20
    manifest = read_manifest(tmp_path / "nas" / "v")
    assert [p["tree"] for p in manifest["packs"]] == ["raw", "vintage"]
    assert (tmp_path / "nas" / "v" / "checksums" / "ARCHIVE.sha256").exists()
    from src.data.fmp_pipeline.planner.config import PlanConfig
    from src.data.fmp_pipeline.validators.verify import FinalVerifier

    verifier = FinalVerifier(writer, PlanConfig(dump_id="v"), backup=tmp_path / "nas", run_tests=False)
    assert verifier._backup_errors("v") == []
    tar = next((tmp_path / "nas" / "v").glob("vintage-prices.tar"))
    tar.write_bytes(tar.read_bytes()[:-10])
    assert any("tar checksum mismatch" in e for e in verifier._backup_errors("v"))
    with pytest.raises(FileNotFoundError):
        backup(writer, "nope", tmp_path / "nas")


def test_publish_then_sync_installs_and_switches(writer: Path, tmp_path: Path):
    handoff = tmp_path / "R"
    result = publish(writer, "v", handoff, increments=True)
    assert result.errors == [] and (handoff / "publish" / "v" / READY).exists()
    assert read_manifest(handoff / "publish" / "v")["increment_seq"] == 3
    assert ready_publications(handoff) == ["v"] and ready_publications(tmp_path / "none") == []

    reader = tmp_path / "pi"
    installed = sync(reader, handoff, increments=True)
    assert installed.errors == [] and read_current(reader) == "v"
    assert FmpStore(reader).resolve("AAA") == FmpStore(writer).resolve("AAA")
    assert IncrementStore(reader).meta()["increment_seq"] == 3
    assert not (reader / "vintages" / ".v.syncing").exists()
    again = sync(reader, handoff, switch=False)  # re-installing replaces the vintage directory cleanly
    assert again.errors == [] and not (reader / "vintages" / ".v.old").exists()


def test_sync_rejects_damaged_or_missing_publications(writer: Path, tmp_path: Path):
    handoff = tmp_path / "R"
    with pytest.raises(FileNotFoundError, match="no READY publication"):
        sync(tmp_path / "pi", handoff)
    publish(writer, "v", handoff)
    with pytest.raises(FileNotFoundError):
        sync(tmp_path / "pi", handoff, dump_id="other")
    result = sync(tmp_path / "pi", handoff, increments=True)  # this publication has no increments
    assert result.errors == ["MANIFEST.json has no increments pack"] and read_current(tmp_path / "pi") is None
    tar = next((handoff / "publish" / "v").glob("vintage-reference.tar"))
    tar.write_bytes(b"broken")
    result = sync(tmp_path / "pi", handoff)
    assert "tar checksum mismatch: vintage-reference.tar" in result.errors and read_current(tmp_path / "pi") is None
    shutil.rmtree(writer / "increments")
    with pytest.raises(FileNotFoundError, match="increment layer"):
        publish(writer, "v", handoff, increments=True)


def test_cli_backup_publish_sync(writer: Path, tmp_path: Path):
    common = ["--data-root", str(writer), "--dump-id", "v"]
    assert cli.main(["backup", *common, "--to", str(tmp_path / "nas")]) == cli.EXIT_OK
    assert cli.main(["publish", *common, "--to", str(tmp_path / "R")]) == cli.EXIT_OK
    assert cli.main(["sync", "--from", str(tmp_path / "R"), "--data-root", str(tmp_path / "pi")]) == cli.EXIT_OK
    assert read_current(tmp_path / "pi") == "v"
    assert cli.main(["sync", "--from", str(tmp_path / "none"), "--data-root", str(tmp_path / "pi")]) == cli.EXIT_USAGE
    assert cli.main(["backup", "--data-root", str(writer), "--dump-id", "x", "--to", str(tmp_path)]) == cli.EXIT_USAGE
    tar = next((tmp_path / "R" / "publish" / "v").glob("vintage-events.tar"))
    tar.write_bytes(b"broken")
    assert cli.main(["sync", "--from", str(tmp_path / "R"), "--data-root", str(tmp_path / "pi2")]) == cli.EXIT_STOPPED
