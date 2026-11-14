"""Command-line tests on synthetic raw files: build, cross-check, deadline and purge without network or FMP data."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.sharadar_pipeline import cli, config
from src.data.sharadar_pipeline.manifest import Manifest
from src.data.sharadar_pipeline.tests.helpers import write_raw


def _run(root: Path, *args: str) -> int:
    return cli.main(["--data-root", str(root), *args])


def test_build_crosscheck_and_purge_through_the_cli(tmp_path: Path):
    root = write_raw(tmp_path / config.DATA_DIR_NAME)
    assert _run(root, "status") == 0
    assert _run(root, "build", "--no-fmp") == 0
    assert (root / config.STORE_DIR / config.SECURITIES).exists()
    assert _run(root, "crosscheck", "--no-fmp") == 0
    assert (root / config.REPORTS_DIR / "crosscheck_summary.json").exists()
    assert _run(root, "purge") == 0 and root.exists()  # dry run
    report = tmp_path / "purge.json"
    assert _run(root, "purge", "--confirm", "--report", str(report)) == 0
    assert not root.exists() and report.exists()


def test_set_end_records_the_deadline(tmp_path: Path):
    root = tmp_path / config.DATA_DIR_NAME
    manifest = Manifest(mode="paid", subscription_start="2026-12-01")
    manifest.save(root)
    assert _run(root, "set-end", "2026-12-15") == 0
    assert str(Manifest.load(root).deletion_deadline()) == "2027-01-14"


def test_paid_commands_need_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "api_key", lambda: None)
    with pytest.raises(SystemExit):
        _run(tmp_path / config.DATA_DIR_NAME, "bulk-status")
    with pytest.raises(SystemExit):
        _run(tmp_path / config.DATA_DIR_NAME, "download")
