"""Tests for ingest/acquirer_config.py. No live DB — repo is a MagicMock."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import load_acquirers, upsert_acquirer_roster


def _write_yaml(tmp_path, body: str) -> Path:
    path = tmp_path / "acquirers.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_acquirers_parses_real_config_file():
    """Round trip against the actual repo config, not just a synthetic fixture."""
    from src.ml.pipeline.p22_biotech_ma.config import ACQUIRERS_YAML

    acquirers = load_acquirers(ACQUIRERS_YAML)

    assert len(acquirers) >= 20  # spec §2.0.4: "~25 acquirers"
    pfizer = next(a for a in acquirers if a.ticker == "PFE")
    assert pfizer.name == "Pfizer Inc"
    assert pfizer.bloc == "us"
    assert pfizer.cik == "0000078003"  # live-verified 2026-08-31 against SEC's company_tickers.json
    assert pfizer.entry_date == date(2010, 1, 1)
    assert pfizer.exit_date is None

    # Roche/Bayer/Ipsen verified to have no SEC CIK at all (unsponsored ADRs) — null is a fact here,
    # not an unverified gap (see the config file's own header).
    roche = next(a for a in acquirers if a.ticker == "RHHBY")
    assert roche.cik is None


def test_load_acquirers_basic(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
acquirers:
  - name: Test Pharma Inc
    ticker: TSTP
    cik: "0000000123"
    bloc: us
    entry_date: 2015-06-01
    exit_date: null
""",
    )
    acquirers = load_acquirers(path)
    assert len(acquirers) == 1
    a = acquirers[0]
    assert a.name == "Test Pharma Inc"
    assert a.ticker == "TSTP"
    assert a.cik == "0000000123"
    assert a.bloc == "us"
    assert a.entry_date == date(2015, 6, 1)
    assert a.exit_date is None


def test_load_acquirers_missing_name_raises(tmp_path):
    path = _write_yaml(tmp_path, "acquirers:\n  - ticker: TSTP\n    bloc: us\n    entry_date: 2015-06-01\n")
    with pytest.raises(ValueError, match="missing name/ticker"):
        load_acquirers(path)


def test_load_acquirers_invalid_bloc_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        "acquirers:\n  - name: X Inc\n    ticker: X\n    bloc: not_a_real_bloc\n    entry_date: 2015-06-01\n",
    )
    with pytest.raises(ValueError, match="invalid bloc"):
        load_acquirers(path)


def test_load_acquirers_empty_file_returns_empty_list(tmp_path):
    path = _write_yaml(tmp_path, "acquirers: []\n")
    assert load_acquirers(path) == []


def test_upsert_acquirer_roster_calls_repo_once_per_entry(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
acquirers:
  - name: A Inc
    ticker: A
    bloc: us
    entry_date: 2010-01-01
  - name: B Inc
    ticker: B
    bloc: allied
    entry_date: 2010-01-01
""",
    )
    acquirers = load_acquirers(path)
    repo = MagicMock()

    count = upsert_acquirer_roster(acquirers, repo)

    assert count == 2
    assert repo.upsert_acquirer_company.call_count == 2
    repo.upsert_acquirer_company.assert_any_call(name="A Inc", ticker="A", cik=None)
