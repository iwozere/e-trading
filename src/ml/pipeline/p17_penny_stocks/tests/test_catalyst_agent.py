"""Tests for P17 CatalystAgent (daily 8-K index cache + legacy EDGAR fallback)."""

from pathlib import Path
import sys
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p17_penny_stocks.agents.catalyst_agent import CatalystAgent
from src.ml.pipeline.p17_penny_stocks.config import P17CatalystConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate


TARGET_DATE = "2026-06-25"
_INDEX_COLS = ["cik", "company", "accession_number", "items",
               "description", "filed_date", "primary_document"]


def _bare_agent(tmp_path) -> CatalystAgent:
    """Agent with a real (empty) EdgarDownloader — for _classify/_recency unit tests."""
    return CatalystAgent(P17CatalystConfig(), tmp_path, TARGET_DATE)


def _cache_agent(tmp_path):
    """Agent whose 8-K index dir is a real tmp dir; TEST resolves to CIK 111."""
    agent = CatalystAgent(P17CatalystConfig(), tmp_path, TARGET_DATE)
    agent._edgar = MagicMock()
    index_dir = tmp_path / "8k_index"
    index_dir.mkdir()
    agent._edgar._8k_index_dir = index_dir
    agent._edgar.load_company_tickers.return_value = {"0": {"ticker": "TEST", "cik_str": 111}}
    return agent, index_dir


def _legacy_agent(tmp_path):
    """Agent whose index dir does NOT exist → falls back to per-CIK EDGAR."""
    agent = CatalystAgent(P17CatalystConfig(), tmp_path, TARGET_DATE)
    agent._edgar = MagicMock()
    agent._edgar._8k_index_dir = tmp_path / "missing_8k_index"  # never created
    agent._edgar.load_company_tickers.return_value = {"0": {"ticker": "TEST", "cik_str": 111}}
    return agent


def _candidate() -> Candidate:
    return Candidate(ticker="TEST", price=5.0)


def _write_index(index_dir: Path, date_str: str, rows: list) -> None:
    full = [{**{c: "" for c in _INDEX_COLS}, "filed_date": date_str, **r} for r in rows]
    pd.DataFrame(full, columns=_INDEX_COLS).to_csv(
        index_dir / f"{date_str}.csv.gz", index=False, compression="gzip"
    )


# ── Classification (no EDGAR) ───────────────────────────────────────────────

def test_fda_keyword_is_tier1(tmp_path):
    cat, pts = _bare_agent(tmp_path)._classify("8.01", "FDA clearance granted")
    assert cat == "tier1_news"
    assert pts == P17CatalystConfig().points_tier1


def test_material_agreement_item_is_tier1(tmp_path):
    cat, pts = _bare_agent(tmp_path)._classify("1.01", "Form 8-K")
    assert cat == "material_agreement"
    assert pts == P17CatalystConfig().points_tier1


def test_press_release_item_is_tier2(tmp_path):
    cat, pts = _bare_agent(tmp_path)._classify("7.01", "Form 8-K")
    assert cat in ("tier2_news", "press_release")
    assert pts == P17CatalystConfig().points_tier2


def test_purely_bearish_items_not_catalyst(tmp_path):
    cat, pts = _bare_agent(tmp_path)._classify("1.03", "Notice of bankruptcy")
    assert cat is None
    assert pts == 0.0


def test_neutral_officer_change_not_catalyst(tmp_path):
    cat, _ = _bare_agent(tmp_path)._classify("5.02", "Departure of director")
    assert cat is None


# ── Recency ─────────────────────────────────────────────────────────────────

def test_recency_decays_with_age(tmp_path):
    agent = _bare_agent(tmp_path)
    assert agent._recency_multiplier(1) == 1.0
    assert agent._recency_multiplier(5) == 0.85
    assert agent._recency_multiplier(10) == 0.6
    assert agent._recency_multiplier(25) == 0.35
    assert agent._recency_multiplier(40) == 0.0


# ── Cache path (primary) ────────────────────────────────────────────────────

def test_cache_recent_fda_sets_high_score(tmp_path):
    agent, index_dir = _cache_agent(tmp_path)
    _write_index(index_dir, "2026-06-24", [
        {"cik": "111", "items": "8.01", "description": "FDA approval of NDA"},
    ])
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == P17CatalystConfig().points_tier1  # 90 * 1.0
    assert any("tier1_news" in s for s in c.catalyst_signals)
    agent._edgar.get_recent_filings.assert_not_called()  # cache path, no network


def test_cache_old_filing_is_decayed(tmp_path):
    agent, index_dir = _cache_agent(tmp_path)
    _write_index(index_dir, "2026-06-05", [
        {"cik": "111", "items": "8.01", "description": "FDA approval"},
    ])
    c = _candidate()
    agent.run([c])
    assert abs(c.catalyst_score - 90.0 * 0.35) < 1e-6  # 20 days old


def test_cache_multi_catalyst_bonus(tmp_path):
    agent, index_dir = _cache_agent(tmp_path)
    cfg = P17CatalystConfig()
    _write_index(index_dir, "2026-06-24", [
        {"cik": "111", "items": "1.01", "description": "Definitive merger agreement"},
        {"cik": "111", "items": "2.02", "description": "Quarterly results"},
    ])
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == min(100.0, cfg.points_tier1 + cfg.multi_catalyst_bonus)


def test_cache_leading_zero_cik_matches(tmp_path):
    """Index CIK stored without leading zeros still matches a resolved int CIK."""
    agent, index_dir = _cache_agent(tmp_path)
    _write_index(index_dir, "2026-06-24", [
        {"cik": "0000111", "items": "8.01", "description": "FDA approval"},
    ])
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == P17CatalystConfig().points_tier1


def test_cache_no_catalyst_leaves_zero(tmp_path):
    agent, index_dir = _cache_agent(tmp_path)
    _write_index(index_dir, "2026-06-24", [
        {"cik": "111", "items": "5.02", "description": "Officer appointment"},
    ])
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == 0.0
    assert c.catalyst_signals == []


def test_cache_filing_outside_lookback_ignored(tmp_path):
    agent, index_dir = _cache_agent(tmp_path)
    # ~85 days before target → no index file even written in window; but write one
    # inside the scanned window with an old date to ensure age filter applies.
    _write_index(index_dir, "2026-06-24", [
        {"cik": "111", "items": "8.01", "description": "FDA approval",
         "filed_date": "2026-04-01"},
    ])
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == 0.0


def test_cache_unrelated_cik_not_scored(tmp_path):
    agent, index_dir = _cache_agent(tmp_path)
    _write_index(index_dir, "2026-06-24", [
        {"cik": "999", "items": "8.01", "description": "FDA approval"},
    ])
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == 0.0


# ── Legacy fallback (no index dir) ──────────────────────────────────────────

def test_legacy_fallback_used_when_no_index(tmp_path):
    agent = _legacy_agent(tmp_path)
    agent._edgar.get_recent_filings.return_value = [
        {"form": "8-K", "items": "8.01", "primaryDocDescription": "FDA approval",
         "filingDate": "2026-06-24"},
    ]
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == P17CatalystConfig().points_tier1
    agent._edgar.get_recent_filings.assert_called()  # fallback path exercised


def test_unresolved_cik_skipped(tmp_path):
    agent, _ = _cache_agent(tmp_path)
    agent._edgar.load_company_tickers.return_value = {}  # no CIK for TEST
    c = _candidate()
    agent.run([c])
    assert c.catalyst_score == 0.0
