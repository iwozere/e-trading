"""Tests for outputs/dossier.py (spec §6.2) — fake repo; checks sourcing rules rather than layout."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import src.ml.pipeline.p22_biotech_ma.features.context as feature_context
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import MarketCapResult
from src.ml.pipeline.p22_biotech_ma.outputs import dossier

_AS_OF = date(2026, 10, 30)
_FILING = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/"


class _Repo:
    def __init__(self) -> None:
        self.facts: Dict[str, List[Dict[str, Any]]] = {
            "cash_and_equivalents": [{"value": 300e6, "source_url": _FILING, "source_id": "sec", "period_end": date(2026, 6, 30)}],
            "shares_outstanding": [{"value": 50e6, "source_url": _FILING, "source_id": "sec", "period_end": date(2026, 6, 30)}],
            "total_debt": [{"value": 10e6, "source_url": None, "source_id": "sec", "period_end": date(2026, 6, 30)}],
            "quarterly_opex_burn": [
                {"value": -50e6, "source_url": _FILING, "source_id": "sec", "period_end": date(2026, 6, 30)},
                {"value": -50e6, "source_url": _FILING, "source_id": "sec", "period_end": date(2026, 3, 31)},
            ],
        }

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of: date):
        del company_id, as_of
        return self.facts.get(metric, [])

    def get_trials_for_company(self, company_id: int, as_of=None):
        del company_id
        return [{"nct_id": "NCT01", "asset_id": 5, "asset_name": "XY-101", "therapeutic_area": "oncology_solid",
                 "phase": "PHASE3", "status": "RECRUITING", "primary_completion_date": date(2027, 3, 1)}]

    def get_verified_process_events(self, company_id: int, as_of: date):
        del company_id, as_of
        return [{"event_date": date(2026, 9, 1), "state": "disclosed_open", "scope": "whole_company",
                 "source_url": "https://www.sec.gov/Archives/edgar/data/1/x/ex99.htm"}]

    def get_verified_activist_positions(self, company_id: int, as_of: date):
        del company_id, as_of
        return []

    def get_verified_partnership_structures(self, company_id: int, as_of: date):
        del company_id, as_of
        return []

    def list_companies_full(self):
        return [_COMPANY]


_COMPANY = {"company_id": 7, "cik": "0000000001", "ticker": "XYZ", "name": "Xyz Bio", "exchange": "Nasdaq", "sic_code": "2834"}
_SCORE = {
    "company_id": 7, "rank_by_composite": 1, "composite": 2.4, "tier": 2, "tier_reason": "disclosed_sale_process",
    "subscores": {"fundamental": 0.4, "coverage": 0.5, "blocks": {"target_asset_quality": 0.8},
                  "likely_acquirer": {"name": "Merck", "source": "fit_argmax"}, "features": {"is_foreign_domiciled": 0.0}},
    "contributions": {"top_pairs": [{"acquirer_name": "Merck", "fit": 0.7, "terms": {"modality_capability": 1.0},
                                     "gates": {"fis": 1.0}, "unknown_gates": ["size_feasibility"]}]},
}


def _patch_market_cap(monkeypatch, fake) -> None:
    """The dossier's identity line and Block C (through `FeatureContext.get_market_cap`) both derive market cap."""
    monkeypatch.setattr(dossier, "compute_market_cap", fake)
    monkeypatch.setattr(feature_context, "compute_market_cap", fake)


def _render(monkeypatch) -> str:
    _patch_market_cap(monkeypatch, lambda repo, cid, as_of: (
        MarketCapResult(company_id=7, market_cap=500e6, price_trade_date=date(2026, 10, 29), shares_outstanding=50e6), None))
    return dossier.render_dossier(_Repo(), _COMPANY, _SCORE, _AS_OF, {"run_id": 3, "model_version": "v", "config_hash": "h"})


def test_dossier_links_sources_and_lists_unsourced_figures(monkeypatch):
    text = _render(monkeypatch)
    assert dossier.DISCLAIMER in text
    assert f"$300.0M ([source]({_FILING})" in text
    assert "Total debt" not in text  # no source_url -> not printed as a figure ...
    assert "## Omitted (no source)" in text and "total_debt" in text  # ... but listed
    assert "[NCT01](https://clinicaltrials.gov/study/NCT01)" in text
    assert "Market cap: $500.0M = close 10.00 on 2026-10-29" in text


def test_dossier_shows_runway_arithmetic_and_process_signals(monkeypatch):
    text = _render(monkeypatch)
    assert "Runway: (cash + short-term investments) ÷ average quarterly burn × 3 = **18.0 months**" in text
    assert "Tier 2 (disclosed_sale_process)" in text
    assert "disclosed_open / whole_company ([filing](https://www.sec.gov/Archives/edgar/data/1/x/ex99.htm))" in text
    assert "| Merck | 0.70 | modality_capability 1.00 | fis 1.00 | size_feasibility |" in text


def test_write_dossiers_only_ranked_top_n(monkeypatch, tmp_path):
    _patch_market_cap(monkeypatch, lambda repo, cid, as_of: (None, "no_price"))
    rows = [_SCORE, {**_SCORE, "company_id": 8, "rank_by_composite": None}]
    paths = dossier.write_dossiers(_Repo(), rows, _AS_OF, {"run_id": 3}, tmp_path, top_n=5)
    assert [p.name for p in paths] == ["001_XYZ.md"]
    assert "Market cap: not computable as of 2026-10-30 (no_price)" in paths[0].read_text(encoding="utf-8")
