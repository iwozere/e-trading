"""Tests for scoring/ (spec §4.4, §5.1–§5.2, M4): weights config, fit(), composite, score run. No live DB."""

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import AcquirerConfig
from src.ml.pipeline.p22_biotech_ma.ingest.size_floor import SizeFloorDecision
from src.ml.pipeline.p22_biotech_ma.scoring import score_run
from src.ml.pipeline.p22_biotech_ma.scoring.composite import block_scores, fundamental_score, rank_desc, rank_percentile
from src.ml.pipeline.p22_biotech_ma.scoring.fit import AcquirerProfile, TargetProfile, best_fits, fit, size_feasibility
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import FeatureWeight, config_hash, load_scoring_config

_AS_OF = date(2026, 10, 29)
CONFIG = load_scoring_config()


# ---------------------------------------------------------------------
# weights_config
# ---------------------------------------------------------------------

def test_repo_weights_file_matches_spec_composite():
    assert CONFIG.composite == {
        "acquirer_pressure_matched": 0.30, "target_asset_quality": 0.25, "strategic_fit": 0.20,
        "financial_attractiveness": 0.15, "friction_penalty": 0.10,
    }
    assert CONFIG.fis_penalty_by_bloc["elevated_scrutiny"] == 0.45
    assert CONFIG.dilution_gate_factor == 0.5


def test_load_scoring_config_rejects_missing_composite_term(tmp_path):
    text = (PROJECT_ROOT / "config" / "pipeline" / "p22_weights.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "w.yaml"
    bad.write_text(text.replace("  strategic_fit: 0.20", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="composite"):
        load_scoring_config(bad)


def test_config_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("x: 1", encoding="utf-8")
    first = config_hash([a])
    a.write_text("x: 2", encoding="utf-8")
    assert config_hash([a]) != first
    assert len(first) == 16


# ---------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------

def _acquirer(**kwargs: Any) -> AcquirerProfile:
    base: Dict[str, Any] = {"company_id": 1, "name": "Acq", "bloc": "us"}
    base.update(kwargs)
    return AcquirerProfile(**base)


def _target(**kwargs: Any) -> TargetProfile:
    base: Dict[str, Any] = {"company_id": 7, "lead_therapeutic_area": "oncology_solid", "lead_modality": "small_molecule"}
    base.update(kwargs)
    return TargetProfile(**base)


def test_size_feasibility_is_full_then_linear_then_zero():
    assert size_feasibility(100.0, 15.0, 0.15) == 1.0
    assert size_feasibility(100.0, 57.5, 0.15) == pytest.approx(0.5)
    assert size_feasibility(100.0, 100.0, 0.15) == 0.0
    assert size_feasibility(None, 10.0, 0.15) is None


def test_fit_renormalizes_over_known_preference_terms():
    result = fit(_acquirer(modalities=frozenset({"small_molecule"})), _target(), CONFIG)
    assert result.terms["ta_overlap"] is None and result.terms["geographic_fit"] is None
    assert result.preference == pytest.approx(1.0)
    assert result.fit == pytest.approx(1.0)
    assert result.unknown_gates == ["size_feasibility", "antitrust"]


def test_fit_is_none_when_no_preference_term_is_known():
    assert fit(_acquirer(), _target(lead_modality="unclassified"), CONFIG).fit is None


def test_fit_gates_multiply_rather_than_add():
    acquirer = _acquirer(
        bloc="elevated_scrutiny",
        modalities=frozenset({"small_molecule"}),
        pipeline_gap_by_ta={"oncology_solid": 5e9},
        late_stage_indications=frozenset({"non-small cell lung cancer"}),
        dry_powder=10e9,
    )
    target = _target(lead_indication="Non-Small  Cell Lung Cancer", enterprise_value=1e9)
    result = fit(acquirer, target, CONFIG)
    assert result.preference == pytest.approx(1.0)
    assert result.gates == {"size_feasibility": 1.0, "fis": pytest.approx(0.55), "antitrust": 0.5}
    assert result.fit == pytest.approx(0.55 * 0.5)
    assert result.unknown_gates == []


def test_best_fits_orders_gated_pairs_below_clean_ones():
    clean = _acquirer(company_id=1, modalities=frozenset({"small_molecule"}))
    gated = _acquirer(company_id=2, bloc="elevated_scrutiny", modalities=frozenset({"small_molecule"}))
    unknown = _acquirer(company_id=3)
    ordered = best_fits([unknown, gated, clean], _target(), CONFIG)
    assert [r.acquirer_id for r in ordered] == [1, 2, 3]


# ---------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------

def test_rank_percentile_maps_ties_to_lower_rank_and_skips_unknown():
    assert rank_percentile({1: 0.0, 2: 0.0, 3: 1.0, 4: None}) == {1: 0.0, 2: 0.0, 3: 1.0}
    assert rank_percentile({1: 5.0}) == {1: 0.5}
    assert rank_percentile({}) == {}


def test_block_scores_respect_direction_and_ignore_zero_weight():
    features = {1: {"a": 1.0, "b": 9.0}, 2: {"a": 2.0, "b": 1.0}, 3: {"a": 3.0, "b": None}}
    weights = {"a": FeatureWeight(1.0, "lower"), "b": FeatureWeight(0.0, "higher")}
    assert block_scores(features, weights) == {1: 1.0, 2: 0.5, 3: 0.0}


def test_block_scores_none_when_no_feature_known():
    scores = block_scores({1: {"a": None}, 2: {"a": 1.0}}, {"a": FeatureWeight(1.0, "higher")})
    assert scores[1] is None and scores[2] == 0.5


def test_fundamental_matches_spec_formula_when_every_block_known():
    blocks = {"acquirer_pressure_matched": 1.0, "target_asset_quality": 0.5, "strategic_fit": 0.5,
              "financial_attractiveness": 1.0, "friction_penalty": 1.0}
    result = fundamental_score(blocks, 0.0, CONFIG)
    assert result.value == pytest.approx(0.30 + 0.125 + 0.10 + 0.15 - 0.10)
    assert result.coverage == pytest.approx(1.0)
    assert result.missing_blocks == []


def test_fundamental_scales_known_blocks_and_records_missing():
    blocks = {"target_asset_quality": 1.0, "friction_penalty": None}
    result = fundamental_score(blocks, None, CONFIG)
    assert result.value == pytest.approx(0.90)  # 0.25 scaled to the full 0.90 positive weight
    assert result.coverage == pytest.approx(0.25 / 0.90)
    assert "friction_penalty" in result.missing_blocks
    assert result.notes == ["dilution_risk_unknown"]


def test_fundamental_dilution_gate_halves_score():
    result = fundamental_score({"target_asset_quality": 1.0}, 1.0, CONFIG)
    assert result.value == pytest.approx(0.45)
    assert result.dilution_gate == 0.5


def test_fundamental_none_without_positive_block():
    assert fundamental_score({"friction_penalty": 1.0}, 0.0, CONFIG).value is None


def test_rank_desc_breaks_ties_by_company_id_and_skips_unknown():
    assert rank_desc({3: 0.5, 1: 0.5, 2: 0.9, 4: None}) == {2: 1, 1: 2, 3: 3}


# ---------------------------------------------------------------------
# score run (end to end over a fake repo)
# ---------------------------------------------------------------------

class _FakeRepo:
    """Three targets and one acquirer; target 12 has a verified acquisition-option partner."""

    def __init__(self) -> None:
        self.companies = [
            {"company_id": 1, "cik": "0000078003", "ticker": "PFE", "name": "Pfizer", "role": "both", "is_active": True},
            {"company_id": 10, "cik": "0000000010", "ticker": "AAA", "name": "Alpha", "role": "target", "is_active": True},
            {"company_id": 11, "cik": "0000000011", "ticker": "BBB", "name": "Beta", "role": "target", "is_active": True},
            {"company_id": 12, "cik": "0000000012", "ticker": "CCC", "name": "Gamma", "role": "target", "is_active": True},
            {"company_id": 13, "cik": "0000000013", "ticker": "DDD", "name": "Delta", "role": "target", "is_active": False},
        ]
        self.trials = {
            1: [{"asset_id": 100, "therapeutic_area": "oncology_solid", "phase": "PHASE3", "primary_completion_date": None}],
            10: [{"asset_id": 110, "therapeutic_area": "oncology_solid", "phase": "PHASE3",
                  "primary_completion_date": date(2026, 12, 1)}],
            11: [{"asset_id": 111, "therapeutic_area": "neurology", "phase": "PHASE1",
                  "primary_completion_date": date(2029, 1, 1)}],
            12: [],
        }
        self.assets = {
            1: [{"asset_id": 100, "modality": "small_molecule", "indication": "Lung Cancer"}],
            10: [{"asset_id": 110, "modality": "small_molecule", "indication": "Melanoma"}],
            11: [{"asset_id": 111, "modality": "gene_therapy", "indication": "ALS"}],
        }
        self.facts: Dict[tuple, float] = {
            (10, "market_cap"): 900e6, (10, "cash_and_equivalents"): 300e6, (10, "is_foreign_domiciled"): 0.0,
            (11, "market_cap"): 3e9, (11, "cash_and_equivalents"): 100e6, (11, "is_foreign_domiciled"): 1.0,
        }

    def list_companies_full(self) -> List[Dict[str, Any]]:
        return self.companies

    def get_trials_for_company(self, company_id: int):
        return self.trials.get(company_id, [])

    def get_assets_for_company(self, company_id: int):
        return self.assets.get(company_id, [])

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of: date):
        del as_of
        value = self.facts.get((company_id, metric))
        return [{"value": value}] if value is not None else []

    def get_patent_expiries_for_acquirer(self, acquirer_id: int):
        del acquirer_id
        return []

    def count_phase3_assets_by_therapeutic_area(self, company_id: int):
        del company_id
        return {}

    def get_verified_process_events(self, company_id: int, as_of: date):
        del company_id, as_of
        return []

    def get_verified_activist_positions(self, company_id: int, as_of: date):
        del company_id, as_of
        return []

    def get_verified_partnership_structures(self, company_id: int, as_of: date):
        del as_of
        if company_id == 12:
            return [{"structure_type": "acquisition_option", "partner_id": 1, "partner_equity_pct": None}]
        return []


_ACQUIRERS = [
    AcquirerConfig(name="Pfizer Inc", ticker="PFE", cik="0000078003", bloc="us",
                   entry_date=date(2010, 1, 1), exit_date=None),
    AcquirerConfig(name="Future Co", ticker="FUT", cik=None, bloc="us", entry_date=date(2030, 1, 1), exit_date=None),
]


def test_resolve_acquirers_matches_by_cik_and_skips_not_yet_entered():
    resolved = score_run.resolve_acquirers(_ACQUIRERS, _FakeRepo().companies, _AS_OF)
    assert [(cfg.ticker, cid) for cfg, cid in resolved] == [("PFE", 1)]


def test_select_targets_excludes_acquirers_inactive_and_below_floor(monkeypatch):
    floors = {10: True, 11: None, 12: False}
    monkeypatch.setattr(score_run, "evaluate_size_floor",
                        lambda repo, cid, as_of: SizeFloorDecision(floors[cid], "market_cap"))
    targets, excluded = score_run.select_targets(_FakeRepo(), _FakeRepo().companies, {1}, _AS_OF)
    assert targets == [10, 11]
    assert excluded == {"inactive": 1, "acquirer": 1, "below_size_floor": 1, "size_floor_unknown_kept": 1}


def test_score_universe_ranks_tiers_and_explains():
    result = score_run.score_universe(_FakeRepo(), _AS_OF, CONFIG, _ACQUIRERS, target_ids=[10, 11, 12])
    rows = {r["company_id"]: r for r in result.rows}

    # Gamma has no fundamentals at all, so it stays unranked even though its partner puts it in tier 3.
    assert rows[12]["tier"] == 3 and rows[12]["tier_reason"] == "partner_acquisition_option"
    assert rows[12]["composite"] is None and rows[12]["rank_by_composite"] is None
    assert rows[12]["subscores"]["likely_acquirer"] == {"company_id": 1, "name": "Pfizer", "source": "partner_identity"}

    # Alpha: later phase, nearer catalyst, cheaper, domestic, modality match -> ranks first.
    assert rows[10]["rank_by_composite"] == 1 and rows[11]["rank_by_composite"] == 2
    assert rows[10]["subscores"]["likely_acquirer"]["source"] == "fit_argmax"
    assert rows[10]["contributions"]["top_pairs"][0]["terms"]["modality_capability"] == 1.0
    assert rows[11]["subscores"]["blocks"]["strategic_fit"] == 0.0
    assert "acquirer_pressure_matched" in rows[10]["subscores"]["missing_blocks"]
    assert [r["company_id"] for r in result.rows] == [10, 11, 12]

    assert result.summary["ranked"] == 2
    assert result.summary["by_tier"] == {0: 2, 1: 0, 2: 0, 3: 1}


def test_persist_score_run_writes_only_score_columns():
    class _Sink:
        def __init__(self) -> None:
            self.rows: List[Dict[str, Any]] = []
            self.run: Optional[Dict[str, Any]] = None

        def create_score_run(self, **kwargs: Any) -> int:
            self.run = kwargs
            return 5

        def insert_scores(self, run_id: int, rows: List[Dict[str, Any]]) -> int:
            assert run_id == 5
            self.rows = rows
            return len(rows)

    result = score_run.score_universe(_FakeRepo(), _AS_OF, CONFIG, _ACQUIRERS, target_ids=[10, 11])
    sink = _Sink()
    assert score_run.persist_score_run(sink, _AS_OF, replace(CONFIG), "abc", result.rows) == 5
    assert sink.run == {"as_of_date": _AS_OF, "model_version": "phase1-rules-v1", "config_hash": "abc"}
    assert "ticker" not in sink.rows[0] and "rank" not in sink.rows[0]
    assert set(sink.rows[0]) == {"company_id", "composite", "rank_by_composite", "tier", "tier_reason",
                                 "subscores", "contributions"}
