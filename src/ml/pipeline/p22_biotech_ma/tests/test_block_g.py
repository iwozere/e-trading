"""Tests for features/block_g.py (spec §2.6, §4.7, §5.2). Fake repo, no live DB."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.block_g import BlockG, apply_process_tier, build_block_g, process_tier
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext

_AS_OF = date(2026, 9, 8)
_COMPANY_ID = 7


class _FakeRepo:
    def __init__(
        self,
        process_events: List[Dict[str, Any]] | None = None,
        activist_positions: List[Dict[str, Any]] | None = None,
        partnerships: List[Dict[str, Any]] | None = None,
    ):
        self._process_events = process_events or []
        self._activist_positions = activist_positions or []
        self._partnerships = partnerships or []

    def get_verified_process_events(self, company_id: int, as_of: date):
        del company_id, as_of
        return self._process_events

    def get_verified_activist_positions(self, company_id: int, as_of: date):
        del company_id, as_of
        return self._activist_positions

    def get_verified_partnership_structures(self, company_id: int, as_of: date):
        del company_id, as_of
        return self._partnerships


def _ctx(**kwargs) -> FeatureContext:
    return FeatureContext(as_of=_AS_OF, repo=_FakeRepo(**kwargs))


def _pos(
    filer_cik: str, filer_type: str | None, form_type: str, filed_date: date, pct: float | None, stated_intent: str | None = None
) -> Dict[str, Any]:
    return {
        "filer_cik": filer_cik,
        "filer_type": filer_type,
        "form_type": form_type,
        "filed_date": filed_date,
        "pct_of_class": pct,
        "stated_intent": stated_intent,
    }


# ---------------------------------------------------------------------
# build_block_g
# ---------------------------------------------------------------------

def test_build_block_g_defaults_when_nothing_verified():
    g = build_block_g(_COMPANY_ID, _AS_OF, _ctx())
    assert g == BlockG()


def test_build_block_g_disclosed_open_computes_days_since():
    ctx = _ctx(process_events=[
        {"state": "disclosed_open", "scope": "whole_company", "event_date": date(2026, 8, 1)},
    ])
    g = build_block_g(_COMPANY_ID, _AS_OF, ctx)
    assert g.process_state == "disclosed_open"
    assert g.process_scope == "whole_company"
    assert g.days_since_process_open == (_AS_OF - date(2026, 8, 1)).days


def test_build_block_g_concluded_state_has_no_days_since():
    ctx = _ctx(process_events=[
        {"state": "concluded_no_deal", "scope": "whole_company", "event_date": date(2026, 8, 1)},
    ])
    g = build_block_g(_COMPANY_ID, _AS_OF, ctx)
    assert g.process_state == "concluded_no_deal"
    assert g.days_since_process_open is None


def test_build_block_g_activist_intent_takes_the_strongest():
    ctx = _ctx(activist_positions=[
        _pos("A1", "activist", "SC 13D", date(2026, 1, 5), 6.0, stated_intent="engagement"),
        _pos("A2", "activist", "SC 13D/A", date(2026, 3, 5), 6.0, stated_intent="sale_demand"),
    ])
    g = build_block_g(_COMPANY_ID, _AS_OF, ctx)
    assert g.has_13d_activist is True
    assert g.activist_intent_max == "sale_demand"
    assert g.activist_escalation == 1  # one SC 13D/A amendment


def test_build_block_g_strategic_toehold_from_corporate_filer():
    ctx = _ctx(activist_positions=[
        _pos("S1", "strategic_corporate", "SC 13D", date(2026, 2, 1), 7.5),
    ])
    g = build_block_g(_COMPANY_ID, _AS_OF, ctx)
    assert g.has_13d_activist is False  # not filer_type == 'activist'
    assert g.has_strategic_toehold is True
    assert g.strategic_toehold_pct == 7.5


def test_build_block_g_exit_amendment_cancels_strategic_toehold():
    ctx = _ctx(activist_positions=[
        _pos("S1", "strategic_corporate", "SC 13G", date(2025, 2, 1), 9.0),
        _pos("S1", "strategic_corporate", "SC 13G/A", date(2026, 2, 1), 3.9),
    ])
    g = build_block_g(_COMPANY_ID, _AS_OF, ctx)
    assert g.has_strategic_toehold is False
    assert g.strategic_toehold_pct is None


def test_build_block_g_null_pct_does_not_count():
    ctx = _ctx(activist_positions=[_pos("A1", "activist", "SC 13D/A", date(2026, 4, 1), None)])
    assert build_block_g(_COMPANY_ID, _AS_OF, ctx).has_13d_activist is False


def test_build_block_g_activist_uses_latest_filing_only():
    # Latest filing is a 13G (activist turned passive) — no longer a 13D activist.
    ctx = _ctx(activist_positions=[
        _pos("A1", "activist", "SC 13D", date(2025, 6, 1), 8.0),
        _pos("A1", "activist", "SC 13G", date(2026, 6, 1), 8.0),
    ])
    assert build_block_g(_COMPANY_ID, _AS_OF, ctx).has_13d_activist is False


def test_build_block_g_crossover_fund_13d_is_not_activist():
    ctx = _ctx(activist_positions=[_pos("C1", "crossover_fund", "SC 13D/A", date(2026, 4, 1), 12.0)])
    assert build_block_g(_COMPANY_ID, _AS_OF, ctx).has_13d_activist is False


def test_build_block_g_min_stake_pct_is_configurable():
    ctx = _ctx(activist_positions=[_pos("S1", "strategic_corporate", "SC 13G/A", date(2026, 2, 1), 4.0)])
    assert build_block_g(_COMPANY_ID, _AS_OF, ctx).has_strategic_toehold is False
    assert build_block_g(_COMPANY_ID, _AS_OF, ctx, min_stake_pct=3.0).has_strategic_toehold is True


def test_build_block_g_partnership_picks_strongest_structure():
    ctx = _ctx(partnerships=[
        {"structure_type": "license_only", "partner_equity_pct": None, "partner_id": 1},
        {"structure_type": "acquisition_option", "partner_equity_pct": 12.0, "partner_id": 2},
        {"structure_type": "rofn_rofr", "partner_equity_pct": 8.0, "partner_id": 3},
    ])
    g = build_block_g(_COMPANY_ID, _AS_OF, ctx)
    assert g.partner_structure_max == "acquisition_option"
    assert g.partner_equity_pct == 12.0
    assert g.partner_identity == 2


# ---------------------------------------------------------------------
# apply_process_tier
# ---------------------------------------------------------------------

def test_apply_process_tier_zero_by_default():
    assert apply_process_tier(0.6, BlockG()) == 0.6


def test_apply_process_tier_three_for_acquisition_option():
    g = BlockG(partner_structure_max="acquisition_option")
    assert apply_process_tier(0.5, g) == 3.5


def test_apply_process_tier_three_for_disclosed_open_whole_company_recent():
    g = BlockG(process_state="disclosed_open", process_scope="whole_company", days_since_process_open=100)
    assert apply_process_tier(0.4, g) == 3.4


def test_apply_process_tier_not_three_when_process_too_old():
    g = BlockG(process_state="disclosed_open", process_scope="whole_company", days_since_process_open=400)
    assert apply_process_tier(0.4, g) == 0.4  # falls through every tier


def test_apply_process_tier_two_for_sale_demand():
    g = BlockG(activist_intent_max="sale_demand")
    assert apply_process_tier(0.3, g) == 2.3


def test_apply_process_tier_two_for_strategic_toehold():
    g = BlockG(has_strategic_toehold=True)
    assert apply_process_tier(0.3, g) == 2.3


def test_apply_process_tier_two_for_rofn_rofr_partnership():
    g = BlockG(partner_structure_max="rofn_rofr")
    assert apply_process_tier(0.3, g) == 2.3


def test_apply_process_tier_one_for_13d_activist():
    g = BlockG(has_13d_activist=True)
    assert apply_process_tier(0.2, g) == 1.2


def test_apply_process_tier_one_for_rumored():
    g = BlockG(process_state="rumored")
    assert apply_process_tier(0.2, g) == 1.2


def test_apply_process_tier_one_for_asset_only_scope():
    g = BlockG(process_scope="asset_only")
    assert apply_process_tier(0.2, g) == 1.2


def test_apply_process_tier_precedence_tier3_beats_tier2():
    g = BlockG(partner_structure_max="acquisition_option", has_13d_activist=True)
    assert apply_process_tier(0.1, g) == 3.1


# ---------------------------------------------------------------------
# process_tier
# ---------------------------------------------------------------------

def test_process_tier_names_the_reason():
    assert process_tier(BlockG()) == (0, "fundamentals_only")
    assert process_tier(BlockG(partner_structure_max="acquisition_option")) == (3, "partner_acquisition_option")
    assert process_tier(BlockG(has_strategic_toehold=True)) == (2, "strategic_toehold")
    assert process_tier(BlockG(partner_structure_max="rofn_rofr")) == (2, "partner_rofn_rofr")
    assert process_tier(BlockG(has_13d_activist=True)) == (1, "13d_activist")
    assert process_tier(BlockG(process_state="rumored")) == (1, "rumored_process")
