"""Tests for backtest/ (spec §0.3, §2.5, §5.3, §8.4): metrics, labels, baselines and the harness loop."""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest import baselines, harness, labels as labels_mod
from src.ml.pipeline.p22_biotech_ma.backtest.labels import (
    CvrPolicy,
    CvrPolicyError,
    DealLabels,
    add_months,
    consideration_value,
    deal_return,
    is_censored,
    load_cvr_policy,
)
from src.ml.pipeline.p22_biotech_ma.backtest.metrics import MetricsTable, PooledMetrics, snapshot_counts
from src.ml.pipeline.p22_biotech_ma.cli.backtest_cli import success_criteria
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import UniverseCandidate
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import load_scoring_config

_ZERO = CvrPolicy(valuation_method="zero", return_basis="upfront_per_share")


def _deal(target_id: int, announced: date, deal_type: str = "strategic_acquisition", **extra: Any) -> Dict[str, Any]:
    return {"deal_id": target_id * 10, "target_id": target_id, "announcement_date": announced, "deal_type": deal_type, **extra}


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------

def test_snapshot_counts_counts_hits_in_top_k_only():
    counts = snapshot_counts([1, 2, 3, 4], positives={2, 4, 9}, universe_size=100, k=2)
    assert (counts.hits, counts.positives, counts.base_rate) == (1, 3, 0.03)


def test_pooled_metrics_micro_averages_and_reports_random():
    pooled = PooledMetrics(k=10)
    pooled.add(snapshot_counts(list(range(10)), {0, 1, 50}, universe_size=100, k=10))  # 2 hits, base 0.03
    pooled.add(snapshot_counts(list(range(10)), {60}, universe_size=200, k=10))  # 0 hits, base 0.005
    m = pooled.as_dict()
    assert m["precision"] == pytest.approx(2 / 20)
    assert m["random_precision"] == pytest.approx((10 * 0.03 + 10 * 0.005) / 20)
    assert m["lift"] == pytest.approx(2 / (0.3 + 0.05))
    assert m["recall"] == pytest.approx(2 / 4)
    assert m["random_recall"] == pytest.approx((0.1 + 0.05) / 2)


def test_precision_divides_by_k_when_ranking_is_short():
    pooled = PooledMetrics(k=5)
    pooled.add(snapshot_counts([7], {7}, universe_size=50, k=5))
    assert pooled.as_dict()["precision"] == pytest.approx(1 / 5)


def test_metrics_with_no_positives_report_none_not_zero_division():
    table = MetricsTable(ks=(5,))
    table.add("model", [1, 2], set(), 10)
    row = table.as_rows()[0]
    assert row["lift"] is None and row["recall"] is None and row["precision"] == 0.0


# ---------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------

def test_add_months_clamps_month_end():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 11, 15), 24) == date(2025, 11, 15)
    assert add_months(date(2025, 3, 31), -1) == date(2025, 2, 28)


def test_positive_window_is_open_at_as_of_and_closed_at_horizon():
    labels = DealLabels([
        _deal(1, date(2020, 1, 1)),   # on as_of: not in (as_of, ...]
        _deal(2, date(2020, 1, 2)),
        _deal(3, date(2022, 1, 1)),   # exactly as_of + 24m: in
        _deal(4, date(2022, 1, 2)),   # past the window
        _deal(5, date(2021, 6, 1), "reverse_merger"),
        _deal(6, date(2021, 6, 1), "pe_take_private"),
    ])
    as_of = date(2020, 1, 1)
    assert labels.positives(as_of) == {2, 3}
    assert labels.pe_positives(as_of) == {6}


def test_terminated_deal_still_counts_as_announced_positive():
    labels = DealLabels([_deal(1, date(2020, 6, 1), status="terminated")])
    assert labels.positives(date(2020, 1, 1)) == {1}


def test_pending_deal_detection():
    labels = DealLabels([
        _deal(1, date(2020, 1, 1), completion_date=date(2020, 4, 1)),
        _deal(2, date(2020, 1, 1), status="terminated"),
        _deal(3, date(2019, 1, 1)),  # no completion date: pending for a year
        _deal(4, date(2020, 3, 1)),  # announced after as_of
    ])
    assert labels.pending(date(2020, 2, 1)) == {1}
    assert labels.pending(date(2019, 6, 1)) == {3}


def test_is_censored():
    assert not is_censored(date(2023, 10, 30), date(2025, 10, 30))
    assert is_censored(date(2023, 11, 1), date(2025, 10, 30))


def test_load_cvr_policy_reads_repo_config():
    policy = load_cvr_policy()
    assert policy == _ZERO


def test_load_cvr_policy_rejects_unimplemented_method(tmp_path):
    path = tmp_path / "cvr.yaml"
    path.write_text("valuation_method: probability_weighted\nreturn_basis: upfront_per_share\n", encoding="utf-8")
    with pytest.raises(CvrPolicyError):
        load_cvr_policy(path)


def test_consideration_ignores_cvr_under_zero_policy():
    assert consideration_value({"upfront_per_share": 40, "cvr_max_per_share": 5}, _ZERO) == 40.0
    assert consideration_value({"upfront_per_share": None}, _ZERO) is None


def test_deal_return_uses_close_on_announcement_basis(monkeypatch):
    monkeypatch.setattr(labels_mod, "is_price_suspect", lambda repo, cid, as_of: False)
    calls: List[tuple] = []

    def fake_close(repo, cid, on_or_before, basis):
        calls.append((cid, on_or_before, basis))
        return {"close": 20.0}

    monkeypatch.setattr(labels_mod, "close_on_basis", fake_close)
    deal = _deal(1, date(2021, 3, 1), upfront_per_share=30.0)
    assert deal_return(None, deal, date(2020, 1, 1), _ZERO) == pytest.approx(0.5)
    assert calls == [(1, date(2020, 1, 1), date(2021, 3, 1))]


def test_deal_return_none_for_suspect_prices(monkeypatch):
    monkeypatch.setattr(labels_mod, "is_price_suspect", lambda repo, cid, as_of: True)
    assert deal_return(None, _deal(1, date(2021, 3, 1), upfront_per_share=30.0), date(2020, 1, 1), _ZERO) is None


# ---------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------

_FEATURES = {
    1: {"phase_max": 3.0, "cash_runway_months": 10.0, "days_to_next_catalyst": 300.0},
    2: {"phase_max": 3.0, "cash_runway_months": 40.0, "days_to_next_catalyst": 30.0},
    3: {"phase_max": 1.0, "cash_runway_months": 99.0, "days_to_next_catalyst": 5.0},
    4: {"phase_max": None, "cash_runway_months": None, "days_to_next_catalyst": None},
    5: {"phase_max": 2.0, "cash_runway_months": None, "days_to_next_catalyst": None},
}


def test_naive_informed_orders_by_phase_then_runway_unknowns_last():
    assert baselines.naive_informed(_FEATURES) == [2, 1, 5, 3, 4]


def test_catalyst_proximity_only_ranks_phase2_plus_with_known_catalyst():
    assert baselines.catalyst_proximity(_FEATURES) == [2, 1]


def test_model_ranking_drops_unranked_and_restricts_population():
    rows = [{"company_id": 1, "composite": 0.4}, {"company_id": 2, "composite": 1.2},
            {"company_id": 3, "composite": None}, {"company_id": 4, "composite": 0.9}]
    assert baselines.model_ranking(rows) == [2, 4, 1]
    assert baselines.model_ranking(rows, {1, 3, 4}) == [4, 1]


def test_block_ranking_sorts_by_block_and_friction_ascending():
    rows = [{"company_id": 1, "subscores": {"blocks": {"strategic_fit": 0.2, "friction_penalty": 1.0}}},
            {"company_id": 2, "subscores": {"blocks": {"strategic_fit": 0.9, "friction_penalty": 0.0}}},
            {"company_id": 3, "subscores": {"blocks": {"strategic_fit": None, "friction_penalty": 0.5}}}]
    assert baselines.block_ranking(rows, "strategic_fit") == [2, 1]  # no score -> left out
    assert baselines.block_ranking(rows, "friction_penalty") == [2, 3, 1]  # subtracted term: low first
    assert baselines.block_ranking(rows, "friction_penalty", {1, 3}) == [3, 1]


def test_score_rows_flatten_blocks_features_and_labels():
    rows = [{"company_id": 7, "ticker": "SEVN", "composite": 0.6, "tier": 0,
             "subscores": {"fundamental": 0.6, "blocks": {"strategic_fit": 0.4}, "features": {"phase_max": 3.0}}}]
    out = harness.score_rows_for_snapshot(date(2020, 1, 1), rows, {7: {"phase_max": 3.0, "cash_runway_months": 12.0}}, {7}, set())
    assert out == [{"as_of": date(2020, 1, 1), "company_id": 7, "ticker": "SEVN", "positive": 1, "pe_positive": 0,
                    "composite": 0.6, "tier": 0, "fundamental": 0.6, "coverage": None, "dilution_gate": None,
                    "strategic_fit_raw": None, "block_strategic_fit": 0.4, "f_phase_max": 3.0, "f_cash_runway_months": 12.0}]


# ---------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------

def _candidate(cik: str, eligible: bool = True, spac: bool = False) -> UniverseCandidate:
    return UniverseCandidate(cik=cik, name=cik, sic=None, ticker=None, exchange=None, most_recent_form=None,
                             most_recent_filed=None, eligible_reporting=eligible, eligible_exchange=None, flagged_spac=spac)


def test_universe_as_of_uses_latest_quarter_not_after_as_of():
    history = {
        "2019q4": [_candidate("A")],
        "2020q1": [_candidate("A"), _candidate("B"), _candidate("C", eligible=False), _candidate("D", spac=True)],
        "2020q2": [_candidate("Z")],
    }
    assert harness.universe_as_of(history, date(2020, 4, 15)) == ("2020q1", ["A", "B"])
    assert harness.universe_as_of(history, date(2019, 1, 1)) == (None, [])


def test_snapshot_dates_skip_censored():
    window = harness.BacktestWindow(start=date(2020, 1, 1), end=date(2020, 6, 1), labels_complete_through=date(2022, 3, 1),
                                    step_months=1)
    assert harness.snapshot_dates(window) == [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]


class _HarnessRepo:
    def __init__(self) -> None:
        self.companies = [
            {"company_id": 1, "cik": "A", "ticker": "AAA", "name": "A", "role": "target", "is_active": False},
            {"company_id": 2, "cik": "B", "ticker": "BBB", "name": "B", "role": "target", "is_active": True},
            {"company_id": 3, "cik": "C", "ticker": "CCC", "name": "C", "role": "target", "is_active": True},
        ]

    def list_companies_full(self) -> List[Dict[str, Any]]:
        return self.companies


def test_run_backtest_keeps_later_acquired_inactive_company_and_drops_pending(monkeypatch):
    seen_targets: List[List[int]] = []

    def fake_select(repo, ctx, companies, acquirer_ids, as_of, require_active=True):
        assert require_active is False
        return [c["company_id"] for c in companies], {"inactive": 0}

    def fake_score(repo, as_of, config, acquirers, target_ids=None, deal_history=None):
        seen_targets.append(list(target_ids or []))
        rows = [{"company_id": cid, "composite": float(cid), "tier": 0} for cid in target_ids or []]
        return SimpleNamespace(rows=rows, summary={})

    monkeypatch.setattr(harness, "select_targets", fake_select)
    monkeypatch.setattr(harness, "score_universe", fake_score)
    monkeypatch.setattr(harness, "resolve_acquirers", lambda acquirers, companies, as_of: [])
    monkeypatch.setattr(harness, "baseline_features",
                        lambda ids, as_of, ctx: {cid: {"phase_max": 2.0, "cash_runway_months": 1.0, "days_to_next_catalyst": 10.0} for cid in ids})
    monkeypatch.setattr(harness, "return_labels", lambda repo, labels, ids, as_of, policy: [{"company_id": i} for i in ids])

    labels = DealLabels([
        _deal(1, date(2021, 1, 1)),  # company 1 is inactive today: acquired later, must stay in
        _deal(3, date(2019, 12, 1), completion_date=date(2020, 6, 1)),  # pending at as_of
    ])
    history = {"2019q4": [_candidate("A"), _candidate("B"), _candidate("C")]}
    window = harness.BacktestWindow(start=date(2020, 1, 1), end=date(2020, 1, 1), labels_complete_through=date(2026, 1, 1), ks=(1,))

    result = harness.run_backtest(_HarnessRepo(), history, window, labels, load_scoring_config(), [], _ZERO)

    assert seen_targets == [[1, 2]]
    snap = result.snapshots[0]
    assert snap["positives"] == 1 and snap["excluded_pending_deal"] == 1
    model = next(r for r in result.all_names.as_rows() if r["ranking"] == "model")
    assert model["hits"] == 0  # model ranks company 2 first (composite 2.0 > 1.0)
    naive = next(r for r in result.all_names.as_rows() if r["ranking"] == "naive_informed")
    assert naive["hits"] == 1  # tie on phase/runway -> company_id order puts 1 first
    assert result.return_rows == [{"company_id": 1}]
    assert [(r["company_id"], r["positive"]) for r in result.score_rows] == [(1, 1), (2, 0)]
    assert result.learned_window.as_rows() == []  # one snapshot: no 24-month-old training rows yet
    assert all("learned_p" not in r for r in result.score_rows)
    block_names = {r["ranking"] for r in result.all_names.as_rows()} - {"model", "naive_informed", "catalyst_proximity"}
    assert block_names == {harness.BLOCK_RANKING_PREFIX + b for b in load_scoring_config().composite}


def test_success_criteria_compares_model_to_targets_and_naive():
    rows = [
        {"ranking": "model", "k": 50, "lift": 3.5, "precision": None, "recall": None, "random_recall": None},
        {"ranking": "model", "k": 25, "lift": None, "precision": 0.10, "recall": None, "random_recall": None},
        {"ranking": "model", "k": 200, "lift": None, "precision": None, "recall": 0.6, "random_recall": 0.29},
        {"ranking": "naive_informed", "k": 50, "lift": 2.0, "precision": None, "recall": None, "random_recall": None},
    ]
    c = success_criteria(rows)
    assert c["lift_at_50_pass"] and not c["precision_at_25_pass"] and c["recall_at_200_pass"]
    assert c["beats_naive_informed_at_50"]
