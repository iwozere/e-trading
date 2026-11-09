"""Tests for `backtest/learned.py` — the walk-forward learned ranking."""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.baselines import tie_break
from src.ml.pipeline.p22_biotech_ma.backtest.learned import (
    LEARNED_FEATURES,
    LearnedParams,
    WalkForwardRanker,
    feature_matrix,
    fit_final_model,
    fit_model,
    load_model,
    price_fill_values,
    rank_by_probability,
    save_model,
)

_SMALL = LearnedParams(min_train_positives=5, max_iter=30, min_samples_leaf=2)


def _row(company_id: int, as_of: date, positive: int, phase: Optional[float] = 2.0, size: Optional[float] = 1.0) -> Dict[str, Any]:
    row: Dict[str, Any] = {name: 1.0 for name in LEARNED_FEATURES}
    row.update({"as_of": as_of, "company_id": company_id, "positive": positive, "f_phase_max": phase, "f_size_band": size})
    return row


def _phase_driven_rows(as_of: date, n: int = 60) -> List[Dict[str, Any]]:
    """Late-phase companies (phase 3) are the positives."""
    return [_row(i, as_of, positive=int(i % 3 == 0), phase=3.0 if i % 3 == 0 else 1.0) for i in range(n)]


def test_feature_matrix_fills_only_price_features():
    rows = [_row(1, date(2020, 1, 1), 0, phase=None, size=None)]
    matrix = feature_matrix(rows, {"f_size_band": 2.5, "f_ev_to_cash": 0.0})
    assert matrix.shape == (1, len(LEARNED_FEATURES))
    assert matrix[0, LEARNED_FEATURES.index("f_size_band")] == 2.5
    assert math.isnan(matrix[0, LEARNED_FEATURES.index("f_phase_max")])


def test_feature_matrix_treats_non_numeric_and_infinite_as_missing():
    rows = [{"f_phase_max": "x", "f_cash_runway_months": float("inf")}]
    matrix = feature_matrix(rows, {})
    assert math.isnan(matrix[0, LEARNED_FEATURES.index("f_phase_max")])
    assert math.isnan(matrix[0, LEARNED_FEATURES.index("f_cash_runway_months")])


def test_price_fill_values_use_median_of_known():
    rows = [_row(i, date(2020, 1, 1), 0, size=s) for i, s in enumerate([0.0, 2.0, 4.0, None])]
    rows[0]["f_ev_to_cash"] = None
    fill = price_fill_values(rows)
    assert fill["f_size_band"] == 2.0
    assert fill["f_ev_to_cash"] == 1.0


def test_fit_model_needs_enough_positives():
    rows = _phase_driven_rows(date(2018, 1, 1), n=9)  # 3 positives
    assert fit_model(rows, date(2018, 1, 1), _SMALL) is None
    assert fit_model(_phase_driven_rows(date(2018, 1, 1)), date(2018, 1, 1), _SMALL) is not None


def test_missing_price_cannot_signal_a_deal():
    """Positives all lack a market cap; once filled, the model must not tell them apart."""
    rows = [_row(i, date(2018, 1, 1), positive=int(i < 20), size=None if i < 20 else 1.0) for i in range(80)]
    model = fit_model(rows, date(2018, 1, 1), _SMALL)
    assert model is not None
    missing, known = model.predict([_row(100, date(2020, 1, 1), 0, size=None), _row(101, date(2020, 1, 1), 0, size=1.0)])
    assert missing == known


def test_walk_forward_trains_only_on_rows_with_complete_labels():
    ranker = WalkForwardRanker(params=_SMALL)
    ranker.add(_phase_driven_rows(date(2018, 1, 1)))
    ranker.add(_phase_driven_rows(date(2019, 6, 1)))
    assert {r["as_of"] for r in ranker.training_rows(date(2020, 1, 1))} == {date(2018, 1, 1)}
    assert ranker.predict(date(2019, 12, 1), [_row(1, date(2019, 12, 1), 0)]) is None  # nothing 24 months old yet

    probabilities = ranker.predict(date(2020, 1, 1), [_row(1, date(2020, 1, 1), 0, phase=1.0), _row(2, date(2020, 1, 1), 0, phase=3.0)])
    assert probabilities is not None
    assert rank_by_probability(probabilities) == [2, 1]


def test_rank_by_probability_breaks_ties_by_hash_and_restricts():
    probabilities = {3: 0.5, 1: 0.5, 2: 0.9}
    tied = sorted((1, 3), key=lambda c: tie_break(c, "s"))
    assert rank_by_probability(probabilities, salt="s") == [2, *tied]
    assert rank_by_probability(probabilities, {1, 3}, "s") == tied


def test_fit_final_model_trains_only_on_complete_labels():
    rows = _phase_driven_rows(date(2018, 1, 1)) + [_row(500 + i, date(2025, 1, 1), 1) for i in range(30)]  # 2025 rows: labels open
    model = fit_final_model(rows, date(2026, 11, 6), _SMALL)
    assert model is not None
    assert model.fitted.train_rows == 60 and model.features == LEARNED_FEATURES
    assert model.labels_through == date(2026, 11, 6)
    assert fit_final_model(_phase_driven_rows(date(2018, 1, 1), n=9), date(2026, 11, 6), _SMALL) is None


def test_save_and_load_model_round_trip(tmp_path):
    model = fit_final_model(_phase_driven_rows(date(2018, 1, 1)), date(2026, 11, 6), _SMALL)
    assert model is not None
    path = tmp_path / "models" / "learned_model.pkl"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded is not None
    rows = [_row(1, date(2026, 11, 9), 0, phase=3.0), _row(2, date(2026, 11, 9), 0, phase=1.0)]
    assert loaded.fitted.predict(rows) == model.fitted.predict(rows)
    assert not list(tmp_path.glob("models/*.tmp"))


def test_load_model_ignores_missing_file_and_other_feature_lists(tmp_path):
    assert load_model(tmp_path / "none.pkl") is None
    model = fit_final_model(_phase_driven_rows(date(2018, 1, 1)), date(2026, 11, 6), _SMALL)
    assert model is not None
    model.features = ("f_phase_max",)
    save_model(model, tmp_path / "old.pkl")
    assert load_model(tmp_path / "old.pkl") is None
