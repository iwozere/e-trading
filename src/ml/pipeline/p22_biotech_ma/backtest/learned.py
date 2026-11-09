"""
P22 — walk-forward learned ranking for the backtest (2026-11-09).

The Phase 1 composite's hand-set weights ranked at random (lift@50 1.00 on 2026-11-09) while one of
its blocks alone ranked well, so this ranks the same raw features with weights learned from past
snapshots instead.

**No lookahead.** A snapshot at `as_of` is ranked by a model trained only on earlier snapshot rows
whose 24-month label window had closed by `as_of` (row `as_of` <= `as_of` - `LABEL_HORIZON_MONTHS`),
i.e. labels that were already known. Until enough such rows exist (`min_train_positives`) there is
no learned ranking for that snapshot.

**Price-gap leak.** Companies with no market cap at `as_of` were acquired about twice as often as
the rest (2026-11-09 analysis): the FMP price history mostly lacks names that later delisted, so a
missing price says the company is about to exit. The price-derived features (`PRICE_FEATURES`) are
therefore filled with the training median, so the model cannot tell "unknown" from "typical". Every
other feature keeps its missing values, which the gradient-boosted trees route on their own.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.labels import add_months
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

LABEL_HORIZON_MONTHS = 24

# `score_rows` columns the model reads (raw features, prefixed `f_` by the harness).
LEARNED_FEATURES = (
    "f_phase_max",
    "f_asset_count_ph2plus",
    "f_catalyst_window",
    "f_days_to_next_catalyst",
    "f_cash_runway_months",
    "f_dilution_risk",
    "f_is_foreign_domiciled",
    "f_ev_to_cash",
    "f_size_band",
    "f_ta_acquirer_demand",
    "f_ta_acquirer_phase3_assets",
    "f_ta_deal_heat_24m",
    "f_ta_acquirer_revenue_growth",
)
# Need a market cap; missing mostly means "price history lost because the company later exited".
PRICE_FEATURES = frozenset({"f_ev_to_cash", "f_size_band"})


@dataclass(frozen=True)
class LearnedParams:
    """Model and training-window settings."""

    min_train_positives: int = 200
    max_depth: int = 3
    max_iter: int = 150
    learning_rate: float = 0.05
    min_samples_leaf: int = 50
    random_state: int = 0


def _to_float(value: Any) -> float:
    """A feature value as float, with None / non-numeric / infinite as NaN."""
    if value is None:
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def feature_matrix(rows: Sequence[Mapping[str, Any]], fill: Mapping[str, float]) -> np.ndarray:
    """
    `rows` as a float matrix in `LEARNED_FEATURES` order. Missing price features take `fill[name]`;
    every other missing value stays NaN.
    """
    matrix = np.array([[_to_float(r.get(name)) for name in LEARNED_FEATURES] for r in rows], dtype=float).reshape(-1, len(LEARNED_FEATURES))
    for j, name in enumerate(LEARNED_FEATURES):
        if name in fill:
            column = matrix[:, j]
            column[np.isnan(column)] = fill[name]
    return matrix


def price_fill_values(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Median of each price feature over `rows` (0.0 when none is known)."""
    fill: Dict[str, float] = {}
    for name in PRICE_FEATURES:
        known = [v for r in rows if not math.isnan(v := _to_float(r.get(name)))]
        fill[name] = float(np.median(known)) if known else 0.0
    return fill


@dataclass
class FittedModel:
    """A trained classifier and the fill values it was trained with."""

    model: HistGradientBoostingClassifier
    fill: Dict[str, float]
    train_rows: int
    train_positives: int
    train_through: date

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[float]:
        """P(positive) per row."""
        if not rows:
            return []
        return [float(p) for p in self.model.predict_proba(feature_matrix(rows, self.fill))[:, 1]]


def fit_model(rows: Sequence[Mapping[str, Any]], train_through: date, params: LearnedParams) -> Optional[FittedModel]:
    """
    Train on `rows` (each with `positive` 0/1). `None` when there are fewer than
    `params.min_train_positives` positives or only one class.
    """
    labels = [int(r["positive"]) for r in rows]
    positives = sum(labels)
    if positives < params.min_train_positives or positives == len(labels):
        return None
    fill = price_fill_values(rows)
    # The sklearn stubs type early_stopping as str, but False is the documented way to turn it off
    # (it would otherwise hold back a random 10% of rows once there are more than 10,000).
    options: Dict[str, Any] = {"early_stopping": False}
    model = HistGradientBoostingClassifier(
        max_depth=params.max_depth,
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        min_samples_leaf=params.min_samples_leaf,
        random_state=params.random_state,
        **options,
    )
    model.fit(feature_matrix(rows, fill), np.array(labels))
    return FittedModel(model=model, fill=fill, train_rows=len(rows), train_positives=positives, train_through=train_through)


@dataclass
class WalkForwardRanker:
    """
    Collects each snapshot's score rows and ranks the next snapshot with a model trained only on rows
    whose labels were complete by then. Call `predict` before `add` for the same snapshot.
    """

    params: LearnedParams = field(default_factory=LearnedParams)
    history: List[Dict[str, Any]] = field(default_factory=list)
    _fitted: Optional[FittedModel] = None
    _fitted_rows: int = -1

    def training_rows(self, as_of: date) -> List[Dict[str, Any]]:
        """History rows whose label window closed on or before `as_of`."""
        cutoff = add_months(as_of, -LABEL_HORIZON_MONTHS)
        return [r for r in self.history if r["as_of"] <= cutoff]

    def model_for(self, as_of: date) -> Optional[FittedModel]:
        """The model usable at `as_of`; refitted only when new training rows have become eligible."""
        train = self.training_rows(as_of)
        if len(train) != self._fitted_rows:
            self._fitted_rows = len(train)
            self._fitted = fit_model(train, add_months(as_of, -LABEL_HORIZON_MONTHS), self.params)
            if self._fitted is not None:
                _logger.info("Learned model for %s: %d rows, %d positives, labels through %s",
                             as_of, self._fitted.train_rows, self._fitted.train_positives, self._fitted.train_through)
        return self._fitted

    def predict(self, as_of: date, rows: Sequence[Dict[str, Any]]) -> Optional[Dict[int, float]]:
        """`{company_id: P(positive)}` for `rows`; `None` when no model is usable at `as_of` yet."""
        model = self.model_for(as_of)
        if model is None:
            return None
        return dict(zip((r["company_id"] for r in rows), model.predict(rows)))

    def add(self, rows: Sequence[Dict[str, Any]]) -> None:
        """Keep a snapshot's rows for later training."""
        self.history.extend(rows)


def rank_by_probability(probabilities: Mapping[int, float], company_ids: Optional[set[int]] = None) -> List[int]:
    """Company ids by probability, best first (ties by id), restricted to `company_ids` when given."""
    items = [(p, cid) for cid, p in probabilities.items() if company_ids is None or cid in company_ids]
    return [cid for _, cid in sorted(items, key=lambda x: (-x[0], x[1]))]
