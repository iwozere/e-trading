"""
P22 — walk-forward learned ranking for the backtest (2026-11-09).

The Phase 1 composite's hand-set weights ranked at random (lift@50 1.00 on 2026-11-09) while one of
its blocks alone ranked well, so this ranks the same raw features with weights learned from past
snapshots instead.

**No lookahead.** A snapshot at `as_of` is ranked by a model trained only on earlier snapshot rows
whose 24-month label window had closed by `as_of` (row `as_of` <= `as_of` - `LABEL_HORIZON_MONTHS`),
i.e. labels that were already known. Until enough such rows exist (`min_train_positives`) there is
no learned ranking for that snapshot.

**Feature set** (2026-11-09 offline replay of the walk-forward on the step 5 `scores.csv`, 75
snapshots, lift @25/@50/@200): the nine target features alone 1.59/1.46/1.33; plus all four
therapeutic-area features 1.52/1.43/1.34; plus `ta_lead_area_known` and `ta_acquirer_demand` only
1.66/1.62/1.36 (naive-informed 1.61/1.54/1.33). The other area features stay in `scores.csv` for
analysis but not in the model.

**Runway** (2026-11-10 replay on the first weekly run's `scores.csv`, same 75 snapshots): the
latest quarter's burn (`cash_runway_latest_q_months`) in place of the trailing four quarters'
1.92/1.76/1.43 vs 1.53/1.57/1.40, and ahead in the second half too (1.54/1.66/1.39 vs
1.32/1.60/1.37); both together 1.49/1.57/1.40. The naive baseline keeps the trailing runway (§0.3).

**Price-gap leak.** Companies with no market cap at `as_of` were acquired about twice as often as
the rest (2026-11-09 analysis): the FMP price history mostly lacks names that later delisted, so a
missing price says the company is about to exit. The price-derived features (`PRICE_FEATURES`) are
therefore filled with the training median, so the model cannot tell "unknown" from "typical". Every
other feature keeps its missing values, which the gradient-boosted trees route on their own.

**Live use.** The weekly backtest job (`jobs/run_backtest.py`) trains one more model on every
backtest row (`fit_final_model`) and saves it (`save_model`, `config.LEARNED_MODEL_PATH`); daily
scoring loads it (`load_model`) and ranks the live universe with it next to the composite.
"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.baselines import tie_break
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
    "f_cash_runway_latest_q_months",
    "f_dilution_risk",
    "f_is_foreign_domiciled",
    "f_ev_to_cash",
    "f_size_band",
    "f_ta_lead_area_known",
    "f_ta_acquirer_demand",
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


@dataclass
class SavedModel:
    """
    The model live scoring uses: trained by the weekly backtest job on every backtest row (all of
    whose 24-month labels were complete), with the feature list it was trained on.
    """

    fitted: FittedModel
    features: Tuple[str, ...]
    labels_through: date
    trained_at: datetime


def fit_final_model(rows: Sequence[Mapping[str, Any]], labels_through: date, params: Optional[LearnedParams] = None) -> Optional[SavedModel]:
    """
    Train on the backtest rows whose label window closed by `labels_through` (normally all of them,
    since the backtest's last snapshot is `labels_through` minus 24 months). `None` when too few positives.
    """
    cutoff = add_months(labels_through, -LABEL_HORIZON_MONTHS)
    train = [r for r in rows if r["as_of"] <= cutoff]
    fitted = fit_model(train, cutoff, params or LearnedParams())
    if fitted is None:
        _logger.warning("Final learned model not trained: too few positives in %d rows through %s", len(train), cutoff)
        return None
    _logger.info("Final learned model: %d rows, %d positives, labels through %s", fitted.train_rows, fitted.train_positives, labels_through)
    return SavedModel(fitted=fitted, features=LEARNED_FEATURES, labels_through=labels_through, trained_at=datetime.now(timezone.utc))


def save_model(model: SavedModel, path: Path) -> None:
    """Write `model` to `path` atomically (a scoring run reading it never sees half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(model, fh)
    tmp.replace(path)


def load_model(path: Path) -> Optional[SavedModel]:
    """
    The model at `path` (written by `save_model` on this machine), or `None` when there is none or it
    was trained on a different feature list than the code now uses.
    """
    if not path.exists():
        _logger.info("No learned model at %s", path)
        return None
    with path.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301 — our own file, written by the weekly backtest job
    if not isinstance(model, SavedModel):
        _logger.warning("Ignoring %s: not a SavedModel", path)
        return None
    if tuple(model.features) != LEARNED_FEATURES:
        _logger.warning("Ignoring learned model %s: trained on %s, code now uses %s", path, model.features, LEARNED_FEATURES)
        return None
    return model


def rank_by_probability(probabilities: Mapping[int, float], company_ids: Optional[set[int]] = None, salt: str = "") -> List[int]:
    """
    Company ids by probability, best first, restricted to `company_ids` when given. Ties are broken
    by `baselines.tie_break(cid, salt)`, never by id order (ids track the label).
    """
    items = [(p, cid) for cid, p in probabilities.items() if company_ids is None or cid in company_ids]
    return [cid for _, cid in sorted(items, key=lambda x: (-x[0], tie_break(x[1], salt)))]
