"""
P22 — Phase 1 scoring configuration (spec §5.1) and `config_hash`.

`p22_weights.yaml` holds every weight the scorer uses. `config_hash` hashes that file together with
the acquirer universe (`p22_acquirers.yaml`, whose header says changes "bump config_hash and require a
re-score") so two score runs are comparable only when their hashes match.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import ACQUIRERS_YAML, WEIGHTS_YAML
from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import VALID_BLOCS

COMPOSITE_POSITIVE_TERMS = (
    "acquirer_pressure_matched", "target_asset_quality", "strategic_fit", "financial_attractiveness",
)
FRICTION_TERM = "friction_penalty"
_DIRECTIONS = ("higher", "lower")


@dataclass(frozen=True)
class FeatureWeight:
    """One feature's weight inside its block, and which way is better."""

    weight: float
    direction: str


@dataclass(frozen=True)
class ScoringConfig:
    """Parsed `p22_weights.yaml`."""

    model_version: str
    composite: Dict[str, float]
    dilution_gate_factor: float
    blocks: Dict[str, Dict[str, FeatureWeight]]
    fit_preference: Dict[str, float]
    size_feasibility_full_below: float
    fis_penalty_by_bloc: Dict[str, float]
    antitrust_same_indication_penalty: float
    top_pairs: int


def load_scoring_config(path: Path = WEIGHTS_YAML) -> ScoringConfig:
    """
    Parse and validate the weights file.

    Raises:
        ValueError: on a missing composite term, a negative weight, an unknown direction, or a bloc
            table that doesn't cover every bloc `p22_acquirers.yaml` may use.
    """
    raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    composite = {k: float(v) for k, v in (raw.get("composite") or {}).items()}
    expected = set(COMPOSITE_POSITIVE_TERMS) | {FRICTION_TERM}
    if set(composite) != expected:
        raise ValueError(f"composite must define exactly {sorted(expected)}, got {sorted(composite)}")

    blocks: Dict[str, Dict[str, FeatureWeight]] = {}
    for block, features in (raw.get("blocks") or {}).items():
        blocks[block] = {}
        for feature, spec in features.items():
            direction = spec.get("direction")
            if direction not in _DIRECTIONS:
                raise ValueError(f"{block}.{feature}: direction must be one of {_DIRECTIONS}, got {direction!r}")
            blocks[block][feature] = FeatureWeight(weight=float(spec["weight"]), direction=direction)

    fis = {k: float(v) for k, v in (raw.get("fis_penalty_by_acquirer_bloc") or {}).items()}
    if set(fis) != set(VALID_BLOCS):
        raise ValueError(f"fis_penalty_by_acquirer_bloc must cover {sorted(VALID_BLOCS)}, got {sorted(fis)}")

    config = ScoringConfig(
        model_version=str(raw["model_version"]),
        composite=composite,
        dilution_gate_factor=float(raw["dilution_gate_factor"]),
        blocks=blocks,
        fit_preference={k: float(v) for k, v in (raw.get("fit_preference") or {}).items()},
        size_feasibility_full_below=float(raw["size_feasibility_full_below"]),
        fis_penalty_by_bloc=fis,
        antitrust_same_indication_penalty=float(raw["antitrust_same_indication_penalty"]),
        top_pairs=int(raw.get("top_pairs", 3)),
    )
    weights = list(composite.values()) + list(config.fit_preference.values()) + list(fis.values())
    weights += [fw.weight for features in blocks.values() for fw in features.values()]
    if any(w < 0 for w in weights):
        raise ValueError("weights and penalties must be non-negative")
    return config


def config_hash(paths: Sequence[Path] = (WEIGHTS_YAML, ACQUIRERS_YAML)) -> str:
    """SHA-256 over the raw bytes of every scoring input file, in order (first 16 hex chars)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
