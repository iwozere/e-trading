"""
P22 — `p22_base_rates.yaml` loader (spec §4.2, §3.5), first real consumer.

The file itself has been fully curated since 2026-08-31 (`docs/Tasks.md` item 2) but had no
reader — every `config/pipeline/*.yaml` file was "not loaded by any code yet" until
`features/block_b.py` needed one. Parsing is intentionally dumb (no validation beyond "is this
valid YAML with the expected top-level keys") — the file's own header already documents which
`by_therapeutic_area` entries are `null` on purpose; this loader doesn't second-guess that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import BASE_RATES_YAML


@dataclass(frozen=True)
class BaseRates:
    """Parsed `p22_base_rates.yaml` — only the fields `features/block_b.py` currently reads.
    `phase_2_success`/`phase_3_to_filing`/`filing_to_approval`/`first_cycle_approval` and
    `pdufa_priors`/`orphan_by_phase` are intentionally NOT modeled here yet — the phase-conditional
    and orphan-conditional combination with `by_therapeutic_area` is genuinely underspecified by
    spec (`docs/Tasks.md` "Decisions needed" item 13), so nothing reads them yet; add fields here
    only once that combination is decided, not speculatively."""

    loa_from_phase_1_overall: float
    by_therapeutic_area: Dict[str, Optional[float]]
    biomarker_selection_modifier: float


def load_base_rates(path: Path = BASE_RATES_YAML) -> BaseRates:
    """Parse `p22_base_rates.yaml`. Raises `ValueError` if the required top-level keys are
    missing — this drives a scoring-relevant probability, so a malformed file should fail loudly,
    not silently score every company on the overall fallback."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = {"loa_from_phase_1_overall", "by_therapeutic_area", "modifiers"} - raw.keys()
    if missing:
        raise ValueError(f"{path} missing required top-level keys: {sorted(missing)}")

    return BaseRates(
        loa_from_phase_1_overall=float(raw["loa_from_phase_1_overall"]),
        by_therapeutic_area=dict(raw["by_therapeutic_area"]),
        biomarker_selection_modifier=float(raw["modifiers"]["biomarker_selection"]),
    )
