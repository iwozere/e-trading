# ---------------------------------------------------------------------------
# registry.py — catalog of indicators and provider priority
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal, Any
from src.indicators.constants import (
    DEFAULT_PARAMETERS,
    INDICATOR_INPUTS,
    INDICATOR_OUTPUTS,
    PROVIDER_SUPPORT,
    TECHNICAL_INDICATOR_NAMES,
    FUNDAMENTAL_INDICATOR_NAMES
)

@dataclass
class IndicatorMeta:
    kind: Literal["tech","fund"]
    inputs: List[str]              # tech: e.g. ["close"], fund: []
    outputs: List[str]             # canonical outputs (e.g. ["value"], or ["upper","middle","lower"])
    providers: List[str]           # priority order keys that match adapters dict
    defaults: dict[str, object] = None  # canonical defaults (optional)
    description: str = ""          # Human-readable description
    legacy_names: List[str] = None # Legacy indicator names for compatibility

# Build indicator metadata dynamically from constants
def _build_indicator_meta() -> Dict[str, IndicatorMeta]:
    """Build indicator metadata from unified constants."""
    from src.indicators.models import TECHNICAL_INDICATORS, FUNDAMENTAL_INDICATORS, LEGACY_INDICATOR_NAMES

    meta = {}

    # Technical indicators
    for name in TECHNICAL_INDICATOR_NAMES:
        description = TECHNICAL_INDICATORS.get(name, name.replace("_", " ").title())
        legacy_names = [k for k, v in LEGACY_INDICATOR_NAMES.items() if v == name and k != name.upper()]

        meta[name] = IndicatorMeta(
            kind="tech",
            inputs=INDICATOR_INPUTS.get(name, []),
            outputs=INDICATOR_OUTPUTS.get(name, ["value"]),
            providers=PROVIDER_SUPPORT.get(name, ["ta-lib", "pandas-ta"]),
            defaults=DEFAULT_PARAMETERS.get(name, {}),
            description=description,
            legacy_names=legacy_names
        )

    # Fundamental indicators
    for name in FUNDAMENTAL_INDICATOR_NAMES:
        description = FUNDAMENTAL_INDICATORS.get(name, name.replace("_", " ").title())
        legacy_names = [k for k, v in LEGACY_INDICATOR_NAMES.items() if v == name and k != name.upper()]

        meta[name] = IndicatorMeta(
            kind="fund",
            inputs=INDICATOR_INPUTS.get(name, []),
            outputs=INDICATOR_OUTPUTS.get(name, ["value"]),
            providers=PROVIDER_SUPPORT.get(name, ["fundamentals"]),
            defaults=DEFAULT_PARAMETERS.get(name, {}),
            description=description,
            legacy_names=legacy_names
        )

    return meta

INDICATOR_META = _build_indicator_meta()

# Import unified naming utilities
from src.indicators.models import get_canonical_name as get_unified_canonical_name

# Legacy name mapping for backward compatibility (built from metadata)
LEGACY_NAME_MAPPING = {}
for canonical_name, meta in INDICATOR_META.items():
    if meta.legacy_names:
        for legacy_name in meta.legacy_names:
            LEGACY_NAME_MAPPING[legacy_name] = canonical_name

def get_canonical_name(indicator_name: str) -> str:
    """Get canonical name for an indicator, handling legacy names."""
    # First try the unified naming system
    unified_name = get_unified_canonical_name(indicator_name)
    if unified_name in INDICATOR_META:
        return unified_name

    # Fall back to registry-specific mapping
    return LEGACY_NAME_MAPPING.get(indicator_name, indicator_name.lower())

def get_indicator_meta(indicator_name: str) -> Optional[IndicatorMeta]:
    """Get indicator metadata by name (handles legacy names)."""
    canonical_name = get_canonical_name(indicator_name)
    return INDICATOR_META.get(canonical_name)

def get_all_technical_indicators() -> Dict[str, IndicatorMeta]:
    """Get all technical indicators."""
    return {name: meta for name, meta in INDICATOR_META.items() if meta.kind == "tech"}

def get_all_fundamental_indicators() -> Dict[str, IndicatorMeta]:
    """Get all fundamental indicators."""
    return {name: meta for name, meta in INDICATOR_META.items() if meta.kind == "fund"}

def get_indicators_by_provider(provider: str) -> Dict[str, IndicatorMeta]:
    """Get all indicators supported by a specific provider."""
    return {name: meta for name, meta in INDICATOR_META.items() if provider in meta.providers}

