"""
P22 — feature registry (spec §4).

A thin name -> function map so the eventual scoring layer (M4) can look
features up by name (e.g. `"block_c.cash_runway_months"`) without importing
every block module individually, and so tooling can enumerate what exists.
Not a plugin system — just a decorator + dict, matching the scale of this
problem (dozens of features across Blocks A-F, not hundreds).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

FeatureFn = Callable[..., Optional[float]]

_REGISTRY: Dict[str, FeatureFn] = {}


def register_feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    """
    Decorator: `@register_feature("block_c.cash_runway_months")`.

    Raises on a duplicate name at import time — a silently-overwritten
    feature function (e.g. two block modules both claiming
    `"block_c.cash_runway_months"` by copy-paste error) is a much worse
    failure than a loud one here.
    """

    def decorator(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise ValueError(f"Feature {name!r} already registered (by {_REGISTRY[name].__module__})")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_feature(name: str) -> FeatureFn:
    if name not in _REGISTRY:
        raise KeyError(f"No feature registered under {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_features() -> List[str]:
    return sorted(_REGISTRY)
