"""Tests for `src.data.pipeline.registry` — the plugin registry itself."""

from __future__ import annotations

from collections import Counter

import pytest

from src.data.pipeline.base_plugin import PluginValidationError
from src.data.pipeline.registry import PLUGIN_REGISTRY, get_by_category, get_by_name, list_categories


def test_registry_not_empty():
    assert len(PLUGIN_REGISTRY) > 0


def test_no_duplicate_names():
    counts = Counter(spec.name for spec in PLUGIN_REGISTRY)
    dupes = [name for name, n in counts.items() if n > 1]
    assert dupes == []


def test_every_spec_validates():
    """Every registered script_path must resolve, exist, and be allowlisted."""
    failures = []
    for spec in PLUGIN_REGISTRY:
        try:
            spec.validate()
        except PluginValidationError as e:
            failures.append(str(e))
    assert failures == [], f"{len(failures)} invalid PluginSpec(s):\n" + "\n".join(failures)


def test_get_by_name_known_plugin():
    spec = get_by_name("P22 Daily Price Ingest")
    assert spec is not None
    assert spec.category == "p22"


def test_get_by_name_unknown_returns_none():
    assert get_by_name("does not exist") is None


def test_get_by_category():
    p20_specs = get_by_category("p20")
    assert len(p20_specs) > 0
    assert all(spec.category == "p20" for spec in p20_specs)


def test_get_by_category_unknown_returns_empty():
    assert get_by_category("no-such-category") == []


def test_list_categories_includes_known_groups():
    categories = list_categories()
    assert {"p20", "p21", "p22"}.issubset(set(categories))


@pytest.mark.parametrize("category", ["p20", "p21", "p22"])
def test_category_crons_are_nonempty_strings(category):
    for spec in get_by_category(category):
        assert isinstance(spec.cron, str) and spec.cron.strip()
