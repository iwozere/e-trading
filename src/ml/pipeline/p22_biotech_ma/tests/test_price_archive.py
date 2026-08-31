"""
Tests for ingest/price_archive.py — the pure split-adjustment math (spec
§2.0.7, added v0.6). No database required; see docs/implementation-plan.md.
"""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.price_archive import (
    CorporateActionRatio,
    adjusted_close,
    compute_adjustment_factor,
)


def _action(ex_date, action_type, ratio, known_from_date):
    return CorporateActionRatio(ex_date=ex_date, action_type=action_type, ratio=ratio, known_from_date=known_from_date)


def test_no_actions_factor_is_one():
    factor = compute_adjustment_factor([], date(2019, 6, 1), date(2023, 1, 1))
    assert factor == 1.0


def test_reverse_split_after_trade_date_and_known_by_as_of_applies():
    # 1-for-20 reverse split, ratio=0.05 per spec's own worked example.
    actions = [_action(date(2023, 3, 1), "reverse_split", 0.05, date(2023, 3, 1))]
    factor = compute_adjustment_factor(actions, date(2019, 6, 1), date(2023, 6, 1))
    assert factor == 0.05


def test_action_after_as_of_is_excluded_even_if_ex_date_after_trade_date():
    """An action whose ex_date is after `as_of` has not happened yet as of that as_of."""
    actions = [_action(date(2024, 1, 1), "reverse_split", 0.05, date(2024, 1, 1))]
    factor = compute_adjustment_factor(actions, date(2019, 6, 1), date(2023, 6, 1))
    assert factor == 1.0


def test_lookahead_guard_excludes_action_not_yet_known_by_as_of():
    """
    The exact failure mode spec §2.0.7 calls out: an action that already took
    effect (ex_date <= as_of) but was not yet publicly known as of `as_of`
    must not be applied — this is what `known_from` is for.
    """
    actions = [_action(date(2019, 8, 1), "reverse_split", 0.05, known_from_date=date(2019, 9, 1))]
    factor = compute_adjustment_factor(actions, date(2019, 6, 1), as_of=date(2019, 8, 15))
    assert factor == 1.0


def test_action_on_or_before_trade_date_is_excluded():
    """An action effective on-or-before the trade_date itself is already baked into the raw print."""
    actions = [_action(date(2019, 6, 1), "reverse_split", 0.05, date(2019, 6, 1))]
    factor = compute_adjustment_factor(actions, trade_date=date(2019, 6, 1), as_of=date(2023, 1, 1))
    assert factor == 1.0


def test_dividend_and_spinoff_do_not_affect_price_adjustment_factor():
    actions = [
        _action(date(2020, 1, 1), "dividend", None, date(2020, 1, 1)),
        _action(date(2020, 6, 1), "spinoff", None, date(2020, 6, 1)),
        _action(date(2020, 9, 1), "ticker_change", None, date(2020, 9, 1)),
    ]
    factor = compute_adjustment_factor(actions, date(2019, 6, 1), date(2023, 1, 1))
    assert factor == 1.0


def test_multiple_splits_compound_multiplicatively():
    actions = [
        _action(date(2020, 1, 1), "split", 4.0, date(2020, 1, 1)),
        _action(date(2022, 1, 1), "reverse_split", 0.1, date(2022, 1, 1)),
    ]
    factor = compute_adjustment_factor(actions, date(2019, 1, 1), date(2023, 1, 1))
    assert factor == 4.0 * 0.1


def test_adjusted_close_forward_split_example():
    # Spec's own worked example: raw=$100 pre-split, 4:1 forward split -> $25 post-split-equivalent.
    actions = [_action(date(2020, 1, 1), "split", 4.0, date(2020, 1, 1))]
    result = adjusted_close(100.0, actions, date(2019, 6, 1), date(2023, 1, 1))
    assert result == 25.0


def test_adjusted_close_propagates_none_for_missing_raw_price():
    """None must propagate as missing, never as zero (spec §4's feature contract)."""
    result = adjusted_close(None, [], date(2019, 6, 1), date(2023, 1, 1))
    assert result is None
