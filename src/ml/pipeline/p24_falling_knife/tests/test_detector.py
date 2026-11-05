"""Tests for the P24 detector on synthetic knives."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from src.ml.pipeline.p24_falling_knife.config import DAILY_PARAMS, VARIANTS
from src.ml.pipeline.p24_falling_knife.detector import BarEval, build_context, data_cliffs, detect
from src.ml.pipeline.p24_falling_knife.tests.helpers import (
    KNIFE_LEGS,
    TWO_LEG_KNIFE,
    bars_from_closes,
    knife_bars,
    path_from_legs,
)

# Bar where the knife's signal low (end of the slowing fall) lands: 1 + sum of the first five leg lengths.
SIGNAL_LOW_BAR = 1 + sum(bars for bars, _ in KNIFE_LEGS[:5]) - 1


def _signals(legs: Tuple[Tuple[int, float], ...], **overrides: object) -> List[BarEval]:
    params = DAILY_PARAMS.with_overrides(**overrides)
    signals, _ = detect(build_context(knife_bars(legs), params), params)
    return signals


def test_textbook_knife_fires_once_near_the_low() -> None:
    signals = _signals(KNIFE_LEGS)
    assert len(signals) == 1
    sig = signals[0]
    assert abs(sig.l2 - SIGNAL_LOW_BAR) <= 2
    assert 1 <= sig.t - sig.l2 <= DAILY_PARAMS.max_bars_since_low
    assert sig.drop >= 0.40
    assert sig.adx_peak >= 30 and sig.ppo_trough < 0
    assert sig.cci_2 - sig.cci_3 >= DAILY_PARAMS.cci_div_min
    assert not sig.missing and sig.knife_seq == 1


def test_point_three_is_the_support_low_not_the_bounce_top() -> None:
    sig = _signals(KNIFE_LEGS)[0]
    support_low_bar = 1 + KNIFE_LEGS[0][0] + KNIFE_LEGS[1][0] - 1
    bounce_top_bar = support_low_bar + KNIFE_LEGS[2][0]
    assert sig.s3 is not None
    assert abs(sig.s3 - support_low_bar) <= 2
    assert abs(sig.s3 - bounce_top_bar) > 2


def test_drop_below_forty_percent_gives_no_signal() -> None:
    shallow = ((330, 100.0), (18, 80.0), (8, 88.0), (10, 66.0), (10, 62.0), (60, 75.0))
    assert _signals(shallow) == []


def test_steeper_second_fall_is_not_a_divergence() -> None:
    # The fall into the low is as violent as the fall into S3: CCI makes a deeper trough, not a shallower one.
    steep = ((330, 100.0), (18, 72.0), (8, 80.0), (4, 50.0), (60, 75.0))
    assert _signals(steep) == []


def test_no_intermediate_support_gives_no_signal() -> None:
    one_way = ((330, 100.0), (36, 49.0), (60, 75.0))
    assert _signals(one_way) == []
    assert _signals(one_way, require_cci=False) != []


def test_wide_channel_gives_no_signal() -> None:
    assert _signals(KNIFE_LEGS, w_narrow=-1.0) == []


def test_flat_cci_requirement_blocks_small_divergence() -> None:
    sig = _signals(KNIFE_LEGS)[0]
    assert _signals(KNIFE_LEGS, cci_div_min=(sig.cci_2 - sig.cci_3) + 50.0) == []


def test_second_leg_fires_after_rearm() -> None:
    signals = _signals(TWO_LEG_KNIFE)
    assert len(signals) == 2
    first, second = signals
    assert first.p1 == second.p1
    assert second.knife_seq == 2
    assert second.l2 > first.t
    assert second.s3 is not None and second.s3 > first.l2


def test_no_repeat_signal_in_the_same_leg() -> None:
    signals = _signals(KNIFE_LEGS, require_cci=False)
    assert len({s.p1 for s in signals}) == len(signals)


def test_drop_only_variant_fires_at_first_forty_percent_bar() -> None:
    signals = _signals(KNIFE_LEGS, **VARIANTS["drop_only"])  # type: ignore[arg-type]
    assert len(signals) == 1
    sig = signals[0]
    assert sig.drop >= 0.40 and sig.t == sig.l2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_truncation_does_not_change_past_signals(seed: int) -> None:
    rng = np.random.default_rng(seed)
    closes = np.concatenate([path_from_legs(TWO_LEG_KNIFE, seed=seed),
                             50.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 400)))])
    bars = bars_from_closes(closes)
    params = DAILY_PARAMS.with_overrides(require_cci=False)
    full, _ = detect(build_context(bars, params), params)
    full_keys = [(s.t, s.l2, s.p1) for s in full]
    for cut in range(360, len(bars), 23):
        part, _ = detect(build_context(bars.iloc[:cut + 1], params), params)
        assert [(s.t, s.l2, s.p1) for s in part] == [k for k in full_keys if k[0] <= cut]


def test_data_cliff_flags_quiet_one_bar_collapse() -> None:
    close = np.array([10.0] * 30 + [6.0] + [6.0] * 5)
    volume = np.full(close.size, 1000.0)
    flags = data_cliffs(close, volume, DAILY_PARAMS)
    assert flags[30] and flags.sum() == 1
    volume[30] = 10_000.0
    assert not data_cliffs(close, volume, DAILY_PARAMS).any()


def test_fmp_jump_flag_counts_only_on_quiet_volume() -> None:
    close = np.array([10.0] * 30 + [5.0] + [5.0] * 5)
    jumps = np.zeros(close.size, dtype=bool)
    jumps[30] = True
    heavy = np.full(close.size, 1000.0)
    heavy[30] = 20_000.0  # a real -50% crash trades many times normal volume
    assert not data_cliffs(close, heavy, DAILY_PARAMS, jumps).any()
    light = np.full(close.size, 1000.0)
    light[30] = 2_000.0  # a missing 2:1 split: volume roughly doubles, no spike
    assert data_cliffs(close, light, DAILY_PARAMS, jumps)[30]


def test_data_cliff_flags_quiet_reorg_splice() -> None:
    close = np.full(60, 0.5)
    close[40:] = 40.0
    volume = np.full(60, 1e6)
    assert data_cliffs(close, volume * np.where(np.arange(60) == 40, 0.1, 1.0), DAILY_PARAMS)[40]
    assert not data_cliffs(close, volume * np.where(np.arange(60) == 40, 10.0, 1.0), DAILY_PARAMS)[40]


def test_ineligible_bars_never_signal() -> None:
    bars = knife_bars()
    ctx = build_context(bars, DAILY_PARAMS, eligible=np.zeros(len(bars), dtype=bool))
    assert detect(ctx, DAILY_PARAMS)[0] == []
