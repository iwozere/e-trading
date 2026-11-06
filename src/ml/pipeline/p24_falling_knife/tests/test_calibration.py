"""Tests for the P24 calibration pieces: plateau pick, grid tables, detection-key sharing, heatmap."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from src.ml.pipeline.p24_falling_knife.chart import plot_heatmap
from src.ml.pipeline.p24_falling_knife.config import DAILY, DAILY_PARAMS, WEEKLY
from src.ml.pipeline.p24_falling_knife.jobs.run_p24_calibrate import REFERENCES, grid_tables, pick_all
from src.ml.pipeline.p24_falling_knife.validation import clean_trades, plateau_pick


def _grid(values: dict) -> pd.DataFrame:
    rows = [{"a": a, "b": b, "median": values[(a, b)], "n": 200} for a, b in itertools.product((1, 2, 3), (1, 2, 3))]
    return pd.DataFrame(rows)


def test_plateau_pick_prefers_stable_cell_over_lone_spike() -> None:
    values = {(a, b): 0.05 for a, b in itertools.product((1, 2, 3), (1, 2, 3))}
    values[(3, 3)] = 0.20          # lone spike: neighbours far below it
    values[(2, 2)] = 0.055         # stable cell, slightly above its neighbours
    annotated, idx, plateau = plateau_pick(_grid(values), ["a", "b"])
    assert plateau
    assert idx is not None
    assert (annotated.loc[idx, "a"], annotated.loc[idx, "b"]) == (2, 2)
    spike = annotated[(annotated["a"] == 3) & (annotated["b"] == 3)].iloc[0]
    assert not spike["on_plateau"] and spike["neighbours"] == 2


def test_plateau_pick_respects_min_trades_and_falls_back() -> None:
    values = {(a, b): float(a * b) / 100 * (1 if (a + b) % 2 else -1) for a, b in itertools.product((1, 2, 3), (1, 2, 3))}
    grid = _grid(values)
    _, idx, plateau = plateau_pick(grid, ["a", "b"])
    assert idx is not None and not plateau
    grid["n"] = 10
    assert plateau_pick(grid, ["a", "b"])[1] is None


def test_clean_trades_drops_flagged_rows() -> None:
    trades = pd.DataFrame({"suspect_gap": [False, True, False], "suspect_in_trade": [False, False, True], "x": [1, 2, 3]})
    assert clean_trades(trades)["x"].tolist() == [1]


def test_detection_params_ignore_exit_fields() -> None:
    a = DAILY_PARAMS.with_overrides(use_shakeout_rule=False, hard_stop_pct=0.3, max_hold_bars=10)
    assert a.detection_params() == DAILY_PARAMS.detection_params()
    assert DAILY_PARAMS.with_overrides(w_narrow=0.4).detection_params() != DAILY_PARAMS.detection_params()


def test_grid_tables_share_names_and_size() -> None:
    tables = grid_tables((DAILY, WEEKLY))
    cells = [k for k in tables[DAILY] if k.startswith("c")]
    assert len(cells) == 3 * 2 * 3 * 3 * 2 * 2
    assert cells == [k for k in tables[WEEKLY] if k.startswith("c")]
    assert set(REFERENCES) <= set(tables[DAILY])
    assert tables[DAILY]["c000"]["max_bars_since_low"] != tables[WEEKLY]["c000"]["max_bars_since_low"]


def test_pick_all_and_heatmap(tmp_path: Path) -> None:
    rows = []
    for i, (w, c, shake) in enumerate(itertools.product((0.4, 0.6, 0.8), (15.0, 25.0, 40.0), (True, False))):
        rows.append({"timeframe": DAILY, "variant": f"c{i:03d}", "zz_pct": 0.25, "support_bounce_pct": 0.06, "w_narrow": w,
                     "cci_div_min": c, "max_bars_since_low": 5, "use_shakeout_rule": shake, "n": 150,
                     "median": 0.02 + 0.001 * w})
    cells, picks = pick_all(pd.DataFrame(rows), min_trades=100)
    assert set(picks) == {(DAILY, True), (DAILY, False)}
    assert int(cells["picked"].sum()) == 2
    table = cells[cells["use_shakeout_rule"]].pivot_table(index="cci_div_min", columns="w_narrow", values="median")
    path = plot_heatmap(table, tmp_path / "h.png", "test")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
