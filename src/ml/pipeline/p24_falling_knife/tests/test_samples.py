"""Tests for the P24 sample-chart selection and report section."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.pipeline.p24_falling_knife.backtest import FULL, TRAIN, UPPER_BOUND
from src.ml.pipeline.p24_falling_knife.report import samples_section
from src.ml.pipeline.p24_falling_knife.samples import sample_trades


def _trades() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append({"variant": FULL, "period": TRAIN, "timeframe": "daily", "signal_date": f"2022-01-{i + 1:02d}",
                     "suspect_gap": i == 0, "suspect_in_trade": False})
    for i in range(10):
        rows.append({"variant": FULL, "period": UPPER_BOUND, "timeframe": "daily", "signal_date": f"2010-01-{i + 1:02d}",
                     "suspect_gap": False, "suspect_in_trade": False})
    rows.append({"variant": "drop_only", "period": TRAIN, "timeframe": "daily", "signal_date": "2022-02-01",
                 "suspect_gap": False, "suspect_in_trade": False})
    return pd.DataFrame(rows)


def test_sample_trades_prefers_headline_and_skips_suspect() -> None:
    sample = sample_trades(_trades(), 8, seed=1)
    assert len(sample) == 8
    assert (sample["variant"] == FULL).all() and not sample["suspect_gap"].any()
    assert int((sample["period"] == TRAIN).sum()) == 5
    assert sample_trades(_trades(), 8, seed=1).equals(sample)


def test_samples_section_lists_images(tmp_path: Path) -> None:
    assert "No sample charts" in samples_section(tmp_path)
    folder = tmp_path / "charts" / "samples"
    folder.mkdir(parents=True)
    (folder / "daily_2022-01-03_ABC.png").write_bytes(b"x")
    assert "](charts/samples/daily_2022-01-03_ABC.png)" in samples_section(tmp_path)
