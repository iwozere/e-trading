"""Tests for the P24 signal chart."""

from __future__ import annotations

from pathlib import Path

from src.ml.pipeline.p24_falling_knife.chart import plot_signal
from src.ml.pipeline.p24_falling_knife.config import DAILY_PARAMS, CostModel
from src.ml.pipeline.p24_falling_knife.detector import build_context, detect
from src.ml.pipeline.p24_falling_knife.exits import simulate_trade
from src.ml.pipeline.p24_falling_knife.tests.helpers import knife_bars


def test_plot_signal_writes_png(tmp_path: Path) -> None:
    ctx = build_context(knife_bars(), DAILY_PARAMS)
    signal = detect(ctx, DAILY_PARAMS)[0][0]
    trade = simulate_trade(ctx, signal, DAILY_PARAMS, CostModel())
    path = plot_signal(ctx, signal, tmp_path / "charts" / "knife.png", "KNF daily", trade)
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
