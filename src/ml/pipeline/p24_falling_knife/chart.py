"""
P24 Falling Knife — signal chart (spec §9.3).

Four stacked panels on one x-axis: price as a line (the chapter prefers a line chart for spotting divergences), PPO
with signal line and histogram, ADX with +DI/−DI and the channel width ``W``, CCI with ±100/±200 guides. Points P1,
S3, L2, the entry E and the exit are marked on price; X on ADX, Y on PPO; the CCI troughs at S3 and L2 are joined by a line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.ml.pipeline.p24_falling_knife.detector import BarEval, SeriesContext  # noqa: E402
from src.ml.pipeline.p24_falling_knife.exits import Trade  # noqa: E402

DAILY_WINDOW = 180
WEEKLY_WINDOW = 104
AFTER_SIGNAL = 60


def plot_signal(ctx: SeriesContext, signal: BarEval, path: Path, title: str, trade: Optional[Trade] = None,
                window: int = DAILY_WINDOW) -> Path:
    """
    Draw one signal (and optionally its trade) to a PNG.

    Args:
        ctx: Series context of the security.
        signal: The signal bar.
        path: Output PNG path (parent folders are created).
        title: Chart title (ticker, timeframe, date).
        trade: Simulated trade to mark the exit, if any.
        window: Bars shown before the signal.

    Returns:
        The written path.
    """
    start = max(0, min(signal.p1, signal.t - window))
    end = min(ctx.n, (trade.exit_bar if trade else signal.t) + AFTER_SIGNAL)
    x = ctx.index[start:end]
    ratios = {"height_ratios": [3, 1.3, 1.3, 1.3]}
    fig, (ax_p, ax_ppo, ax_adx, ax_cci) = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw=ratios)
    ax_p.plot(x, ctx.close[start:end], color="#1f3b57", lw=1.3)
    ax_p.set_title(title)
    ax_p.set_ylabel("price")
    marks = [("P1", signal.p1, "#555555"), ("L2", signal.l2, "#b03a2e")]
    if signal.s3 is not None:
        marks.append(("S3", signal.s3, "#7d6608"))
    entry_bar = signal.t + 1 if signal.t + 1 < ctx.n else signal.t
    marks.append(("E", entry_bar, "#1e8449"))
    if trade is not None:
        marks.append(("Exit", trade.exit_bar, "#6c3483"))
    for label, bar, color in marks:
        if start <= bar < end:
            ax_p.scatter(ctx.index[bar], ctx.close[bar], color=color, zorder=3)
            ax_p.annotate(label, (ctx.index[bar], ctx.close[bar]), textcoords="offset points", xytext=(4, 6),
                          color=color, fontsize=9)

    ax_ppo.plot(x, ctx.ppo[start:end], color="#1f3b57", lw=1.1, label="PPO")
    ax_ppo.plot(x, ctx.ppo_signal[start:end], color="#c0392b", lw=0.9, label="signal")
    ax_ppo.bar(x, ctx.ppo_hist[start:end], color="#95a5a6", width=1.0, alpha=0.6)
    ax_ppo.axhline(0, color="#999999", lw=0.7)
    if signal.bar_y is not None and start <= signal.bar_y < end:
        ax_ppo.annotate("Y", (ctx.index[signal.bar_y], ctx.ppo[signal.bar_y]), color="#b03a2e", fontsize=9)
    ax_ppo.set_ylabel("PPO %")
    ax_ppo.legend(loc="upper left", fontsize=8)

    ax_adx.plot(x, ctx.adx[start:end], color="#1f3b57", lw=1.2, label="ADX")
    ax_adx.plot(x, ctx.plus_di[start:end], color="#27ae60", lw=0.6, alpha=0.6, label="+DI")
    ax_adx.plot(x, ctx.minus_di[start:end], color="#c0392b", lw=0.6, alpha=0.6, label="−DI")
    ax_adx.axhline(30, color="#999999", lw=0.7, ls="--")
    if signal.bar_x is not None and start <= signal.bar_x < end:
        ax_adx.annotate("X", (ctx.index[signal.bar_x], ctx.adx[signal.bar_x]), color="#b03a2e", fontsize=9)
    ax_w = ax_adx.twinx()
    ax_w.plot(x, ctx.w[start:end], color="#e67e22", lw=0.8, alpha=0.8, label="W")
    ax_w.set_ylabel("W", color="#e67e22")
    ax_adx.set_ylabel("ADX")
    ax_adx.legend(loc="upper left", fontsize=8)

    ax_cci.plot(x, ctx.cci[start:end], color="#1f3b57", lw=1.0)
    for level in (-200, -100, 100, 200):
        ax_cci.axhline(level, color="#bbbbbb", lw=0.6, ls=":")
    if signal.s3 is not None and start <= signal.s3 < end:
        ax_cci.plot([ctx.index[signal.s3], ctx.index[signal.l2]], [signal.cci_3, signal.cci_2], color="#b03a2e", lw=1.2)
    ax_cci.set_ylabel("CCI")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
