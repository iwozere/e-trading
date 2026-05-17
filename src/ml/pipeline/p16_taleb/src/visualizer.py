"""
Plotly chart library for the P16 Taleb barbell pipeline.

All charts are dark-mode compatible and return go.Figure objects that can
be displayed in Jupyter, saved to HTML, or exported to PNG via kaleido.

Crisis events used for annotations:
    2011-08-08  US Debt Downgrade
    2015-08-24  China Flash Crash
    2018-12-24  Q4 Selloff
    2020-03-23  COVID-19 Low
    2022-06-16  Fed Hike Cycle Peak
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

_logger = logging.getLogger(__name__)

_TEMPLATE = "plotly_dark"

_CRISIS_EVENTS: dict = {
    "2011-08-08": "US Downgrade",
    "2015-08-24": "China Crash",
    "2018-12-24": "Q4 Selloff",
    "2020-03-23": "COVID Low",
    "2022-06-16": "Fed Peak",
}

_PALETTE = [
    "#00B4D8", "#90E0EF", "#F77F00", "#FCBF49",
    "#E63946", "#A8DADC", "#457B9D", "#1D3557",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_chart(fig: go.Figure, name: str, charts_dir: Path) -> None:
    """
    Save a figure to HTML and optionally PNG.

    Args:
        fig:        Plotly figure to save.
        name:       Base filename (without extension).
        charts_dir: Output directory; created if it does not exist.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    html_path = charts_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    _logger.info("Saved chart → %s", html_path)
    try:
        png_path = charts_dir / f"{name}.png"
        fig.write_image(str(png_path), width=1400, height=700)
        _logger.info("Saved PNG → %s", png_path)
    except Exception:
        _logger.debug("PNG export failed (kaleido not installed?)")


def _add_crisis_lines(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Add vertical lines and annotations at standard crisis dates."""
    for date_str, label in _CRISIS_EVENTS.items():
        fig.add_vline(
            x=date_str,
            line_dash="dot",
            line_color="rgba(255,100,100,0.5)",
            line_width=1,
            row=row, col=col,  # type: ignore[call-arg]
        )
        fig.add_annotation(
            x=date_str,
            y=1.0,
            yref="paper",
            text=label,
            showarrow=False,
            font={"size": 9, "color": "#ff6464"},
            textangle=-90,
            xanchor="right",
            row=row, col=col,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Chart 1: S&P 500 Price + Drawdown (dual axis) + VIX subplot
# ---------------------------------------------------------------------------

def chart_sp500_drawdown(df: pd.DataFrame) -> go.Figure:
    """
    S&P 500 close price with drawdown and VIX subplot.

    Args:
        df: Master DataFrame with columns: close, drawdown, vix.

    Returns:
        go.Figure with two rows: price+drawdown above, VIX below.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # SP500 price
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["close"],
            name="S&P 500 (SPY)",
            line={"color": _PALETTE[0], "width": 1.5},
        ),
        row=1, col=1, secondary_y=False,
    )

    # Drawdown as filled area
    draw = df["drawdown"] * 100
    deep_mask = draw < -10
    fig.add_trace(
        go.Scatter(
            x=df.index, y=draw,
            name="Drawdown %",
            fill="tozeroy",
            line={"width": 0},
            fillcolor="rgba(231,76,60,0.35)",
            marker={"color": np.where(deep_mask, "#e74c3c", "#f39c12")},  # type: ignore[arg-type]
        ),
        row=1, col=1, secondary_y=True,
    )

    # Threshold line
    fig.add_hline(y=-10, line_dash="dash", line_color="#e74c3c", line_width=1, row=1, col=1,  # type: ignore[call-arg]
                  secondary_y=True)

    # VIX
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["vix"],
            name="VIX",
            line={"color": _PALETTE[2], "width": 1.2},
        ),
        row=2, col=1,
    )
    fig.add_hline(y=30, line_dash="dash", line_color="#e74c3c", line_width=1, row=2, col=1)  # type: ignore[call-arg]

    _add_crisis_lines(fig, row=1, col=1)

    fig.update_layout(
        template=_TEMPLATE,
        title="S&P 500 — Price, Drawdown & VIX",
        height=650,
        legend={"orientation": "h", "y": -0.08},
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Strike Optimization Heatmap
# ---------------------------------------------------------------------------

def chart_heatmap(opt_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of net_roi_pct and crisis_capture_rate across the strike grid.

    Args:
        opt_df: Output of optimize_strikes().

    Returns:
        go.Figure with a Heatmap trace and metric dropdown.
    """
    metrics = {
        "net_roi_pct": "Net ROI (%)",
        "crisis_capture_rate": "Crisis Capture Rate (%)",
        "win_rate_pct": "Win Rate (%)",
        "sharpe_analog": "Sharpe Analog",
        "payoff_to_cost_ratio": "Payoff / Cost",
    }
    T_vals = sorted(opt_df["T_days"].unique())
    otm_vals = sorted(opt_df["strike_otm_pct"].unique())

    traces = []
    buttons = []
    for i, (metric_col, metric_label) in enumerate(metrics.items()):
        pivot = opt_df.pivot_table(
            index="T_days", columns="strike_otm_pct", values=metric_col
        )
        pivot = pivot.reindex(index=T_vals, columns=otm_vals)
        traces.append(
            go.Heatmap(
                x=[f"{v:.0f}%" for v in pivot.columns],
                y=[f"{t}d" for t in pivot.index],
                z=pivot.values.tolist(),
                colorscale="RdYlGn",
                colorbar={"title": metric_label},
                visible=(i == 0),
                hovertemplate=(
                    f"OTM: %{{x}}<br>Tenor: %{{y}}<br>{metric_label}: %{{z:.2f}}<extra></extra>"
                ),
            )
        )
        visibility = [j == i for j in range(len(metrics))]
        buttons.append({
            "label": metric_label,
            "method": "update",
            "args": [{"visible": visibility}, {"title": f"Strike Optimization — {metric_label}"}],
        })

    fig = go.Figure(data=traces)
    fig.update_layout(
        template=_TEMPLATE,
        title="Strike Optimization — Net ROI (%)",
        height=450,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.01,
            "xanchor": "left",
            "y": 1.18,
        }],
        xaxis_title="OTM %",
        yaxis_title="Tenor",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Cumulative P&L Comparison
# ---------------------------------------------------------------------------

def chart_cumulative_pnl(
    results: dict,
    df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Multi-line cumulative P&L, one trace per simulated strike.

    Args:
        results: Dict mapping label (str) → simulation DataFrame (output of simulate_barbell).
        df:      Optional master DataFrame for shading drawdown periods > 10%.

    Returns:
        go.Figure with labelled P&L lines and optional drawdown shading.
    """
    fig = go.Figure()

    # Drawdown shading
    if df is not None and "drawdown" in df.columns:
        in_drawdown = False
        band_start = None
        for date, dd in df["drawdown"].items():
            if dd < -0.10 and not in_drawdown:
                in_drawdown = True
                band_start = date
            elif dd >= -0.10 and in_drawdown:
                fig.add_vrect(
                    x0=str(band_start), x1=str(date),
                    fillcolor="rgba(231,76,60,0.12)",
                    line_width=0,
                )
                in_drawdown = False

    for i, (label, sim) in enumerate(results.items()):
        if sim.empty:
            continue
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=sim.index,
            y=sim["cum_pnl"],
            name=label,
            line={"color": color, "width": 1.8},
        ))
        # Annotate spike payoffs (top 3 payoffs)
        top_idx = sim["payoff"].nlargest(3).index
        for date in top_idx:
            payoff_val = sim.loc[date, "payoff"]
            if payoff_val > 0:
                fig.add_annotation(
                    x=date,
                    y=float(sim.loc[date, "cum_pnl"]),
                    text=f"+${payoff_val:,.0f}",
                    showarrow=True,
                    arrowhead=2,
                    font={"size": 9, "color": color},
                )

    _add_crisis_lines(fig)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        template=_TEMPLATE,
        title="Cumulative P&L by Strike",
        xaxis_title="Date",
        yaxis_title="Cumulative Net P&L ($)",
        height=550,
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Payoff Distribution
# ---------------------------------------------------------------------------

def chart_payoff_distribution(sim: pd.DataFrame) -> go.Figure:
    """
    Histogram of per-period P&L with normal distribution overlay.

    Args:
        sim: Output of simulate_barbell().

    Returns:
        go.Figure showing the fat-tail distribution of P&L.
    """
    pnl = sim["pnl"].dropna()
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=pnl,
        nbinsx=60,
        name="P&L per Period",
        marker_color=_PALETTE[0],
        opacity=0.75,
        histnorm="probability density",
    ))

    # Normal overlay
    x_range = np.linspace(pnl.min(), pnl.max(), 300)
    normal_y = norm.pdf(x_range, pnl.mean(), pnl.std())
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_y,
        name="Normal fit",
        line={"color": _PALETTE[2], "dash": "dash", "width": 1.5},
    ))

    mean_loss = float(pnl[pnl < 0].mean())
    max_gain = float(pnl.max())
    fig.add_vline(x=mean_loss, line_dash="dot", line_color="#e74c3c", line_width=1,
                  annotation_text=f"Avg bleed: ${mean_loss:,.0f}")
    fig.add_vline(x=max_gain, line_dash="dot", line_color="#2ecc71", line_width=1,
                  annotation_text=f"Max gain: ${max_gain:,.0f}")

    fig.update_layout(
        template=_TEMPLATE,
        title="Per-Period P&L Distribution",
        xaxis_title="P&L ($)",
        yaxis_title="Probability Density",
        height=450,
        bargap=0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 5: Premium Bleed vs Payoff
# ---------------------------------------------------------------------------

def chart_premium_bleed(sim: pd.DataFrame, window: int = 12) -> go.Figure:
    """
    Rolling 12-month total premium paid vs payoff received.

    Args:
        sim:    Output of simulate_barbell().
        window: Rolling window in rebalance periods (default 12 ≈ 1 year).

    Returns:
        go.Figure with two area traces.
    """
    roll_cost = sim["budget_spent"].rolling(window).sum()
    roll_payoff = sim["payoff"].rolling(window).sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim.index, y=roll_cost,
        name=f"Rolling {window}-period Cost",
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.25)",
        line={"color": "#e74c3c", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=sim.index, y=roll_payoff,
        name=f"Rolling {window}-period Payoff",
        fill="tozeroy",
        fillcolor="rgba(46,204,113,0.25)",
        line={"color": "#2ecc71", "width": 1.5},
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title=f"Premium Bleed vs Payoff (rolling {window}-period)",
        xaxis_title="Date",
        yaxis_title="Total ($)",
        height=450,
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 6: VIX vs Option Cost Scatter
# ---------------------------------------------------------------------------

def chart_vix_vs_cost(sim: pd.DataFrame) -> go.Figure:
    """
    Scatter: VIX level on purchase date vs put price as % of S.

    Colour encodes OTM percentage to show how the skew premium varies.

    Args:
        sim: Output of simulate_barbell() — or a merged frame from multiple runs
             with columns: vix, put_price_pct_of_S, otm_pct.

    Returns:
        go.Figure scatter plot.
    """
    fig = go.Figure()

    if "otm_pct" in sim.columns:
        groups = sim.groupby("otm_pct")
    else:
        groups = [(sim.get("moneyness", pd.Series([0.85])).iloc[0], sim)]

    for i, (otm_val, grp) in enumerate(groups):
        fig.add_trace(go.Scatter(
            x=grp["vix"],
            y=grp["put_price_pct_of_S"],
            mode="markers",
            name=f"{otm_val:.0f}% OTM",
            marker={
                "size": 5,
                "color": _PALETTE[i % len(_PALETTE)],
                "opacity": 0.7,
            },
            hovertemplate=(
                "VIX: %{x:.1f}<br>"
                "Put cost: %{y:.2f}% of S<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=_TEMPLATE,
        title="VIX Level vs Put Cost (% of Spot)",
        xaxis_title="VIX on Purchase Date",
        yaxis_title="Put Price (% of S)",
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 7: GDELT Tone vs Crisis Events (optional)
# ---------------------------------------------------------------------------

def chart_gdelt_tone(df: pd.DataFrame) -> go.Figure:
    """
    GDELT daily average tone and rolling 5-day MA vs S&P 500 drawdown.

    Args:
        df: Master DataFrame with columns: avgtone, gdelt_tone_ma5, drawdown.

    Returns:
        go.Figure with dual y-axes. Annotates the GDELT start date.
    """
    if "avgtone" not in df.columns or df["avgtone"].isna().all():
        fig = go.Figure()
        fig.add_annotation(
            text="GDELT data not available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 18, "color": "gray"},
        )
        fig.update_layout(template=_TEMPLATE, title="GDELT Tone vs Drawdown", height=450)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    gdelt_start = df["avgtone"].first_valid_index()

    # GDELT tone
    fig.add_trace(go.Scatter(
        x=df.index, y=df["avgtone"],
        name="GDELT AvgTone",
        line={"color": _PALETTE[1], "width": 0.8},
        opacity=0.4,
    ), secondary_y=False)

    if "gdelt_tone_ma5" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["gdelt_tone_ma5"],
            name="GDELT Tone MA5",
            line={"color": _PALETTE[0], "width": 1.5},
        ), secondary_y=False)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=df.index, y=df["drawdown"] * 100,
        name="S&P 500 Drawdown %",
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.2)",
        line={"color": "#e74c3c", "width": 1.2},
    ), secondary_y=True)

    if gdelt_start is not None:
        fig.add_vline(x=str(gdelt_start), line_dash="dash", line_color="gray",
                      annotation_text="GDELT v2 Start", annotation_position="top right")

    fig.update_layout(
        template=_TEMPLATE,
        title="GDELT Global Tone vs S&P 500 Drawdown",
        height=480,
        legend={"orientation": "h", "y": -0.12},
    )
    fig.update_yaxes(title_text="GDELT AvgTone", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Chart 8: Pareto Frontier (Win Rate vs ROI)
# ---------------------------------------------------------------------------

def chart_pareto(opt_df: pd.DataFrame) -> go.Figure:
    """
    Scatter of (win_rate_pct, net_roi_pct) for all grid combinations.

    Highlights the Pareto-efficient frontier (non-dominated points).

    Args:
        opt_df: Output of optimize_strikes().

    Returns:
        go.Figure with Pareto frontier highlighted in gold.
    """
    x = opt_df["win_rate_pct"].values
    y = opt_df["net_roi_pct"].values
    labels = opt_df.apply(
        lambda r: f"{r['strike_otm_pct']:.0f}% OTM, {r['T_days']:.0f}d", axis=1
    )

    # Identify Pareto-efficient points (maximise both axes)
    is_pareto = _pareto_mask(x, y)

    fig = go.Figure()

    # All points
    fig.add_trace(go.Scatter(
        x=x[~is_pareto], y=y[~is_pareto],
        mode="markers",
        text=labels[~is_pareto],
        name="Grid Points",
        marker={"size": 7, "color": _PALETTE[4], "opacity": 0.6},
        hovertemplate="%{text}<br>Win Rate: %{x:.1f}%<br>ROI: %{y:.1f}%<extra></extra>",
    ))

    # Pareto frontier
    pf_x = x[is_pareto]
    pf_y = y[is_pareto]
    sort_idx = np.argsort(pf_x)
    fig.add_trace(go.Scatter(
        x=pf_x[sort_idx], y=pf_y[sort_idx],
        mode="markers+lines",
        text=labels[is_pareto].values[sort_idx],
        name="Pareto Frontier",
        marker={"size": 10, "color": "#FFD700", "symbol": "star"},
        line={"color": "#FFD700", "width": 2},
        hovertemplate="%{text}<br>Win Rate: %{x:.1f}%<br>ROI: %{y:.1f}%<extra></extra>",
    ))

    med_win = float(np.median(x))
    med_roi = float(np.median(y))
    fig.add_vline(x=med_win, line_dash="dash", line_color="gray", line_width=1,
                  annotation_text=f"Median WR {med_win:.1f}%")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1,
                  annotation_text="Break-even")
    fig.add_hline(y=med_roi, line_dash="dash", line_color="#aaa", line_width=1,
                  annotation_text=f"Median ROI {med_roi:.1f}%")

    fig.update_layout(
        template=_TEMPLATE,
        title="Pareto Frontier — Win Rate vs Net ROI",
        xaxis_title="Win Rate (%)",
        yaxis_title="Net ROI (% of premium spent)",
        height=500,
    )
    return fig


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points (maximise both x and y)."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                dominated[i] = True
                break
    return ~dominated
