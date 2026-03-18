"""
pca_mri.visualization.descriptive_plots — standalone Plotly figures for
descriptive statistics (KDE density and stacked bar charts).

All functions return ``plotly.graph_objects.Figure`` instances that can be
displayed in a notebook (``fig.show()``) **or** exported to HTML via
``export_html.save_figure(fig, ...)``.

Functions
---------
plot_kde(df, cat_col, cont_col)     KDE density plot of a continuous variable
                                     stratified by a categorical variable.
plot_category_bar(df, cat_col)      Stacked bar chart of a categorical variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from pca_mri.analysis.descriptive import _CATEGORICAL, _CONTINUOUS

_COLOR_SEQ = pc.qualitative.Plotly


def _resolve_labels(
    cat_col: str, cont_col: str | None = None
) -> tuple[str, str | None]:
    """Return human-readable labels for the given column names."""
    cat_label = next((lbl for col, lbl in _CATEGORICAL if col == cat_col), cat_col)
    cont_label = (
        next((lbl for col, lbl in _CONTINUOUS if col == cont_col), cont_col)
        if cont_col is not None
        else None
    )
    return cat_label, cont_label


def _category_colors(cats) -> dict:
    """Assign a Plotly colour to each unique category value."""
    return {c: _COLOR_SEQ[i % len(_COLOR_SEQ)] for i, c in enumerate(cats)}


# ─── Public API ──────────────────────────────────────────────────────────────


def plot_kde(
    df: pd.DataFrame,
    cat_col: str,
    cont_col: str,
    *,
    height: int = 420,
) -> go.Figure:
    """KDE density plot of *cont_col* stratified by *cat_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned patient dataframe.
    cat_col : str
        Categorical column used for colour grouping.
    cont_col : str
        Continuous column plotted on the x-axis.
    height : int
        Figure height in pixels (default 420).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cat_label, cont_label = _resolve_labels(cat_col, cont_col)
    cats = df[cat_col].dropna().unique()
    colors = _category_colors(cats)

    fig = go.Figure()
    for cat_val in cats:
        vals = (
            df.loc[df[cat_col] == cat_val, cont_col]
            .pipe(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        span = vals.max() - vals.min() or 1
        x_grid = np.linspace(vals.min() - 0.1 * span, vals.max() + 0.1 * span, 300)
        fig.add_trace(go.Scatter(
            x=x_grid,
            y=kde(x_grid),
            mode="lines",
            name=str(cat_val),
            line=dict(color=colors[cat_val], width=2),
            hovertemplate=(
                f"<b>{cat_val}</b><br>"
                f"{cont_label}: %{{x:.2f}}<br>"
                f"Density: %{{y:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"PDF of <b>{cont_label}</b> stratified by <b>{cat_label}</b>",
        xaxis_title=cont_label,
        yaxis_title="Probability density",
        legend_title=cat_label,
        template="plotly_dark",
        height=height,
    )
    return fig


def plot_category_bar(
    df: pd.DataFrame,
    cat_col: str,
    *,
    height: int = 380,
) -> go.Figure:
    """Stacked bar chart showing the distribution of *cat_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned patient dataframe.
    cat_col : str
        Categorical column to visualise.
    height : int
        Figure height in pixels (default 380).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cat_label, _ = _resolve_labels(cat_col)
    cats = df[cat_col].dropna().unique()
    colors = _category_colors(cats)
    total = int(df[cat_col].notna().sum())

    fig = go.Figure()
    for cat_val in cats:
        count = int((df[cat_col] == cat_val).sum())
        pct = count / total * 100 if total else 0
        fig.add_trace(go.Bar(
            name=str(cat_val),
            x=[cat_label],
            y=[count],
            marker_color=colors[cat_val],
            text=f"{cat_val}<br>N={count} ({pct:.1f}%)",
            textposition="inside",
            hovertemplate=(
                f"<b>{cat_val}</b><br>"
                f"N = {count}<br>"
                f"{pct:.1f}% of {total}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        title=f"Distribution of <b>{cat_label}</b>",
        yaxis_title="Count",
        legend_title=cat_label,
        template="plotly_dark",
        height=height,
    )
    return fig
