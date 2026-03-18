"""
pca_mri.visualization.interactive — interactive Plotly/ipywidgets explorer.

Functions
---------
plot_interactive_explorer(df)    Dropdown-driven KDE + stacked bar chart
                                 for any categorical × continuous pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from scipy.stats import gaussian_kde

from pca_mri.analysis.descriptive import _CATEGORICAL, _CONTINUOUS

_COLOR_SEQ = pc.qualitative.Plotly


def plot_interactive_explorer(df: pd.DataFrame) -> None:
    """Interactive KDE + stacked-bar explorer for the cleaned dataset.

    Displays two linked Plotly figures controlled by two dropdowns:

    * **Categorical** — one of the standard categorical variables (e.g.
      treatment type, biopsy result).  Drives the colour grouping.
    * **Continuous** — one of the standard continuous variables (e.g. age,
      PSA).  Drives the KDE x-axis.

    The upper panel shows a probability-density estimate (KDE) of the chosen
    continuous variable, stratified by the chosen categorical variable.
    The lower panel shows a stacked bar chart with absolute counts and
    percentages for the chosen categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned patient dataframe (output of ``load_clean()``).
    """
    cat_options  = [(label, col) for col, label in _CATEGORICAL if col in df.columns]
    cont_options = [(label, col) for col, label in _CONTINUOUS  if col in df.columns]

    cat_dd = widgets.Dropdown(
        options=cat_options,
        value=cat_options[0][1] if cat_options else None,
        description="Categorical:",
        style={"description_width": "initial"},
    )
    cont_dd = widgets.Dropdown(
        options=cont_options,
        value=cont_options[0][1] if cont_options else None,
        description="Continuous:",
        style={"description_width": "initial"},
    )

    out = widgets.Output()

    def _update(cat: str, cont: str) -> None:
        out.clear_output(wait=True)
        with out:
            cats   = df[cat].dropna().unique()
            colors = {c: _COLOR_SEQ[i % len(_COLOR_SEQ)] for i, c in enumerate(cats)}

            cont_label = next((lbl for col, lbl in _CONTINUOUS  if col == cont), cont)
            cat_label  = next((lbl for col, lbl in _CATEGORICAL if col == cat),  cat)

            # ── KDE panel ────────────────────────────────────────────────────
            fig1 = go.Figure()
            for cat_val in cats:
                vals = (
                    df.loc[df[cat] == cat_val, cont]
                    .pipe(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if len(vals) < 2:
                    continue
                kde    = gaussian_kde(vals, bw_method="scott")
                span   = vals.max() - vals.min() or 1
                x_grid = np.linspace(vals.min() - 0.1 * span, vals.max() + 0.1 * span, 300)
                fig1.add_trace(go.Scatter(
                    x=x_grid, y=kde(x_grid),
                    mode="lines",
                    name=str(cat_val),
                    line=dict(color=colors[cat_val], width=2),
                    hovertemplate=(
                        f"<b>{cat_val}</b><br>"
                        f"{cont_label}: %{{x:.2f}}<br>"
                        f"Density: %{{y:.4f}}<extra></extra>"
                    ),
                ))
            fig1.update_layout(
                title=f"PDF of <b>{cont_label}</b> stratified by <b>{cat_label}</b>",
                xaxis_title=cont_label,
                yaxis_title="Probability density",
                legend_title=cat_label,
                template="plotly_dark",
                height=420,
            )
            fig1.show()

            # ── Stacked bar panel ─────────────────────────────────────────────
            total = int(df[cat].notna().sum())
            fig2  = go.Figure()
            for cat_val in cats:
                count = int((df[cat] == cat_val).sum())
                pct   = count / total * 100 if total else 0
                fig2.add_trace(go.Bar(
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
            fig2.update_layout(
                barmode="stack",
                title=f"Distribution of <b>{cat_label}</b>",
                yaxis_title="Count",
                legend_title=cat_label,
                template="plotly_dark",
                height=380,
            )
            fig2.show()

    cat_dd.observe(lambda _: _update(cat_dd.value, cont_dd.value), names="value")
    cont_dd.observe(lambda _: _update(cat_dd.value, cont_dd.value), names="value")

    display(widgets.VBox([widgets.HBox([cat_dd, cont_dd]), out]))
    _update(cat_dd.value, cont_dd.value)
