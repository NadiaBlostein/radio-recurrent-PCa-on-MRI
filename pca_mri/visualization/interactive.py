"""
pca_mri.visualization.interactive — interactive Plotly/ipywidgets explorer.

Functions
---------
plot_interactive_explorer(df)    Dropdown-driven KDE + stacked bar chart
                                 for any categorical × continuous pair.
"""

from __future__ import annotations

import pandas as pd
import ipywidgets as widgets
from IPython.display import display

from pca_mri.analysis.descriptive import _CATEGORICAL, _CONTINUOUS
from pca_mri.visualization.descriptive_plots import plot_kde, plot_category_bar


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
            plot_kde(df, cat, cont).show()
            plot_category_bar(df, cat).show()

    cat_dd.observe(lambda _: _update(cat_dd.value, cont_dd.value), names="value")
    cont_dd.observe(lambda _: _update(cat_dd.value, cont_dd.value), names="value")

    display(widgets.VBox([widgets.HBox([cat_dd, cont_dd]), out]))
    _update(cat_dd.value, cont_dd.value)
