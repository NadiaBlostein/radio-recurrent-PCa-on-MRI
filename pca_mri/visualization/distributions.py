"""
pca_mri.visualization.distributions — cohort-level distribution plots.

Functions
---------
plot_pirads_distribution(df)    Bar chart of PIRADS scores at first MRI.
plot_outcome_distribution(df)   Bar chart of MRI and biopsy result breakdown.
plot_tx_type_distribution(df)   Bar / pie chart of treatment type breakdown.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_pirads_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of PIRADS scores at the first MRI (``mri_1-pirads_score``).

    Returns
    -------
    Plotly Figure.
    """
    col = "mri_1-pirads_score"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Run rename_columns() first.")

    counts = (
        pd.to_numeric(df[col], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["pirads_score", "n"]

    fig = px.bar(
        counts,
        x="pirads_score",
        y="n",
        labels={"pirads_score": "PI-RADS Score", "n": "Number of patients"},
        title="PI-RADS Score Distribution at First Surveillance MRI",
        color="pirads_score",
        color_continuous_scale="Blues",
        text="n",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        coloraxis_showscale=False,
        template="plotly_white",
    )
    return fig


def plot_outcome_distribution(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of MRI and biopsy result distributions.

    Shows counts for positive, negative, and missing/other results for both
    ``mri_1-result`` and ``biopsy-result``.

    Returns
    -------
    Plotly Figure.
    """
    result_cols = {
        "MRI 1": "mri_1-result",
        "Biopsy": "biopsy-result",
    }
    present = {label: col for label, col in result_cols.items() if col in df.columns}

    records: list[dict] = []
    for label, col in present.items():
        normed = df[col].str.strip().str.lower()
        pos = normed.isin({"positive", "positif", "positiv"}).sum()
        neg = normed.isin({"negative", "negativ", "négative", "négatif"}).sum()
        missing = df[col].isna().sum()
        other = len(df) - pos - neg - missing
        records.extend(
            [
                {"investigation": label, "result": "Positive", "n": int(pos)},
                {"investigation": label, "result": "Negative", "n": int(neg)},
                {"investigation": label, "result": "Missing", "n": int(missing)},
            ]
        )
        if other > 0:
            records.append({"investigation": label, "result": "Other", "n": int(other)})

    fig = px.bar(
        pd.DataFrame(records),
        x="investigation",
        y="n",
        color="result",
        barmode="group",
        labels={"investigation": "", "n": "Number of patients", "result": "Result"},
        title="MRI and Biopsy Outcome Distribution",
        color_discrete_map={
            "Positive": "#d62728",
            "Negative": "#2ca02c",
            "Missing": "#aec7e8",
            "Other": "#ffbb78",
        },
        text="n",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white")
    return fig


def plot_tx_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of treatment type breakdown (``tx-type``).

    Returns
    -------
    Plotly Figure.
    """
    col = "tx-type"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Run rename_columns() first.")

    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = ["tx_type", "n"]
    counts["tx_type"] = counts["tx_type"].fillna("Unknown")

    fig = px.pie(
        counts,
        names="tx_type",
        values="n",
        title="Treatment Type Distribution",
        hole=0.35,
    )
    fig.update_traces(textinfo="label+percent+value")
    fig.update_layout(template="plotly_white")
    return fig
