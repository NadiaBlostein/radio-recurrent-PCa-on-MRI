"""
pca_mri.visualization.psa_kinetics — PSA trajectory and kinetics plots.

Functions
---------
plot_psa_trajectory(df, patient_id)   PSA over time (one or all patients).
plot_psa_doubling_time(df)            Histogram of PSA doubling time.
plot_psa_by_outcome(df)               Box plots of PSA at each MRI by outcome.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# (visit label, date column, PSA column)
_MRI_VISITS = [
    ("MRI 1", "mri_1-date", "mri_1-psa"),
    ("MRI 2", "mri_2-date", "mri_2-psa"),
    ("MRI 3", "mri_3-date", "mri_3-psa"),
    ("MRI 4", "mri_4-date", "mri_4-psa"),
]

_OUTCOME_COLOURS = {
    True: "#d62728",   # recurrence-positive — red
    False: "#2ca02c",  # recurrence-negative — green
    "positive": "#d62728",
    "positif": "#d62728",
    "negative": "#2ca02c",
    "négatif": "#2ca02c",
    "négative": "#2ca02c",
}


def _long_psa(df: pd.DataFrame, patient_ids: list | None = None) -> pd.DataFrame:
    """Melt the wide PSA / date columns into a long-format DataFrame."""
    rows: list[dict] = []
    if patient_ids is not None:
        subset = df[df["patient_id"].isin(patient_ids)]
    else:
        subset = df

    for _, row in subset.iterrows():
        pid = row.get("patient_id", "")
        for label, date_col, psa_col in _MRI_VISITS:
            if date_col not in df.columns or psa_col not in df.columns:
                continue
            date_val = row.get(date_col)
            psa_val = row.get(psa_col)
            if pd.isna(date_val) or pd.isna(psa_val):
                continue
            rows.append(
                {
                    "patient_id": pid,
                    "visit": label,
                    "date": pd.to_datetime(date_val, errors="coerce"),
                    "psa": pd.to_numeric(psa_val, errors="coerce"),
                    "recurrence": row.get("recurrence"),
                    "mri_1_result": row.get("mri_1-result"),
                }
            )
    return pd.DataFrame(rows)


def plot_psa_trajectory(
    df: pd.DataFrame,
    patient_id: int | str | None = None,
) -> go.Figure:
    """PSA over time for one patient or the full cohort (spaghetti plot).

    Parameters
    ----------
    df:          Cleaned DataFrame (should include ``mri_N-date`` and
                 ``mri_N-psa`` columns).
    patient_id:  If given, plot only that patient's trajectory.  If None, plot
                 all patients with a cohort mean overlay.

    Returns
    -------
    Plotly Figure.
    """
    if patient_id is not None:
        ids = [patient_id]
        title = f"PSA Trajectory — Patient {patient_id}"
    else:
        ids = None
        title = "PSA Trajectories — Full Cohort"

    long = _long_psa(df, ids)
    if long.empty:
        raise ValueError("No PSA data found. Check mri_N-date / mri_N-psa columns.")

    fig = go.Figure()

    if patient_id is not None:
        fig.add_trace(
            go.Scatter(
                x=long["date"],
                y=long["psa"],
                mode="lines+markers",
                name=str(patient_id),
                marker=dict(size=9),
                line=dict(width=2),
            )
        )
    else:
        for pid, group in long.groupby("patient_id"):
            group = group.sort_values("date")
            result = str(group["mri_1_result"].iloc[0]).strip().lower() if not group.empty else ""
            colour = _OUTCOME_COLOURS.get(result, "#aec7e8")
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["psa"],
                    mode="lines+markers",
                    name=str(pid),
                    line=dict(color=colour, width=1),
                    marker=dict(size=5, color=colour),
                    opacity=0.5,
                    showlegend=False,
                )
            )
        # Cohort mean per visit
        mean_per_visit = long.groupby("visit")["psa"].mean().reset_index()
        mean_dates = long.groupby("visit")["date"].median().reset_index()
        mean_df = mean_per_visit.merge(mean_dates, on="visit").sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=mean_df["date"],
                y=mean_df["psa"],
                mode="lines+markers",
                name="Cohort mean",
                line=dict(color="black", width=3, dash="dash"),
                marker=dict(size=10, color="black"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="PSA (ng/mL)",
        template="plotly_white",
    )
    return fig


def plot_psa_doubling_time(df: pd.DataFrame) -> go.Figure:
    """Histogram of PSA doubling time (``psa_doubling_time_months``).

    Requires the ``psa_doubling_time_months`` column added by
    ``features.add_psa_doubling_time()``.

    Returns
    -------
    Plotly Figure.
    """
    col = "psa_doubling_time_months"
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. Run features.add_psa_doubling_time() first."
        )

    data = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = px.histogram(
        data,
        nbins=20,
        labels={"value": "PSA Doubling Time (months)"},
        title="Distribution of PSA Doubling Time",
        template="plotly_white",
    )
    fig.update_layout(yaxis_title="Number of patients")
    return fig


def plot_psa_by_outcome(
    df: pd.DataFrame,
    outcome_col: str = "mri_1-result",
) -> go.Figure:
    """Box plots of PSA at each MRI visit, stratified by outcome.

    Parameters
    ----------
    df:          Cleaned DataFrame.
    outcome_col: Column used to colour groups (default: ``mri_1-result``).

    Returns
    -------
    Plotly Figure.
    """
    long = _long_psa(df)
    if long.empty:
        raise ValueError("No PSA data found.")

    # Merge outcome column into long df
    if outcome_col in df.columns:
        outcome_map = df.set_index("patient_id")[outcome_col].to_dict()
        long["outcome"] = long["patient_id"].map(outcome_map).fillna("Unknown")
        long["outcome"] = long["outcome"].str.strip().str.capitalize()
    else:
        long["outcome"] = "All"

    fig = px.box(
        long,
        x="visit",
        y="psa",
        color="outcome",
        points="all",
        labels={
            "visit": "MRI Visit",
            "psa": "PSA (ng/mL)",
            "outcome": outcome_col.replace("-", " ").replace("_", " ").title(),
        },
        title="PSA Values by MRI Visit and Outcome",
        color_discrete_map={
            "Positive": "#d62728",
            "Positif": "#d62728",
            "Negative": "#2ca02c",
            "Négative": "#2ca02c",
        },
        template="plotly_white",
        category_orders={"visit": [v for v, _, _ in _MRI_VISITS]},
    )
    return fig
