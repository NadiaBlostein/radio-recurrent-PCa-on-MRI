"""
pca_mri.visualization.timelines — longitudinal patient timelines.

Functions
---------
plot_patient_timeline(df, patient_id)   Gantt-style timeline for one patient.
plot_cohort_timelines(df, max_patients) Stacked timeline for many patients.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Events to plot in order, as (label, date_col, result_col_or_None)
_TIMELINE_EVENTS = [
    ("Treatment", "tx-date", None),
    ("MRI 1", "mri_1-date", "mri_1-result"),
    ("MRI 2", "mri_2-date", "mri_2-result"),
    ("MRI 3", "mri_3-date", "mri_3-result"),
    ("MRI 4", "mri_4-date", "mri_4-result"),
    ("Biopsy", "biopsy-date", "biopsy-result"),
    ("PET", "pet-date", "pet-result"),
    ("Biochemical failure", "bf-date", None),
    ("Death", "date_death", None),
]

_RESULT_COLOURS = {
    "positive": "#d62728",
    "positif": "#d62728",
    "positiv": "#d62728",
    "negative": "#2ca02c",
    "negativ": "#2ca02c",
    "négative": "#2ca02c",
    "négatif": "#2ca02c",
}
_DEFAULT_COLOUR = "#1f77b4"
_TX_COLOUR = "#ff7f0e"
_SPECIAL_COLOUR = "#9467bd"  # BF, death


def _event_colour(label: str, result) -> str:
    if label == "Treatment":
        return _TX_COLOUR
    if label in {"Biochemical failure", "Death"}:
        return _SPECIAL_COLOUR
    if pd.notna(result):
        return _RESULT_COLOURS.get(str(result).strip().lower(), _DEFAULT_COLOUR)
    return _DEFAULT_COLOUR


def plot_patient_timeline(
    df: pd.DataFrame,
    patient_id: int | str,
) -> go.Figure:
    """Horizontal timeline for a single patient.

    Shows all dated events (treatment, MRI visits, biopsy, PET, biochemical
    failure, death) as markers on a time axis.  MRI and biopsy events are
    colour-coded by result (red = positive, green = negative).

    Parameters
    ----------
    df:          Cleaned, one-row-per-patient DataFrame.
    patient_id:  Value matching the ``patient_id`` column.

    Returns
    -------
    Plotly Figure.
    """
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise ValueError(f"Patient {patient_id!r} not found in DataFrame.")
    row = row.iloc[0]

    fig = go.Figure()

    for label, date_col, result_col in _TIMELINE_EVENTS:
        if date_col not in df.columns:
            continue
        date_val = row.get(date_col)
        if pd.isna(date_val):
            continue
        result_val = row.get(result_col) if result_col else None
        colour = _event_colour(label, result_val)
        hover = f"<b>{label}</b><br>Date: {date_val}"
        if pd.notna(result_val):
            hover += f"<br>Result: {result_val}"

        # MRI PSA annotation
        psa_col = date_col.replace("-date", "-psa")
        if psa_col in df.columns and pd.notna(row.get(psa_col)):
            hover += f"<br>PSA: {row[psa_col]:.2f}"

        fig.add_trace(
            go.Scatter(
                x=[date_val],
                y=[label],
                mode="markers",
                marker=dict(size=14, color=colour),
                hovertemplate=hover + "<extra></extra>",
                name=label,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Patient {patient_id} — Clinical Timeline",
        xaxis_title="Date",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=400,
        template="plotly_white",
    )
    return fig


def plot_cohort_timelines(
    df: pd.DataFrame,
    max_patients: int | None = None,
    sort_by: str = "tx-date",
) -> go.Figure:
    """Stacked horizontal timelines for the whole cohort (or a subset).

    Each row in the figure is one patient.  Events are shown as symbols;
    MRI / biopsy results drive the colour.

    Parameters
    ----------
    df:           Cleaned, one-row-per-patient DataFrame.
    max_patients: If provided, only the first *max_patients* rows are plotted
                  (after sorting by *sort_by*).
    sort_by:      Column to sort patients by before plotting (default: tx-date).

    Returns
    -------
    Plotly Figure.
    """
    if sort_by in df.columns:
        df = df.sort_values(sort_by)
    if max_patients is not None:
        df = df.head(max_patients)

    if "patient_id" not in df.columns:
        patient_labels = [str(i) for i in df.index]
    else:
        patient_labels = df["patient_id"].astype(str).tolist()

    fig = go.Figure()
    seen_labels: set[str] = set()

    for (_, row), pid in zip(df.iterrows(), patient_labels):
        for label, date_col, result_col in _TIMELINE_EVENTS:
            if date_col not in df.columns:
                continue
            date_val = row.get(date_col)
            if pd.isna(date_val):
                continue
            result_val = row.get(result_col) if result_col else None
            colour = _event_colour(label, result_val)
            show_legend = label not in seen_labels
            seen_labels.add(label)

            hover = f"<b>Patient {pid}</b><br>{label}: {date_val}"
            if pd.notna(result_val):
                hover += f"<br>Result: {result_val}"

            fig.add_trace(
                go.Scatter(
                    x=[date_val],
                    y=[pid],
                    mode="markers",
                    marker=dict(size=8, color=colour),
                    hovertemplate=hover + "<extra></extra>",
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                )
            )

    fig.update_layout(
        title="Cohort Clinical Timelines",
        xaxis_title="Date",
        yaxis_title="Patient ID",
        yaxis=dict(autorange="reversed"),
        height=max(400, 18 * len(df)),
        template="plotly_white",
    )
    return fig
