"""
pca_mri.visualization.timelines — time-to-event and MRI follow-up plots.

Functions
---------
plot_time_to_bf(df)           Histogram of days from treatment to biochemical failure.
plot_time_to_rec_mri(df)      Histogram of days from treatment to first positive MRI.
plot_bf_to_mri_lag(df)        Histogram of the lag between BF and MRI recurrence detection.
plot_mri_followup_count(df)   Bar chart of how many serial MRI visits each patient received.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_TX_COLORS: dict[str, str] = {
    "LDR": "#4C72B0",
    "HDR": "#DD8452",
    "RT":  "#55A868",
}
_TX_MAP: dict[str, str] = {
    "Curietherapie LDR": "LDR",
    "Curietherapie HDR": "HDR",
    "Radiotherapie":     "RT",
}

_MRI_RESULT_COLS = ["mri_1-result", "mri_2-result", "mri_3-result", "mri_4-result"]


def _add_tx_group(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_tx"] = work["tx-type"].map(_TX_MAP).fillna(work["tx-type"])
    return work


def _tx_legend(ax: plt.Axes) -> None:
    patches = [mpatches.Patch(color=c, label=g) for g, c in _TX_COLORS.items()]
    ax.legend(handles=patches, title="Treatment", framealpha=0.8)


def plot_time_to_bf(df: pd.DataFrame) -> plt.Figure:
    """Stacked histogram of days from treatment to biochemical failure (BF).

    Only patients who experienced BF are included (non-NaN ``bf-time_to_bf-days``).

    Returns
    -------
    matplotlib Figure
    """
    work = _add_tx_group(df)
    col = "bf-time_to_bf-days"
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work[col].notna()]

    groups = [g for g in ["LDR", "HDR", "RT"] if g in work["_tx"].values]
    data = [work.loc[work["_tx"] == g, col].values for g in groups]
    colors = [_TX_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, work[col].max() + 365, 365)  # yearly bins
    ax.hist(data, bins=bins, stacked=True, color=colors, edgecolor="#444", linewidth=0.5)

    ax.set_xlabel("Days from treatment to biochemical failure")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Time to biochemical failure (N={len(work)})")
    _tx_legend(ax)

    # Secondary x-axis in years
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / 365.25)
    ax2.set_xlabel("Years from treatment")

    fig.tight_layout()
    return fig


def plot_time_to_rec_mri(df: pd.DataFrame) -> plt.Figure:
    """Stacked histogram of days from treatment to first positive MRI.

    Only patients with a recorded positive MRI are included
    (non-NaN ``rec_mri-time_to_rec-days``).

    Returns
    -------
    matplotlib Figure
    """
    work = _add_tx_group(df)
    col = "rec_mri-time_to_rec-days"
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work[col].notna()]

    groups = [g for g in ["LDR", "HDR", "RT"] if g in work["_tx"].values]
    data = [work.loc[work["_tx"] == g, col].values for g in groups]
    colors = [_TX_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, work[col].max() + 365, 365)
    ax.hist(data, bins=bins, stacked=True, color=colors, edgecolor="#444", linewidth=0.5)

    ax.set_xlabel("Days from treatment to first positive MRI")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Time to MRI recurrence detection (N={len(work)})")
    _tx_legend(ax)

    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / 365.25)
    ax2.set_xlabel("Years from treatment")

    fig.tight_layout()
    return fig


def plot_bf_to_mri_lag(df: pd.DataFrame) -> plt.Figure:
    """Histogram of the interval between BF and MRI recurrence detection.

    ``bf_to_rec_mri-days`` sign convention (from features.py):
        Positive → BF detected first (MRI recurrence found later)
        Negative → MRI recurrence found first (BF recorded later)

    A vertical dashed line marks zero (simultaneous detection).

    Returns
    -------
    matplotlib Figure
    """
    col = "bf_to_rec_mri-days"
    vals = pd.to_numeric(df[col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(vals.min() - 90, vals.max() + 180, 180)
    ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="#444", linewidth=0.5)
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.4,
               label="Simultaneous (BF = MRI recurrence)")

    ax.set_xlabel("Days (positive = BF first; negative = MRI recurrence first)")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Lag between biochemical failure and MRI recurrence (N={len(vals)})")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_mri_followup_count(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of how many serial MRI visits each patient received.

    Counts the number of non-null MRI result columns (mri_1 through mri_4)
    per patient.

    Returns
    -------
    matplotlib Figure
    """
    result_cols = [c for c in _MRI_RESULT_COLS if c in df.columns]
    mri_counts = df[result_cols].notna().sum(axis=1).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(mri_counts.index.astype(str), mri_counts.values,
           color="#4C72B0", edgecolor="#444")

    for x, y in zip(mri_counts.index, mri_counts.values):
        ax.text(str(x), y + 0.3, str(y), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Number of MRI visits")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Serial MRI follow-up visits per patient (N={len(df)})")
    fig.tight_layout()
    return fig
