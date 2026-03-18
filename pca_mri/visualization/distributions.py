"""
pca_mri.visualization.distributions — baseline characteristic distributions.

Functions
---------
plot_age_distribution(df)      Histogram of patient age, coloured by tx-type.
plot_psa_distribution(df)      Histogram of baseline PSA at diagnosis.
plot_gleason_distribution(df)  Grouped bar chart of Gleason score by tx-type.
plot_t_stage_distribution(df)  Grouped bar chart of clinical T-stage by tx-type.
plot_capra_distribution(df)    Histogram of CAPRA total score with risk-band shading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Consistent colour palette for the three treatment types
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


def _add_tx_group(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_tx"] = work["tx-type"].map(_TX_MAP).fillna(work["tx-type"])
    return work


def _tx_legend(ax: plt.Axes) -> None:
    patches = [mpatches.Patch(color=c, label=g) for g, c in _TX_COLORS.items()]
    ax.legend(handles=patches, title="Treatment", framealpha=0.8)


def plot_age_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of patient age at treatment, stacked by treatment type.

    Returns
    -------
    matplotlib Figure
    """
    work = _add_tx_group(df)
    groups = [g for g in ["LDR", "HDR", "RT"] if g in work["_tx"].values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(work["tx-age"].min() - 0.5, work["tx-age"].max() + 1.5, 2)

    data = [work.loc[work["_tx"] == g, "tx-age"].dropna().values for g in groups]
    colors = [_TX_COLORS[g] for g in groups]
    ax.hist(data, bins=bins, stacked=True, color=colors, edgecolor="#444", linewidth=0.5)

    ax.set_xlabel("Age at treatment (years)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Age distribution at time of treatment")
    _tx_legend(ax)
    fig.tight_layout()
    return fig


def plot_psa_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of baseline PSA at diagnosis, log-scaled x-axis.

    Returns
    -------
    matplotlib Figure
    """
    work = _add_tx_group(df)
    psa = pd.to_numeric(work["psa-val"], errors="coerce")
    psa_pos = psa[psa > 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(psa_pos.min()), np.log10(psa_pos.max()), 25)

    groups = [g for g in ["LDR", "HDR", "RT"] if g in work["_tx"].values]
    data = [psa_pos[work.loc[psa_pos.index, "_tx"] == g].values for g in groups]
    colors = [_TX_COLORS[g] for g in groups]
    ax.hist(data, bins=bins, stacked=True, color=colors, edgecolor="#444", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("PSA at diagnosis (ng/mL, log scale)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Baseline PSA distribution at diagnosis")
    _tx_legend(ax)
    fig.tight_layout()
    return fig


def plot_gleason_distribution(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of Gleason total score by treatment type.

    Returns
    -------
    matplotlib Figure
    """
    work = _add_tx_group(df)
    gleason_order = sorted(work["tx-gleason_total"].dropna().unique())
    groups = [g for g in ["LDR", "HDR", "RT"] if g in work["_tx"].values]

    counts = (
        work.groupby(["tx-gleason_total", "_tx"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=gleason_order, columns=groups, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(gleason_order))
    width = 0.25
    offset = np.linspace(-(len(groups) - 1) / 2, (len(groups) - 1) / 2, len(groups)) * width

    for g, off in zip(groups, offset):
        ax.bar(x + off, counts[g], width, label=g, color=_TX_COLORS[g], edgecolor="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"GS {int(g)}" for g in gleason_order])
    ax.set_xlabel("Gleason score at diagnosis")
    ax.set_ylabel("Number of patients")
    ax.set_title("Gleason score distribution by treatment type")
    ax.legend(title="Treatment", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_t_stage_distribution(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of clinical T-stage by treatment type.

    Returns
    -------
    matplotlib Figure
    """
    work = _add_tx_group(df)
    stage_order = ["T1b", "T1c", "T2a", "T2b", "T3a"]
    stage_order = [s for s in stage_order if s in work["tx-t_stage"].values]
    groups = [g for g in ["LDR", "HDR", "RT"] if g in work["_tx"].values]

    counts = (
        work.groupby(["tx-t_stage", "_tx"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=stage_order, columns=groups, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stage_order))
    width = 0.25
    offset = np.linspace(-(len(groups) - 1) / 2, (len(groups) - 1) / 2, len(groups)) * width

    for g, off in zip(groups, offset):
        ax.bar(x + off, counts[g], width, label=g, color=_TX_COLORS[g], edgecolor="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_xlabel("Clinical T-stage")
    ax.set_ylabel("Number of patients")
    ax.set_title("T-stage distribution by treatment type")
    ax.legend(title="Treatment", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_capra_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of CAPRA total score with risk-group shading.

    Risk bands:
        Low (0–2):          green shade
        Intermediate (3–5): yellow shade
        High (6–10):        red shade

    Returns
    -------
    matplotlib Figure
    """
    capra = pd.to_numeric(df["psa-capra_total"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7, 4))

    # Risk-group background shading
    ax.axvspan(-0.5, 2.5,  alpha=0.15, color="green",  label="Low risk (0–2)")
    ax.axvspan(2.5,  5.5,  alpha=0.15, color="orange", label="Intermediate risk (3–5)")
    ax.axvspan(5.5,  10.5, alpha=0.15, color="red",    label="High risk (6–10)")

    bins = np.arange(capra.min() - 0.5, capra.max() + 1.5, 1)
    ax.hist(capra, bins=bins, color="#4C72B0", edgecolor="#444", linewidth=0.6, zorder=2)

    ax.set_xlabel("CAPRA score")
    ax.set_ylabel("Number of patients")
    ax.set_title("CAPRA score distribution")
    ax.set_xlim(-0.5, 10.5)
    ax.legend(framealpha=0.8, fontsize=8)
    fig.tight_layout()
    return fig
