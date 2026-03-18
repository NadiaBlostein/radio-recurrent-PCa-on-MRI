"""
pca_mri.visualization.kinetics — PSA kinetics plots.

Functions
---------
plot_psa_doubling_time(df)          Side-by-side boxplots of PSA difference and PSA-DT at recurrence MRI.
plot_psa_trajectory(df, patient_ids) Per-patient PSA spaghetti plot across serial MRI visits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_MRI_PSA_COLS  = ["mri_1-psa", "mri_2-psa", "mri_3-psa", "mri_4-psa"]
_MRI_DATE_COLS = ["mri_1-date", "mri_2-date", "mri_3-date", "mri_4-date"]

_POSITIVE_VALUES = {"positive", "positif", "positiv"}

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


def _finite(series: pd.Series) -> pd.Series:
    """Return a Series with ±inf replaced by NaN."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def plot_psa_doubling_time(df: pd.DataFrame) -> plt.Figure:
    """Side-by-side boxplots comparing PSA difference and PSA-DT at recurrence MRI.

    Extreme values (|PSA-DT| > 200 months) are excluded to keep the axes
    readable; the excluded count is noted in the title.

    Returns
    -------
    matplotlib Figure
    """
    psa_diff    = _finite(df["psa_diff-rec_mri-days"]).dropna()
    psa_dt_rec  = _finite(df["psa_dt-rec_mri-months"]).dropna()

    clip = 200  # months — exclude extreme outliers for readability
    n_excl_rec  = (psa_dt_rec.abs()  > clip).sum()
    psa_dt_rec  = psa_dt_rec[psa_dt_rec.abs()   <= clip]

    mean_diff = psa_diff.mean()
    mean_dt   = psa_dt_rec.mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: PSA difference
    axes[0].boxplot(
        [psa_diff.values],
        labels=[f"PSA difference\n(N={len(psa_diff)})"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": "#4C72B0", "alpha": 0.7},
    )
    axes[0].axhline(0, color="crimson", linestyle="--", linewidth=1)
    axes[0].set_ylabel("PSA difference (t2 − t1)")
    axes[0].set_title("PSA difference to recurrence MRI")

    # Right panel: PSA-DT
    axes[1].boxplot(
        [psa_dt_rec.values],
        labels=[f"PSA-DT to\nrecurrence MRI\n(N={len(psa_dt_rec)})"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": "#DD8452", "alpha": 0.7},
    )
    axes[1].axhline(0, color="crimson", linestyle="--", linewidth=1)
    axes[1].set_ylabel("PSA doubling time (months)")
    axes[1].set_title(f"PSA-DT to recurrence MRI\n(excluded |DT|>{clip} mo: {n_excl_rec})")

    fig.suptitle(
        f"PSA Kinetics at Recurrence MRI — "
        f"Mean PSA diff: {mean_diff:.2f}, Mean PSA-DT: {mean_dt:.1f} mo",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_psa_trajectory(
    df: pd.DataFrame,
    patient_ids: list | None = None,
    max_patients: int = 30,
) -> plt.Figure:
    """Per-patient PSA spaghetti plot across serial MRI visits.

    Each line represents one patient's PSA measured at each MRI visit.
    Lines are coloured by treatment type.  Patients with a positive MRI
    result at any visit are drawn with full opacity; others are drawn faint.

    Parameters
    ----------
    df:           Cleaned dataset.
    patient_ids:  Specific patient IDs to plot.  If None, up to
                  ``max_patients`` patients with ≥2 PSA measurements are shown.
    max_patients: Maximum number of patients when ``patient_ids`` is None.

    Returns
    -------
    matplotlib Figure
    """
    work = df.copy()
    work["_tx"] = work["tx-type"].map(_TX_MAP).fillna(work["tx-type"])

    # Gather per-patient PSA and date at each MRI visit
    psa_cols  = [c for c in _MRI_PSA_COLS  if c in work.columns]
    date_cols = [c for c in _MRI_DATE_COLS if c in work.columns]
    result_col_names = ["mri_1-result", "mri_2-result", "mri_3-result", "mri_4-result"]
    result_cols = [c for c in result_col_names if c in work.columns]

    # Select patients with ≥2 non-null PSA values
    has_enough = work[psa_cols].notna().sum(axis=1) >= 2
    pool = work[has_enough]

    if patient_ids is not None:
        pool = pool[pool["patient_id"].isin(patient_ids)]
    else:
        pool = pool.head(max_patients)

    tx_groups = pool["_tx"].unique()
    cmap = {g: _TX_COLORS.get(g, "#888888") for g in tx_groups}

    fig, ax = plt.subplots(figsize=(10, 5))

    for _, row in pool.iterrows():
        # Collect (days_from_tx, psa) pairs for this patient
        tx_date = pd.to_datetime(row.get("tx-date"), errors="coerce")
        pts: list[tuple[float, float]] = []
        for p_col, d_col in zip(psa_cols, date_cols):
            psa_val = pd.to_numeric(row.get(p_col), errors="coerce")
            mri_date = pd.to_datetime(row.get(d_col), errors="coerce")
            if pd.notna(psa_val) and pd.notna(mri_date) and pd.notna(tx_date):
                days = (mri_date - tx_date).days
                pts.append((days, psa_val))

        if len(pts) < 2:
            continue

        pts.sort()
        xs, ys = zip(*pts)
        color = cmap.get(row["_tx"], "#888888")

        # Check if patient ever had a positive MRI result
        results = [str(row.get(c, "")).strip().lower() for c in result_cols]
        has_positive = any(r in _POSITIVE_VALUES for r in results)

        alpha = 0.85 if has_positive else 0.25
        lw    = 1.4  if has_positive else 0.7
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw)

    # Legend: treatment colours
    import matplotlib.patches as mpatches
    tx_patches = [mpatches.Patch(color=c, label=g) for g, c in cmap.items() if g in pool["_tx"].values]
    style_patches = [
        plt.Line2D([0], [0], color="grey", lw=1.4, alpha=0.85, label="Positive MRI (any visit)"),
        plt.Line2D([0], [0], color="grey", lw=0.7, alpha=0.25, label="No positive MRI"),
    ]
    ax.legend(handles=tx_patches + style_patches, framealpha=0.8, fontsize=8)

    ax.set_xlabel("Days from treatment")
    ax.set_ylabel("PSA (ng/mL)")
    ax.set_title(f"PSA trajectories across serial MRI visits (N={len(pool)})")
    fig.tight_layout()
    return fig
