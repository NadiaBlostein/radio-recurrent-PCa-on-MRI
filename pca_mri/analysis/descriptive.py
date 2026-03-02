"""
pca_mri.analysis.descriptive — descriptive statistics and Table 1.

Functions
---------
missingness_report(df)              Per-column missing-data summary.
table_one(df, groupby, cont_cols,
          cat_cols)                 Clinical Table 1 (median [IQR] / n (%)).
capra_summary(df)                   Summary of CAPRA score components.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Default continuous variables for Table 1
_DEFAULT_CONTINUOUS = [
    "tx-age",
    "tx-gleason_total",
    "tx-total_dose_prostate",
    "tx-d28_vol_d90",
    "tx-d28_vol_v100",
    "psa-val",
    "psa-nadir_02",
    "psa-nadir_05",
    "psa-capra_total",
    "psa-time_since_tx",
    "mri_1-psa",
    "mri_1-pirads_score",
    "mri_1-prostate_vol",
    "biopsy-positive_ratio",
    "time_to_recurrence_days",
    "psa_doubling_time_months",
]

# Default categorical variables for Table 1
_DEFAULT_CATEGORICAL = [
    "tx-type",
    "tx-t_stage",
    "tx-protocol",
    "tx-adt",
    "mri_1-result",
    "biopsy-result",
    "pet-result",
    "is_converter",
    "psa_doubled",
]

# CAPRA component columns
_CAPRA_COLS = [
    "psa-capra_psa",
    "psa-capra_gleason",
    "psa-capra_t_stage",
    "psa-capra_biopsy",
    "psa-capra_age",
    "psa-capra_total",
]


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing-data summary.

    Columns
    -------
    column       : column name
    n_missing    : count of NaN values
    pct_missing  : percentage of NaN values (0–100)
    dtype        : pandas dtype
    n_present    : count of non-NaN values

    The result is sorted by pct_missing descending.
    """
    n = len(df)
    records = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        records.append(
            {
                "column": col,
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n, 1) if n > 0 else 0.0,
                "n_present": n - n_missing,
                "dtype": str(df[col].dtype),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("pct_missing", ascending=False)
        .reset_index(drop=True)
    )


def _fmt_continuous(series: pd.Series) -> str:
    """Format a continuous variable as 'median [Q1–Q3]'."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "—"
    median = s.median()
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{median:.1f} [{q1:.1f}–{q3:.1f}]"


def _fmt_categorical(series: pd.Series) -> dict[str, str]:
    """Format a categorical variable as {value: 'n (%)'} dict."""
    n_total = series.notna().sum()
    counts = series.value_counts(dropna=True)
    if n_total == 0:
        return {}
    return {
        str(val): f"{cnt} ({100 * cnt / n_total:.1f}%)"
        for val, cnt in counts.items()
    }


def table_one(
    df: pd.DataFrame,
    groupby: str | None = None,
    cont_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate a clinical Table 1.

    Continuous variables are reported as median [IQR].
    Categorical variables are reported as n (%).

    Parameters
    ----------
    df:        Cleaned DataFrame.
    groupby:   Optional column to stratify by (e.g. ``"tx-type"`` for LDR vs
               HDR, or ``"mri_1-result"`` for recurrence vs no-recurrence).
               If None, the overall cohort is reported.
    cont_cols: Continuous variable column names to include. Defaults to
               ``_DEFAULT_CONTINUOUS`` (only columns present in df are used).
    cat_cols:  Categorical variable column names to include. Defaults to
               ``_DEFAULT_CATEGORICAL`` (only columns present in df are used).

    Returns
    -------
    DataFrame with columns: variable, [group columns or "Overall"].
    """
    cont_cols = [c for c in (cont_cols or _DEFAULT_CONTINUOUS) if c in df.columns]
    cat_cols = [c for c in (cat_cols or _DEFAULT_CATEGORICAL) if c in df.columns]

    if groupby and groupby in df.columns:
        groups = {str(k): v for k, v in df.groupby(groupby)}
    else:
        groups = {"Overall": df}

    group_keys = list(groups.keys())
    rows: list[dict] = []

    # Overall N
    rows.append({"variable": "N", **{k: str(len(v)) for k, v in groups.items()}})

    # Continuous variables
    for col in cont_cols:
        row = {"variable": col}
        for k, g in groups.items():
            row[k] = _fmt_continuous(g[col])
        rows.append(row)

    # Categorical variables
    for col in cat_cols:
        # Header row for this variable
        rows.append({"variable": col, **{k: "" for k in group_keys}})
        # Gather all unique values across groups
        all_vals = sorted(
            set(
                v
                for g in groups.values()
                for v in g[col].dropna().astype(str).unique()
            )
        )
        for val in all_vals:
            row = {"variable": f"  {val}"}
            for k, g in groups.items():
                n_total = g[col].notna().sum()
                cnt = (g[col].astype(str) == val).sum()
                pct = 100 * cnt / n_total if n_total > 0 else 0.0
                row[k] = f"{cnt} ({pct:.1f}%)"
            rows.append(row)

    return pd.DataFrame(rows)


def capra_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise CAPRA score components and total.

    Returns a DataFrame with each CAPRA column as a row and columns:
    variable, median, Q1, Q3, min, max, n_missing.
    """
    present = [c for c in _CAPRA_COLS if c in df.columns]
    records = []
    for col in present:
        s = pd.to_numeric(df[col], errors="coerce")
        records.append(
            {
                "variable": col,
                "median": round(s.median(), 2),
                "Q1": round(s.quantile(0.25), 2),
                "Q3": round(s.quantile(0.75), 2),
                "min": round(s.min(), 2),
                "max": round(s.max(), 2),
                "n_missing": int(s.isna().sum()),
            }
        )
    return pd.DataFrame(records)
