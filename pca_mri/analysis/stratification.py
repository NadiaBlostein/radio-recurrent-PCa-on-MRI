"""
pca_mri.analysis.stratification — dataset splits for subgroup analysis.

Functions
---------
split_by_tx_type(df)        Dict of DataFrames keyed by treatment type short label.
split_by_recurrence(df)     (recurrent_df, non_recurrent_df) based on biopsy or MRI.
"""

from __future__ import annotations

import pandas as pd

_TX_TYPE_SHORT: dict[str, str] = {
    "Curietherapie LDR": "LDR",
    "Curietherapie HDR": "HDR",
    "Radiotherapie":     "RT",
}

_POSITIVE_BIOPSY = {"positif", "positive", "positiv"}
_NEGATIVE_BIOPSY = {"négative", "négatif", "negative", "negativ"}


def split_by_tx_type(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the dataset by treatment type.

    Returns
    -------
    Dict with keys ``'LDR'``, ``'HDR'``, ``'RT'`` and corresponding
    DataFrames.  Rows with unrecognised ``tx-type`` values are excluded.
    """
    result: dict[str, pd.DataFrame] = {}
    for full_name, short in _TX_TYPE_SHORT.items():
        result[short] = df[df["tx-type"] == full_name].reset_index(drop=True)
    return result


def split_by_recurrence(
    df: pd.DataFrame,
    method: str = "biopsy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into recurrent and non-recurrent groups.

    Parameters
    ----------
    df:     Cleaned DataFrame (output of data-cleanup.ipynb).
    method: ``'biopsy'`` — use ``biopsy-result`` (Positif vs Négatif).
            ``'mri'``    — use ``rec_mri-index`` (not-NaN = recurrent).

    Returns
    -------
    ``(recurrent_df, non_recurrent_df)``

    Notes
    -----
    Patients without a biopsy result (NaN) are excluded from both groups
    when ``method='biopsy'``.
    """
    if method == "biopsy":
        norm = df["biopsy-result"].astype(str).str.strip().str.lower()
        is_recurrent = norm.isin(_POSITIVE_BIOPSY)
        is_non_recurrent = norm.isin(_NEGATIVE_BIOPSY)
    elif method == "mri":
        is_recurrent = df["rec_mri-index"].notna()
        is_non_recurrent = ~is_recurrent
    else:
        raise ValueError(f"method must be 'biopsy' or 'mri', got {method!r}")

    return (
        df[is_recurrent].reset_index(drop=True),
        df[is_non_recurrent].reset_index(drop=True),
    )
