"""
pca_mri.analysis.stratification — cohort stratification helpers.

Addresses the analysis strategy from data-processing-TO-DOS.ipynb:
  - Stratify by treatment type (LDR vs HDR)
  - Stratify by recurrence status (positive vs negative)
  - Row-selection rules for analysis: keep initial positive MRI row for
    recurrence-positive patients; keep latest negative MRI row for
    recurrence-negative patients.

Functions
---------
split_by_tx_type(df)                    Dict of DataFrames by treatment type.
split_by_recurrence(df)                 Dict of DataFrames by recurrence outcome.
select_recurrence_positive_row(df)      One row per recurrence-positive patient.
select_recurrence_negative_row(df)      One row per recurrence-negative patient.
build_analysis_dataset(df)              Full analysis-ready dataset.
"""

from __future__ import annotations

import pandas as pd

_POSITIVE_VALUES = {"positive", "positif", "positiv"}
_NEGATIVE_VALUES = {"negative", "negativ", "négative", "négatif"}


def _result_is_positive(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower().isin(_POSITIVE_VALUES)


def _result_is_negative(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower().isin(_NEGATIVE_VALUES)


def split_by_tx_type(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the cohort by treatment type.

    Uses ``tx-type`` column. Returns a dict keyed by unique values found in
    that column (e.g. ``{"Curietherapie LDR": ..., "Curietherapie HDR": ...}``).
    An ``"Unknown"`` key collects rows where tx-type is NaN.

    Returns
    -------
    Dict mapping treatment-type string → subset DataFrame.
    """
    if "tx-type" not in df.columns:
        return {"All": df.copy()}

    result: dict[str, pd.DataFrame] = {}
    for val, group in df.groupby("tx-type", dropna=True):
        result[str(val)] = group.copy()

    missing = df[df["tx-type"].isna()]
    if not missing.empty:
        result["Unknown"] = missing.copy()

    return result


def split_by_recurrence(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the cohort by recurrence outcome.

    A patient is classified as:
      - ``"positive"``: ``mri_1-result`` is positive *or* ``biopsy-result`` is
        positive (at least one positive investigation).
      - ``"negative"``: all available MRI results and biopsy result are negative.
      - ``"indeterminate"``: result columns are all NaN / uninterpretable.

    Returns
    -------
    Dict with keys ``"positive"``, ``"negative"``, ``"indeterminate"``.
    """
    mri_col = "mri_1-result"
    biopsy_col = "biopsy-result"

    pos_mask = pd.Series(False, index=df.index)
    neg_mask = pd.Series(False, index=df.index)

    if mri_col in df.columns:
        pos_mask |= _result_is_positive(df[mri_col].fillna(""))
        neg_mask |= _result_is_negative(df[mri_col].fillna(""))

    if biopsy_col in df.columns:
        pos_mask |= _result_is_positive(df[biopsy_col].fillna(""))
        neg_mask_biopsy = _result_is_negative(df[biopsy_col].fillna(""))
        # Only count biopsy negative if MRI is also not positive
        neg_mask |= neg_mask_biopsy

    indet_mask = ~pos_mask & ~neg_mask

    return {
        "positive": df[pos_mask].copy(),
        "negative": df[neg_mask & ~pos_mask].copy(),
        "indeterminate": df[indet_mask].copy(),
    }


def select_recurrence_positive_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per recurrence-positive patient.

    Strategy (from TO-DOs):
      Keep the row corresponding to the patient's *initial positive MRI* and
      *initial positive biopsy*. Since the current dataset is one-row-per-patient
      (after deduplication), this simply returns all patients whose
      ``mri_1-result`` or ``biopsy-result`` is positive.

    If a future dataset has multiple rows per patient (e.g. one per MRI visit),
    this will correctly select the earliest positive MRI row.
    """
    mri_col = "mri_1-result"
    biopsy_col = "biopsy-result"

    pos_mask = pd.Series(False, index=df.index)
    if mri_col in df.columns:
        pos_mask |= _result_is_positive(df[mri_col].fillna(""))
    if biopsy_col in df.columns:
        pos_mask |= _result_is_positive(df[biopsy_col].fillna(""))

    result = df[pos_mask].copy()

    # If multiple rows exist per patient, keep earliest positive MRI date
    if "patient_id" in result.columns and result["patient_id"].duplicated().any():
        if "mri_1-date" in result.columns:
            result = (
                result.sort_values("mri_1-date")
                .groupby("patient_id", as_index=False)
                .first()
            )

    return result.reset_index(drop=True)


def select_recurrence_negative_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per recurrence-negative patient.

    Strategy (from TO-DOs):
      For patients with no recurrence, keep the data from their *latest
      negative MRI*.
    """
    mri_col = "mri_1-result"
    biopsy_col = "biopsy-result"

    pos_mask = pd.Series(False, index=df.index)
    neg_mask = pd.Series(False, index=df.index)

    if mri_col in df.columns:
        pos_mask |= _result_is_positive(df[mri_col].fillna(""))
        neg_mask |= _result_is_negative(df[mri_col].fillna(""))
    if biopsy_col in df.columns:
        pos_mask |= _result_is_positive(df[biopsy_col].fillna(""))
        neg_mask |= _result_is_negative(df[biopsy_col].fillna(""))

    pure_neg = df[neg_mask & ~pos_mask].copy()

    # If multiple rows per patient, keep the latest negative MRI date
    if "patient_id" in pure_neg.columns and pure_neg["patient_id"].duplicated().any():
        if "mri_1-date" in pure_neg.columns:
            pure_neg = (
                pure_neg.sort_values("mri_1-date")
                .groupby("patient_id", as_index=False)
                .last()
            )

    return pure_neg.reset_index(drop=True)


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Combine recurrence-positive and recurrence-negative rows into one analysis df.

    Adds a ``recurrence`` column (``True`` / ``False``) and concatenates the
    two subsets returned by ``select_recurrence_positive_row`` and
    ``select_recurrence_negative_row``.

    Returns
    -------
    Combined DataFrame sorted by patient_id.
    """
    pos_df = select_recurrence_positive_row(df)
    pos_df["recurrence"] = True

    neg_df = select_recurrence_negative_row(df)
    neg_df["recurrence"] = False

    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    if "patient_id" in combined.columns:
        combined = combined.sort_values("patient_id").reset_index(drop=True)
    return combined
