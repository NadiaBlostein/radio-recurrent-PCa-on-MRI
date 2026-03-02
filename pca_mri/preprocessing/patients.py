"""
pca_mri.preprocessing.patients — patient-level cleaning and flagging.

Addresses TO-DOs from data-processing-TO-DOS.ipynb:
  - Duplicate patients: 2060194, 2054846, 2205490
  - Converter patients (positive → negative MRI): 7049838, 7051280, 5519879
  - Weird patients (PSA roughly doubles between MRI visits): 2102593, 2205490, 2162006

Functions
---------
flag_duplicate_patients(df)             Add is_duplicate column.
resolve_duplicate_patients(df, strategy) Collapse duplicate rows per patient.
flag_converter_patients(df)             Add is_converter column.
flag_weird_patients(df, threshold)      Add psa_doubled column.
"""

from __future__ import annotations

import pandas as pd

# MRI result columns in visit order (must exist after rename_columns())
_MRI_RESULT_COLS = ["mri_1-result", "mri_2-result", "mri_3-result", "mri_4-result"]

# MRI PSA columns in visit order
_MRI_PSA_COLS = ["mri_1-psa", "mri_2-psa", "mri_3-psa", "mri_4-psa"]

_POSITIVE_VALUES = {"positive", "positif", "positiv"}
_NEGATIVE_VALUES = {"negative", "negativ", "négative", "négatif"}


def _normalise_result(val) -> str | None:
    """Lowercase and strip a result string, return None if NaN."""
    if pd.isna(val):
        return None
    return str(val).strip().lower()


def flag_duplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``is_duplicate`` boolean column.

    A row is marked True if its ``patient_id`` appears more than once in the
    dataset. All rows for a duplicated patient are flagged (not just the later
    ones), so you can inspect both before resolving.

    Returns
    -------
    df with ``is_duplicate`` column added (does not modify in place).
    """
    df = df.copy()
    df["is_duplicate"] = df["patient_id"].duplicated(keep=False)
    return df


def resolve_duplicate_patients(
    df: pd.DataFrame,
    strategy: str = "keep_latest_positive",
) -> pd.DataFrame:
    """Collapse duplicate patient rows to one row per patient.

    Parameters
    ----------
    df:       DataFrame with ``is_duplicate`` already set (from
              ``flag_duplicate_patients``). If the column is absent it is added
              automatically.
    strategy: How to pick the representative row for each duplicated patient.

              ``"keep_latest_positive"`` (default):
                  Keep the row whose ``biopsy-date`` is latest among rows with
                  a positive biopsy result. If no positive biopsy exists, keep
                  the row with the latest ``mri_1-date``.

              ``"keep_first"``:
                  Keep the first row for each patient_id (original Excel order).

              ``"keep_last"``:
                  Keep the last row for each patient_id.

    Returns
    -------
    De-duplicated DataFrame (one row per patient_id). The ``is_duplicate``
    column is dropped from the output.
    """
    if "is_duplicate" not in df.columns:
        df = flag_duplicate_patients(df)

    unique_df = df[~df["is_duplicate"]].copy()
    dup_df = df[df["is_duplicate"]].copy()

    if dup_df.empty:
        return df.drop(columns=["is_duplicate"])

    resolved_rows: list[pd.DataFrame] = []

    for pid, group in dup_df.groupby("patient_id"):
        if strategy == "keep_latest_positive":
            positive_mask = (
                group.get("biopsy-result", pd.Series(dtype=str))
                .str.strip()
                .str.lower()
                .isin(_POSITIVE_VALUES)
            )
            positives = group[positive_mask]
            if not positives.empty and "biopsy-date" in positives.columns:
                row = positives.sort_values("biopsy-date").iloc[[-1]]
            elif "mri_1-date" in group.columns:
                row = group.sort_values("mri_1-date").iloc[[-1]]
            else:
                row = group.iloc[[-1]]
        elif strategy == "keep_first":
            row = group.iloc[[0]]
        elif strategy == "keep_last":
            row = group.iloc[[-1]]
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        resolved_rows.append(row)

    resolved = pd.concat(resolved_rows, ignore_index=True)
    result = pd.concat([unique_df, resolved], ignore_index=True)
    return result.drop(columns=["is_duplicate"])


def flag_converter_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``is_converter`` boolean column.

    A converter is a patient who had at least one positive MRI result followed
    by a subsequent negative MRI result (i.e., their lesion appeared to resolve
    over the follow-up period).

    Known converters: 7049838, 7051280, 5519879.

    Returns
    -------
    df with ``is_converter`` column added.
    """
    df = df.copy()

    result_cols = [c for c in _MRI_RESULT_COLS if c in df.columns]

    def _is_converter(row: pd.Series) -> bool:
        results = [_normalise_result(row.get(c)) for c in result_cols]
        seen_positive = False
        for r in results:
            if r is None:
                continue
            if r in _POSITIVE_VALUES:
                seen_positive = True
            elif r in _NEGATIVE_VALUES and seen_positive:
                return True
        return False

    df["is_converter"] = df.apply(_is_converter, axis=1)
    return df


def flag_weird_patients(
    df: pd.DataFrame,
    threshold: float = 1.8,
) -> pd.DataFrame:
    """Add a ``psa_doubled`` boolean column.

    A patient is flagged if their PSA at MRI 2 is ≥ ``threshold`` × their PSA
    at MRI 1. Default threshold of 1.8 (~doubling) matches the clinical
    observation in the TO-DOs.

    Known cases: 2102593, 2205490, 2162006.

    Parameters
    ----------
    df:        Input DataFrame.
    threshold: Ratio mri_2_psa / mri_1_psa above which a patient is flagged.

    Returns
    -------
    df with ``psa_doubled`` column added.
    """
    df = df.copy()

    if "mri_1-psa" in df.columns and "mri_2-psa" in df.columns:
        ratio = pd.to_numeric(df["mri_2-psa"], errors="coerce") / pd.to_numeric(
            df["mri_1-psa"], errors="coerce"
        )
        df["psa_doubled"] = ratio >= threshold
    else:
        df["psa_doubled"] = False

    return df
