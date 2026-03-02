"""
pca_mri.preprocessing.columns — column-level cleaning.

Migrated from data-cleanup.ipynb (steps 2–4).

Functions
---------
drop_empty_columns(df)          Remove all-NaN columns.
drop_duplicate_columns(df)      Remove columns whose non-NaN values duplicate another column.
rename_columns(df, col_map)     Apply the project-standard column naming convention.
reorder_columns(df)             Sort columns into logical groups.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Project-standard column map
# Keys = original names from the raw Excel file.
# Values = target names in the cleaned dataset.
# Prefix conventions:
#   tx-       treatment variables
#   psa-      PSA / CAPRA variables
#   mri_N-    serial MRI visits (N = 1..4)
#   biopsy-   recurrence-investigation biopsy
#   pet-      PET scan
#   bf-       biochemical failure
# ---------------------------------------------------------------------------
DEFAULT_COL_MAP: dict[str, str] = {
    "PatientID": "patient_id",
    # Treatment
    "all patients.Age": "tx-age",
    "all patients.ÉchantPélevés": "tx-biopsy_num_samples",
    "all patients.ÉchantPositifs": "tx-biopsy_num_positive",
    "all patients.GleasonTotal": "tx-gleason_total",
    "all patients.T": "tx-t_stage",
    "all patients.protocole": "tx-protocol",
    "date treatment.DateBrachy": "tx-date",
    "all patients.TypeTX": "tx-type",
    "all patients.DoseTotalProstate": "tx-total_dose_prostate",
    "all patients.D28VolD90": "tx-d28_vol_d90",
    "all patients.D28VolV100": "tx-d28_vol_v100",
    "all patients.ADT": "tx-adt",
    # PSA / CAPRA
    "all patients.ApsMonth": "psa-time_since_tx",
    "all patients.Aps": "psa-val",
    "all patients.nadiraps02": "psa-nadir_02",
    "all patients.nadiraps05": "psa-nadir_05",
    "all patients.CAPRA": "psa-capra_total",
    "all patients.aps_capra": "psa-capra_psa",
    "all patients.gleason_capra": "psa-capra_gleason",
    "all patients.tstage_capra": "psa-capra_t_stage",
    "all patients.biopsy_capra": "psa-capra_biopsy",
    "all patients.age_capra": "psa-capra_age",
    # Outcome dates
    "all patients.ddeces": "date_death",
    "all patients.biochemical recurrence": "bf-date",
    # MRI 1
    "DateRecInvIRM": "mri_1-date",
    "ResultatIRMRecInv": "mri_1-result",
    "PIRADSLesionRecInv": "mri_1-pirads_score",
    "VolProstateIRM": "mri_1-prostate_vol",
    "PSA": "mri_1-psa",
    # MRI 2
    "2nd mri date": "mri_2-date",
    "2nd mri resultat": "mri_2-result",
    "2ndmri  vol": "mri_2-prostate_vol",
    "2nd mri psa": "mri_2-psa",
    # MRI 3
    "3rdmri date": "mri_3-date",
    "3rdmri resultat": "mri_3-result",
    "3rdmri vol": "mri_3-prostate_vol",
    "3rdmri psa ": "mri_3-psa",
    # MRI 4
    "4th mri date ": "mri_4-date",
    "4thmri resultat": "mri_4-result",
    "4th mri vol": "mri_4-prostate_vol",
    "4th mri psa ": "mri_4-psa",
    # Recurrence biopsy
    "RecInvbiopsie.DateRecInvBiopsie": "biopsy-date",
    "RecInvbiopsie.ResultatRecInvBiopsie": "biopsy-result",
    "RecInvbiopsie.GleasonPrimRecInv": "biopsy-gleason_1ary",
    "RecInvbiopsie.GleasonSecRecInv": "biopsy-gleason_2ary",
    "RecInvbiopsie.GleasonTertRecInv": "biopsy-gleason_3ary",
    # PET
    "pet.DateRecInvRadiative": "pet-date",
    "pet.TypeRecInvRadiative": "pet-tracer",
    "pet.ResultatRecInvRadiative": "pet-result",
}

# Logical column order for the cleaned dataset
_COLUMN_ORDER_PREFIXES = [
    "patient_id",
    "tx-",
    "psa-",
    "mri_1-",
    "mri_2-",
    "mri_3-",
    "mri_4-",
    "biopsy-",
    "pet-",
    "bf-",
    "date_death",
]


def drop_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns that are entirely NaN.

    Returns
    -------
    (cleaned_df, dropped_column_names)
    """
    empty_cols = df.columns[df.isna().all()].tolist()
    return df.drop(columns=empty_cols), empty_cols


def drop_duplicate_columns(df):
    """Remove columns whose overlapping non-NaN values are identical to another column.

    Parameters
    ----------
    df: Input DataFrame.

    Returns
    -------
    remove_cols: List of columns to remove
    """

    cols = df.columns.tolist()
    remove_cols = []
    same_cols = []

    for i, col_i in enumerate(cols):
        if col_i in remove_cols:
            continue
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if col_j in remove_cols:
                continue
            mask = df[col_i].notna() & df[col_j].notna()
            if mask.sum() > 0 and (df[col_i][mask] == df[col_j][mask]).all():
                remove_cols.append(col_j)
                same_cols.append(f"{col_i} same as {col_j}")

    return same_cols, remove_cols


def rename_columns(
    df: pd.DataFrame,
    col_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply the project-standard column naming convention.

    Parameters
    ----------
    df:      Input DataFrame.
    col_map: Custom mapping {old_name: new_name}. If None, uses DEFAULT_COL_MAP.

    Returns
    -------
    DataFrame with renamed columns (unrecognized columns are kept as-is).
    """
    mapping = col_map if col_map is not None else DEFAULT_COL_MAP
    return df.rename(columns=mapping)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns into the logical group order defined by _COLUMN_ORDER_PREFIXES.

    Columns that don't match any prefix are placed at the end in their
    original relative order.

    Returns
    -------
    DataFrame with reordered columns.
    """
    ordered: list[str] = []
    remaining = list(df.columns)

    for prefix in _COLUMN_ORDER_PREFIXES:
        group = [c for c in remaining if c == prefix or c.startswith(prefix)]
        ordered.extend(group)
        for c in group:
            remaining.remove(c)

    ordered.extend(remaining)
    return df[ordered]
