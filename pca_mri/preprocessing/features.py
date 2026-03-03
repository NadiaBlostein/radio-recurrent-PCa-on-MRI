"""
pca_mri.preprocessing.features — derived feature engineering.

Addresses TO-DOs from data-processing-TO-DOS.ipynb:
  - Add biopsy positive-sample ratio
  - Add time-to-recurrence
  - Add inter-MRI intervals
  - Add PSA doubling time

Functions
---------
add_biopsy_ratio(df)            biopsy-positive_ratio column.
add_time_to_recurrence(df)      time_to_recurrence_days column.
add_mri_intervals(df)           mri_N_to_M_days columns.
add_psa_doubling_time(df)       psa_doubling_time_months column.
add_time_to_bf(df)              time_to_bf-days column.
add_all_features(df)            Convenience wrapper — calls all of the above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_POSITIVE_VALUES = {"positive", "positif", "positiv"}
_NEGATIVE_VALUES = {"negative", "negativ", "négative", "négatif"}


def add_biopsy_ratio(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add ``tx-biopsy_positive_ratio`` = positive samples / samples taken.

    Requires columns ``tx-biopsy_num_positive`` and ``tx-biopsy_num_samples``.
    Rows where either value is NaN or samples = 0 produce NaN.
    """
    df = df_in.copy()
    if "tx-biopsy_num_positive" in df.columns and "tx-biopsy_num_samples" in df.columns:
        pos = pd.to_numeric(df["tx-biopsy_num_positive"], errors="coerce")
        total = pd.to_numeric(df["tx-biopsy_num_samples"], errors="coerce")
        df["tx-biopsy_positive_ratio"] = pos / total.replace(0, np.nan)
    return df


def add_time_to_bf(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add ``time_to_bf-days`` = days from treatment to biochemical failure.

    Requires columns ``tx-date`` and ``bf-date``.
    Rows where either date is NaN produce NaN.
    """
    df = df_in.copy()
    if "tx-date" in df.columns and "bf-date" in df.columns:
        tx = pd.to_datetime(df["tx-date"], errors="coerce")
        bf = pd.to_datetime(df["bf-date"], errors="coerce")
        df["time_to_bf-days"] = (bf - tx).dt.days
    return df


def add_time_to_recurrence_MRI(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the first positive MRI date for each patient and calculates the interval in days from treatment to that date. 
    
    Adds the following 3 columns:
        - ``rec_mri-index``: 0 if no positive MRI, otherwise the index (1–4) of the first positive MRI visit.
        - ``rec_mri-date``: the date of the first positive MRI, or NaT if none.
        - ``rec_MRI-time_to_rec-days``: 0 if no positive MRI, otherwise the number of days from treatment to the first positive MRI.
    
    """
    df = df_in.copy()

    for i in range(1, 5):
        df[f"mri_{i}-result"] = (
            df[f"mri_{i}-result"].astype(str).str.strip().str.lower())

    def extract_recurrence(row):
        for i in range(1, 5):
            result = row[f"mri_{i}-result"]
            if result in _POSITIVE_VALUES:
                rec_date = row[f"mri_{i}-date"]
                if pd.notna(rec_date) and pd.notna(row["tx-date"]):
                    delta_days = (rec_date - row["tx-date"]).days
                else:
                    delta_days = np.nan
                return pd.Series([i, rec_date, delta_days])
        # No positive MRI found
        return pd.Series([0, np.nan, 0])
    
    df[["rec_mri-index", "rec_mri-date", "time_to_recurrence-days"]] = \
        df.apply(extract_recurrence, axis=1)

    return df


def add_mri_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Add inter-MRI interval columns in days.

    Adds up to three columns:
      - ``mri_1_to_2_days``
      - ``mri_2_to_3_days``
      - ``mri_3_to_4_days``

    Each is the number of days between consecutive MRI dates. Negative values
    (later MRI recorded before earlier one) are set to NaN.
    """
    df = df.copy()
    date_cols = ["mri_1-date", "mri_2-date", "mri_3-date", "mri_4-date"]
    pairs = [
        ("mri_1-date", "mri_2-date", "mri_1_to_2_days"),
        ("mri_2-date", "mri_3-date", "mri_2_to_3_days"),
        ("mri_3-date", "mri_4-date", "mri_3_to_4_days"),
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for early, late, out in pairs:
        if early in df.columns and late in df.columns:
            delta = (df[late] - df[early]).dt.days
            df[out] = delta.where(delta >= 0)
    return df


def add_psa_doubling_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``psa_doubling_time_months`` between MRI 1 and MRI 2.

    Formula: PSA-DT = (t₂ − t₁) × ln(2) / ln(PSA₂ / PSA₁)
    where t₂ − t₁ is the interval in months between ``mri_1-date`` and
    ``mri_2-date``.

    Requires ``mri_1-psa``, ``mri_2-psa``, ``mri_1-date``, ``mri_2-date``.
    Rows with non-positive PSA values or zero time interval produce NaN.
    """
    df = df.copy()
    required = {"mri_1-psa", "mri_2-psa", "mri_1-date", "mri_2-date"}
    if not required.issubset(df.columns):
        return df

    psa1 = pd.to_numeric(df["mri_1-psa"], errors="coerce")
    psa2 = pd.to_numeric(df["mri_2-psa"], errors="coerce")
    date1 = pd.to_datetime(df["mri_1-date"], errors="coerce")
    date2 = pd.to_datetime(df["mri_2-date"], errors="coerce")

    interval_months = (date2 - date1).dt.days / 30.44  # avg days per month

    ratio = psa2 / psa1
    # Require positive PSA values and positive time interval
    valid = (psa1 > 0) & (psa2 > 0) & (interval_months > 0) & (ratio > 0)

    psadt = np.where(
        valid,
        interval_months * np.log(2) / np.log(ratio.where(valid, other=np.nan)),
        np.nan,
    )
    df["psa_doubling_time_months"] = psadt
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps in sequence.
    """
    df = add_biopsy_ratio(df)
    df = add_time_to_bf(df)
    df = add_time_to_recurrence_MRI(df)
    # df = add_mri_intervals(df)
    # df = add_psa_doubling_time(df)
    return df
