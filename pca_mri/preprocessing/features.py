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
add_all_features(df)            Convenience wrapper — calls all of the above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_biopsy_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``biopsy-positive_ratio`` = positive samples / samples taken.

    Requires columns ``tx-biopsy_num_positive`` and ``tx-biopsy_num_samples``.
    Rows where either value is NaN or samples = 0 produce NaN.
    """
    df = df.copy()
    if "tx-biopsy_num_positive" in df.columns and "tx-biopsy_num_samples" in df.columns:
        pos = pd.to_numeric(df["tx-biopsy_num_positive"], errors="coerce")
        total = pd.to_numeric(df["tx-biopsy_num_samples"], errors="coerce")
        df["biopsy-positive_ratio"] = pos / total.replace(0, np.nan)
    return df


def add_time_to_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``time_to_recurrence_days`` = days from treatment to first positive MRI.

    Uses ``tx-date`` as the reference date and ``mri_1-date`` as the event date.
    Negative values (MRI before tx) are set to NaN.
    """
    df = df.copy()
    if "tx-date" in df.columns and "mri_1-date" in df.columns:
        tx = pd.to_datetime(df["tx-date"], errors="coerce")
        mri = pd.to_datetime(df["mri_1-date"], errors="coerce")
        delta = (mri - tx).dt.days
        df["time_to_recurrence_days"] = delta.where(delta >= 0)
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
    """Apply all feature-engineering steps in sequence.

    Calls, in order:
    1. add_biopsy_ratio
    2. add_time_to_recurrence
    3. add_mri_intervals
    4. add_psa_doubling_time
    """
    df = add_biopsy_ratio(df)
    df = add_time_to_recurrence(df)
    df = add_mri_intervals(df)
    df = add_psa_doubling_time(df)
    return df
