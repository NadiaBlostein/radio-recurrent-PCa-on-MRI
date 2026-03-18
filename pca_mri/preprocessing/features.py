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
add_time_to_bf(df)              bf-time_to_bf-days column.
add_time_to_biopsy(df)          biopsy-time_to_biopsy-days column.
add_psa_timepoints(df)          psa_t1-val/date, psa_t2-val/date columns.
add_psa_doubling_time_rec_mri(df) psa_dt-rec_mri-months column.
add_psa_diff_rec_mri(df)        psa_diff-rec_mri-days column.
add_bf_to_rec_mri(df)           bf_to_rec_mri-days column.
add_all_features(df)            Convenience wrapper — calls all of the above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_POSITIVE_VALUES = {"positive", "positif", "positiv"}


def _psa_doubling_time(
    psa1: pd.Series,
    date1: pd.Series,
    psa2: pd.Series,
    date2: pd.Series,
) -> pd.Series:
    """Return PSA doubling time in months between two time points.

    Formula: PSA-DT = (t₂ − t₁) × ln(2) / ln(PSA₂ / PSA₁)

    Returns NaN for rows where either PSA is non-positive, the time interval
    is zero or negative, or the PSA ratio is non-positive.
    """
    psa1 = pd.to_numeric(psa1, errors="coerce")
    psa2 = pd.to_numeric(psa2, errors="coerce")
    date1 = pd.to_datetime(date1, errors="coerce")
    date2 = pd.to_datetime(date2, errors="coerce")

    interval_months = (date2 - date1).dt.days / 30.44
    ratio = psa2 / psa1
    valid = (psa1 > 0) & (psa2 > 0) & (interval_months > 0) & (ratio > 0)

    return pd.Series(
        np.where(
            valid,
            interval_months * np.log(2) / np.log(ratio.where(valid, other=np.nan)),
            np.nan,
        ),
        index=psa1.index,
    )


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
    """Add ``bf-time_to_bf-days`` = days from treatment to biochemical failure.

    Requires columns ``tx-date`` and ``bf-date``.
    Rows where either date is NaN produce NaN.
    """
    df = df_in.copy()
    if "tx-date" in df.columns and "bf-date" in df.columns:
        tx = pd.to_datetime(df["tx-date"], errors="coerce")
        bf = pd.to_datetime(df["bf-date"], errors="coerce")
        df["bf-time_to_bf-days"] = (bf - tx).dt.days
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
        return pd.Series([np.nan, np.nan, np.nan])
    
    df[["rec_mri-index", "rec_mri-date", "rec_mri-time_to_rec-days"]] = \
        df.apply(extract_recurrence, axis=1)

    return df


def add_psa_timepoints(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add PSA timepoint columns used by downstream doubling-time calculations.

    Columns added:

    * ``psa_t1-val``: baseline PSA value (same as ``psa-val``).
    * ``psa_t1-date``: baseline PSA date = ``tx-date`` + ``psa-time_since_tx`` days.
    * ``psa_t2-val``: PSA at the first *positive* MRI; if no MRI recurrence,
      PSA at the first MRI visit.
    * ``psa_t2-date``: date of the first positive MRI; date of the first MRI
      if no recurrence.

    Requires ``add_time_to_recurrence_MRI`` to have been called first so that
    ``rec_mri-index`` exists.
    """
    df = df_in.copy()
    required = {"psa-val", "psa-time_since_tx", "tx-date"}
    if not required.issubset(df.columns):
        return df

    # --- t1 (baseline) ---
    df["psa_t1-val"] = pd.to_numeric(df["psa-val"], errors="coerce")
    df["psa_t1-date"] = pd.to_datetime(df["tx-date"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(df["psa-time_since_tx"], errors="coerce"), unit="D"
    )

    # --- t2 (recurrence MRI or first MRI) ---
    rec_index = pd.to_numeric(df.get("rec_mri-index"), errors="coerce") if "rec_mri-index" in df.columns else pd.Series(np.nan, index=df.index)
    has_rec = rec_index.notna()

    psa_t2 = pd.Series(np.nan, index=df.index, dtype=float)
    date_t2 = pd.Series(pd.NaT, index=df.index)

    # For patients WITH MRI recurrence: use the recurrence MRI PSA & date
    for i in range(1, 5):
        psa_col = f"mri_{i}-psa"
        date_col = f"mri_{i}-date"
        mask = has_rec & (rec_index == i)
        if psa_col in df.columns:
            psa_t2[mask] = pd.to_numeric(df.loc[mask, psa_col], errors="coerce")
        if date_col in df.columns:
            date_t2[mask] = pd.to_datetime(df.loc[mask, date_col], errors="coerce")

    # For patients WITHOUT MRI recurrence: fall back to first MRI
    no_rec = ~has_rec
    if "mri_1-psa" in df.columns:
        psa_t2[no_rec] = pd.to_numeric(df.loc[no_rec, "mri_1-psa"], errors="coerce")
    if "mri_1-date" in df.columns:
        date_t2[no_rec] = pd.to_datetime(df.loc[no_rec, "mri_1-date"], errors="coerce")

    df["psa_t2-val"] = psa_t2
    df["psa_t2-date"] = date_t2

    return df


def add_psa_doubling_time_rec_mri(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add ``psa_dt-rec_mri-months``: PSA doubling time between t1 and t2.

    Uses ``psa_t1-val/date`` and ``psa_t2-val/date`` (created by
    ``add_psa_timepoints``).

    * If PSA rose (t2 > t1): standard doubling-time formula
      PSA-DT = (t₂ − t₁) × ln(2) / ln(PSA₂ / PSA₁), in months.
    * If PSA did not increase (t2 ≤ t1): assigns NaN.

    Requires ``add_psa_timepoints`` to have been called first.
    """
    df = df_in.copy()
    required = {"psa_t1-val", "psa_t1-date", "psa_t2-val", "psa_t2-date"}
    if not required.issubset(df.columns):
        return df

    psa1 = pd.to_numeric(df["psa_t1-val"], errors="coerce")
    date1 = pd.to_datetime(df["psa_t1-date"], errors="coerce")
    psa2 = pd.to_numeric(df["psa_t2-val"], errors="coerce")
    date2 = pd.to_datetime(df["psa_t2-date"], errors="coerce")

    # PSA rose → compute doubling time
    dt = _psa_doubling_time(psa1, date1, psa2, date2)

    # PSA did not increase → NaN
    no_rise = psa2 <= psa1
    dt[no_rise] = np.nan

    df["psa_dt-rec_mri-months"] = dt
    return df


def add_psa_diff_rec_mri(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add ``psa_diff-rec_mri-days``: PSA difference between t2 and t1.

    Computed as ``psa_t2-val`` − ``psa_t1-val``.

    Requires ``add_psa_timepoints`` to have been called first.
    """
    df = df_in.copy()
    required = {"psa_t1-val", "psa_t2-val"}
    if not required.issubset(df.columns):
        return df

    psa1 = pd.to_numeric(df["psa_t1-val"], errors="coerce")
    psa2 = pd.to_numeric(df["psa_t2-val"], errors="coerce")
    df["psa_diff-rec_mri-days"] = psa2 - psa1
    return df


def add_time_to_biopsy(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add ``biopsy-time_to_biopsy-days`` = days from treatment to recurrence-investigation biopsy.

    Requires columns ``tx-date`` and ``biopsy-date``.
    Rows where either date is NaN produce NaN.
    """
    df = df_in.copy()
    if "tx-date" in df.columns and "biopsy-date" in df.columns:
        tx = pd.to_datetime(df["tx-date"], errors="coerce")
        biopsy = pd.to_datetime(df["biopsy-date"], errors="coerce")
        df["biopsy-time_to_biopsy-days"] = (biopsy - tx).dt.days
    return df


def add_bf_to_rec_mri(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add ``bf_to_rec_mri-days`` = days from biochemical failure to first positive MRI.

    Computed as ``rec_mri-time_to_rec-days`` − ``bf-time_to_bf-days`` (both measured from treatment date), so the sign convention is:
      - Positive  → BF happened first (MRI recurrence detected later)
      - Negative  → MRI recurrence detected first (BF recorded later)
      - NaN       → either ``bf-time_to_bf-days`` or ``rec_mri-time_to_rec-days`` is NaN

    Requires ``add_time_to_bf`` and ``add_time_to_recurrence_MRI`` to have been called first.
    """
    df = df_in.copy()
    if "bf-time_to_bf-days" in df.columns and "rec_mri-time_to_rec-days" in df.columns:
        df["bf_to_rec_mri-days"] = (
            pd.to_numeric(df["rec_mri-time_to_rec-days"], errors="coerce")
            - pd.to_numeric(df["bf-time_to_bf-days"], errors="coerce")
        )
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps in sequence.
    """
    df = add_biopsy_ratio(df)
    df = add_time_to_bf(df)
    df = add_time_to_recurrence_MRI(df)
    df = add_time_to_biopsy(df)
    df = add_psa_timepoints(df)
    df = add_psa_doubling_time_rec_mri(df)
    df = add_psa_diff_rec_mri(df)
    df = add_bf_to_rec_mri(df)
    return df
