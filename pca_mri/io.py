"""
pca_mri.io — load and save project data files.

Functions
---------
load_raw(path)          Read the original merged Excel file.
load_clean(path)        Read a cleaned CSV or Excel file.
save(df, stem, tz)      Write df to CSV + Excel with a timestamped filename.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# def load_raw(path: str | Path = "Everything merged 18.12._new.xlsx") -> pd.DataFrame:
#     """Read the original raw dataset (N=110, 71 columns)."""
#     return pd.read_excel(path)


def load_clean(path: str | Path = "df-clean-01.csv") -> pd.DataFrame:
    """Read a cleaned dataset from CSV or Excel."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def save(
    df: pd.DataFrame,
    stem: str = "df-clean",
    tz: str = "America/Montreal",
) -> tuple[Path, Path]:
    """Save *df* to ``<stem>_<timestamp>.csv`` and ``<stem>_<timestamp>.xlsx``.

    Parameters
    ----------
    df:   DataFrame to save.
    stem: Base filename (without extension or timestamp).
    tz:   Timezone for the timestamp (default: "America/Montreal").

    Returns
    -------
    Tuple of (csv_path, xlsx_path) as Path objects.
    """
    ts = datetime.now(tz=ZoneInfo(tz)).strftime("%Y%m%d_%H%M%S")
    csv_path = Path(f"{stem}_{ts}.csv")
    xlsx_path = Path(f"{stem}_{ts}.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"Saved: {csv_path}, {xlsx_path}")
    return csv_path, xlsx_path
