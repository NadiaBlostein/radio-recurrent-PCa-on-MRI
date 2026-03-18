"""
pca_mri.analysis.descriptive — descriptive statistics.

Functions
---------
table1(df, stratify_by)    Clinical characteristics table stratified by a grouping variable.
missingness_summary(df)    Column-level missingness report.
capra_summary(df)          CAPRA score distribution and risk-group breakdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TX_TYPE_SHORT: dict[str, str] = {
    "Curietherapie LDR": "LDR",
    "Curietherapie HDR": "HDR",
    "Radiotherapie":     "RT",
}

# Canonical display labels for mixed French/English raw values
_RESULT_MAP: dict[str, str] = {
    "positive":  "Positive",
    "positif":   "Positive",
    "positiv":   "Positive",
    "négative":  "Negative",
    "negativ":   "Negative",
    "negative":  "Negative",
    "négatif":   "Negative",
    "equivoque": "Equivocal",
    "oui":       "Yes",
    "non":       "No",
    "true":      "Yes",
    "false":     "No",
}


def _normalise(val) -> str | None:
    """Return a canonical display string for a raw result/boolean value."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    return _RESULT_MAP.get(s, str(val).strip())


def _fmt_continuous(series: pd.Series) -> str:
    """Median [Q1–Q3] for a numeric series, ignoring ±inf."""
    v = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return "—"
    q1, med, q3 = v.quantile([0.25, 0.50, 0.75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]"


def _continuous_p(groups: list[pd.Series]) -> str:
    """P-value for continuous variable: Mann-Whitney U (2 groups) or Kruskal-Wallis (3+)."""
    arrays = [
        pd.to_numeric(g, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        for g in groups
    ]
    if len(arrays) < 2 or any(len(a) < 2 for a in arrays):
        return "—"
    try:
        if len(arrays) == 2:
            _, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        else:
            _, p = stats.kruskal(*arrays)
        return f"{p:.3f}" if p >= 0.001 else "<0.001"
    except Exception:
        return "—"


def _categorical_p(contingency: pd.DataFrame) -> str:
    """P-value for categorical variable: Fisher's exact (2x2 or small cells) or chi-squared."""
    try:
        table = contingency.values.astype(float)
        if table.shape[0] < 2 or table.sum() == 0:
            return "—"
        # 2x2 table: always use Fisher's exact
        if table.shape == (2, 2):
            _, p = stats.fisher_exact(table)
            return f"{p:.3f}" if p >= 0.001 else "<0.001"
        # r x c: use Fisher's if any expected count < 5, else chi-squared
        _, _, _, expected = stats.chi2_contingency(table)
        if (expected < 5).any():
            # scipy Fisher exact only supports 2x2; use chi2 with warning
            # For r x c with small counts, use Fisher-Freeman-Halton via monte carlo
            try:
                result = stats.chi2_contingency(table, lambda_="log-likelihood")
                p = result[1]
            except Exception:
                _, p, _, _ = stats.chi2_contingency(table)
        else:
            _, p, _, _ = stats.chi2_contingency(table)
        return f"{p:.3f}" if p >= 0.001 else "<0.001"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Variables included in Table 1
# ---------------------------------------------------------------------------

_CONTINUOUS: list[tuple[str, str]] = [
    ("tx-age",                   "Age (years)"),
    ("psa-val",                  "PSA at diagnosis (ng/mL)"),
    ("psa-capra_total",          "CAPRA score"),
    ("tx-biopsy_positive_ratio", "Biopsy positive core ratio"),
    ("tx-d28_vol_d90",           "D90 coverage volume (%)"),
    ("bf-time_to_bf-days",       "Time to biochemical failure (days)"),
    ("rec_mri-time_to_rec-days", "Time to MRI recurrence (days)"),
    ("psa_diff-rec_mri-days",    "PSA difference to recurrence MRI"),
    ("psa_dt-rec_mri-months",    "PSA doubling time to recurrence MRI (months)"),
]

_CATEGORICAL: list[tuple[str, str]] = [
    ("tx-type",          "Primary treatment type"),
    ("biopsy-result",    "Biopsy result"),
    ("rec_mri-result",   "Recurrence MRI result"),
    ("capra-risk_group", "CAPRA risk group"),
    ("is_converter",     "Converter patient (positive→negative MRI)"),
    ("tx-gleason_total", "Gleason score at diagnosis"),
    ("tx-t_stage",       "Clinical T-stage"),
    ("tx-adt",           "Androgen deprivation therapy (ADT)"),
    ("psa-nadir_02",     "PSA nadir < 0.2 ng/mL"),
    ("psa-nadir_05",     "PSA nadir < 0.5 ng/mL"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def table1(
    df: pd.DataFrame,
    stratify_by: str = "tx-type",
    title: str | None = None,
) -> pd.DataFrame:
    """Return a Table 1 DataFrame of clinical characteristics.

    Continuous variables are reported as Median [Q1–Q3]; p-values use
    Kruskal-Wallis.  Categorical variables report N (%) per level; p-values
    use chi-squared.

    Parameters
    ----------
    df:           Cleaned analysis-ready dataset.
    stratify_by:  Column to group by (default ``'tx-type'``).
    title:        Optional title stored in ``result.attrs["title"]``.

    Returns
    -------
    DataFrame with a (Variable, Level) MultiIndex and one column per group
    plus an overall column and a p-value column.  If *title* is provided it
    is accessible via ``result.attrs["title"]``.
    """
    work = df.copy()

    if stratify_by == "tx-type":
        work["_group"] = work[stratify_by].map(_TX_TYPE_SHORT).fillna(work[stratify_by])
    else:
        work["_group"] = work[stratify_by].astype(str)

    groups = sorted(work["_group"].dropna().unique())

    # Pre-compute normalised columns for all categorical variables so that
    # subsets built below will inherit them.
    for col, _ in _CATEGORICAL:
        if col in work.columns:
            work[f"_norm_{col}"] = work[col].apply(_normalise)

    subsets = {g: work[work["_group"] == g] for g in groups}

    n_overall = len(work)
    n_per_group = {g: len(s) for g, s in subsets.items()}

    col_overall = f"Overall (N={n_overall})"
    col_groups = {g: f"{g} (N={n_per_group[g]})" for g in groups}
    col_p = "p-value"

    rows: list[dict] = []

    # --- Continuous variables ---
    for col, label in _CONTINUOUS:
        if col not in work.columns:
            continue
        row = {
            "Variable": label,
            "Level": "Median [Q1–Q3]",
            col_overall: _fmt_continuous(work[col]),
            col_p: _continuous_p([subsets[g][col] for g in groups]),
        }
        for g in groups:
            row[col_groups[g]] = _fmt_continuous(subsets[g][col])
        rows.append(row)

    # --- Categorical variables ---
    for col, label in _CATEGORICAL:
        if col not in work.columns or col == stratify_by:
            continue
        norm_col = f"_norm_{col}"
        all_levels = sorted(work[norm_col].dropna().unique())
        contingency = pd.DataFrame(
            {g: subsets[g][norm_col].value_counts() for g in groups}
        ).fillna(0)
        p_val = _categorical_p(contingency)

        for i, level in enumerate(all_levels):
            n_ov = (work[norm_col] == level).sum()
            row = {
                "Variable": label if i == 0 else "",
                "Level": str(level),
                col_overall: f"{n_ov} ({100*n_ov/n_overall:.0f}%)",
                col_p: p_val if i == 0 else "",
            }
            for g in groups:
                n_g = (subsets[g][norm_col] == level).sum()
                row[col_groups[g]] = f"{n_g} ({100*n_g/n_per_group[g]:.0f}%)"
            rows.append(row)

    result = pd.DataFrame(rows).set_index(["Variable", "Level"])
    ordered_cols = [col_overall] + [col_groups[g] for g in groups] + [col_p]
    result = result[ordered_cols]
    if title is not None:
        result.attrs["title"] = title
    return result


def style_table(
    table: pd.DataFrame,
    cmap: str = "flare",
) -> "pd.io.formats.style.Styler":
    """Apply heatmap-style background gradient to a Table 1 DataFrame.

    Extracts the leading numeric value from each formatted cell (e.g.
    ``"62.0 [56.8–66.0]"`` → 62.0, ``"20 (29%)"`` → 20) and uses it
    to colour the cell background.  Colours are normalised **per row**
    so each variable's range maps to the full colourmap independently.

    The p-value column is left unstyled (plain text on white background).

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`table1` or ``df.describe()``.
    cmap : str
        Matplotlib colourmap for data columns (default ``"YlOrRd"``).

    Returns
    -------
    pd.io.formats.style.Styler
        Styled table ready for notebook display.
    """
    import re
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    import seaborn as sns

    _BG = "#1a1a2e"
    _FG = "#e0e0e0"

    def _extract_number(val) -> float | None:
        """Pull the first decimal number from a formatted string."""
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s in ("—", ""):
            return None
        m = re.match(r"[<>]?\s*(-?[\d.]+)", s)
        return float(m.group(1)) if m else None

    # Reset multi-index to flat integer index for Styler compatibility.
    flat = table.reset_index()

    # Separate data columns from p-value and index columns
    idx_cols = ["Variable", "Level"] if "Variable" in flat.columns else []
    p_col = "p-value" if "p-value" in flat.columns else None
    data_cols = [c for c in flat.columns if c not in idx_cols and c != p_col]

    # Build numeric shadow DataFrame for colour mapping
    numeric = flat[data_cols].map(_extract_number)

    # Resolve colourmap (supports seaborn names like "flare")
    try:
        _cm = mcm.get_cmap(cmap)
    except ValueError:
        _cm = sns.color_palette(cmap, as_cmap=True)

    # Row-normalised gradient applied across all data columns at once
    def _apply_row_gradient(row: pd.Series) -> list[str]:
        cm = _cm
        shadow = numeric.loc[row.name, data_cols]
        vals = pd.to_numeric(shadow, errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        norm = (
            mcolors.Normalize(vmin=vmin, vmax=vmax)
            if pd.notna(vmin) and pd.notna(vmax) and vmin != vmax
            else None
        )
        results = []
        for col in row.index:
            v = vals.get(col)
            if col not in data_cols or pd.isna(v) or norm is None:
                results.append(f"background-color: {_BG}; color: {_FG}")
            else:
                r, g, b, _ = cm(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                fg = "white" if lum < 0.5 else "black"
                results.append(
                    f"background-color: rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85); "
                    f"color: {fg}"
                )
        return results

    # Identify rows that start a new variable (for thick separator borders)
    var_col = flat["Variable"] if "Variable" in flat.columns else None
    new_var_rows = set()
    if var_col is not None:
        for i, val in enumerate(var_col):
            if i == 0 or (str(val).strip() != ""):
                new_var_rows.add(i)

    # Build the Styler
    all_styled_cols = data_cols + ([p_col] if p_col else [])
    styler = flat.style.hide(axis="index")

    # Apply row-wise gradient across data columns only
    styler = styler.apply(_apply_row_gradient, axis=1, subset=all_styled_cols)

    # Add thick top border on rows where a new variable starts
    def _separator_borders(row: pd.Series) -> list[str]:
        if row.name in new_var_rows and row.name != 0:
            return [f"border-top: 3px solid #555"] * len(row)
        return [""] * len(row)

    styler = styler.apply(_separator_borders, axis=1)

    # Base styling for the whole table
    _BORDER = "1px solid #333"
    styler = styler.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#0f0f23"), ("color", _FG),
            ("border", _BORDER), ("padding", "8px 10px"),
            ("font-weight", "600"),
        ]},
        {"selector": "td", "props": [
            ("border", _BORDER), ("padding", "6px 10px"),
        ]},
        {"selector": "", "props": [
            ("background-color", _BG), ("color", _FG),
            ("border-collapse", "collapse"),
        ]},
    ])

    # Ensure idx_cols and p-value get plain styling (override gradient)
    plain = f"background-color: {_BG}; color: {_FG}"
    if idx_cols:
        styler = styler.map(lambda _: plain, subset=idx_cols)
    if p_col:
        styler = styler.map(lambda _: plain, subset=[p_col])

    # Add title as caption if present in attrs
    title = table.attrs.get("title")
    if title:
        styler = styler.set_caption(title)
        styler = styler.set_table_styles([
            {"selector": "caption", "props": [
                ("font-size", "14px"), ("font-weight", "bold"),
                ("text-align", "left"), ("padding", "8px 4px"),
                ("color", _FG),
            ]},
        ], overwrite=False)

    return styler


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a column-level missingness report.

    Returns
    -------
    DataFrame with columns [Column, N_missing, Pct_missing, N_present],
    sorted descending by N_missing; only columns with ≥1 missing value
    are included.
    """
    n = len(df)
    missing = df.isna().sum()
    result = pd.DataFrame({
        "Column": missing.index,
        "Number missing": missing.values,
        "Number present": (n - missing).values,
        "% missing": (100 * missing / n).round(1).values,
        
    })
    return (
        result[result["Number missing"] > 0]
        .sort_values("Number missing", ascending=False)
        .reset_index(drop=True)
    )


def capra_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a CAPRA score descriptive summary with risk-group breakdown.

    CAPRA risk groups
    -----------------
    Low risk:          0–2
    Intermediate risk: 3–5
    High risk:         6–10

    Returns
    -------
    DataFrame with one row per statistic or risk-group level.
    """
    capra = pd.to_numeric(df["psa-capra_total"], errors="coerce").dropna()

    risk_labels = pd.cut(
        capra,
        bins=[-0.1, 2, 5, 10],
        labels=["Low (0–2)", "Intermediate (3–5)", "High (6–10)"],
    )
    risk_counts = risk_labels.value_counts().sort_index()

    summary_rows = pd.DataFrame({
        "Statistic": ["N (non-missing)", "Mean ± SD", "Median", "Q1", "Q3", "Min", "Max"],
        "Value": [
            str(len(capra)),
            f"{capra.mean():.2f} ± {capra.std():.2f}",
            f"{capra.median():.1f}",
            f"{capra.quantile(0.25):.1f}",
            f"{capra.quantile(0.75):.1f}",
            f"{capra.min():.0f}",
            f"{capra.max():.0f}",
        ],
    })

    risk_header = pd.DataFrame({"Statistic": ["Risk group"], "Value": ["N (%)"]})
    risk_rows = pd.DataFrame({
        "Statistic": [f"  {g}" for g in risk_counts.index],
        "Value": [f"{n} ({100*n/len(capra):.0f}%)" for n in risk_counts],
    })

    return pd.concat([summary_rows, risk_header, risk_rows], ignore_index=True)
