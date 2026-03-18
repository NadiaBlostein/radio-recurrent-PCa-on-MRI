"""
pca_mri.analysis.diagnostic — diagnostic accuracy & prevalence metrics.

Functions
---------
prevalence(df)                  MRI-positive prevalence with 95% CI.
contingency_table(df)           2x2 table: MRI result vs biopsy result.
diagnostic_accuracy(df)         Sensitivity, specificity, PPV, NPV, accuracy, kappa.
prevalence_by_subgroup(df, by)  Prevalence stratified by a grouping column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MRI_POSITIVE = {"positive", "positif", "positiv"}
_MRI_NEGATIVE = {"négative", "negative", "négatif", "negativ"}
_BIOPSY_POSITIVE = {"positif", "positive", "positiv"}
_BIOPSY_NEGATIVE = {"négative", "négatif", "negative", "negativ"}

_TX_SHORT = {
    "Curietherapie LDR": "LDR",
    "Curietherapie HDR": "HDR",
    "Radiotherapie": "RT",
}


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval."""
    if n == 0:
        return (np.nan, np.nan)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prevalence(
    df: pd.DataFrame,
    col: str = "rec_mri-result",
    alpha: float = 0.05,
) -> dict:
    """Compute prevalence of MRI-positive recurrence with exact 95% CI.

    Parameters
    ----------
    df:    Analysis-ready DataFrame (BF cohort).
    col:   Column containing MRI result.
    alpha: Significance level for confidence interval.

    Returns
    -------
    Dict with keys: n_positive, n_total, prevalence, ci_lower, ci_upper.
    """
    norm = _norm(df[col])
    n_pos = norm.isin(_MRI_POSITIVE).sum()
    n_total = norm.isin(_MRI_POSITIVE | _MRI_NEGATIVE).sum()
    prev = n_pos / n_total if n_total > 0 else np.nan
    lo, hi = _clopper_pearson(int(n_pos), int(n_total), alpha)
    return {
        "n_positive": int(n_pos),
        "n_total": int(n_total),
        "prevalence": prev,
        "ci_lower": lo,
        "ci_upper": hi,
    }


def prevalence_by_subgroup(
    df: pd.DataFrame,
    by: str = "tx-type",
    mri_col: str = "rec_mri-result",
) -> pd.DataFrame:
    """Prevalence of MRI+ recurrence stratified by a grouping variable.

    Returns
    -------
    DataFrame with columns: group, n_positive, n_total, prevalence,
    ci_lower, ci_upper, and a final row for the Fisher/chi-squared p-value.
    """
    rows = []
    contingency_counts = []
    for group, sub in df.groupby(by, dropna=True):
        label = _TX_SHORT.get(str(group), str(group))
        p = prevalence(sub, mri_col)
        p["group"] = label
        rows.append(p)
        norm = _norm(sub[mri_col])
        n_pos = norm.isin(_MRI_POSITIVE).sum()
        n_neg = norm.isin(_MRI_NEGATIVE).sum()
        contingency_counts.append([n_pos, n_neg])

    result = pd.DataFrame(rows)[
        ["group", "n_positive", "n_total", "prevalence", "ci_lower", "ci_upper"]
    ]

    # p-value across subgroups
    table = np.array(contingency_counts)
    if table.shape[0] >= 2 and table.sum() > 0:
        if table.shape == (2, 2):
            _, p_val = stats.fisher_exact(table)
        else:
            _, p_val, _, _ = stats.chi2_contingency(table)
        result.attrs["p_value"] = p_val
    return result


def contingency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build 2x2 contingency table: MRI result (rows) vs biopsy result (cols).

    Returns
    -------
    DataFrame with index ['MRI+', 'MRI-'], columns ['Biopsy+', 'Biopsy-'],
    and integer counts.
    """
    mri = _norm(df["rec_mri-result"])
    bx = _norm(df["biopsy-result"])

    mri_pos = mri.isin(_MRI_POSITIVE)
    mri_neg = mri.isin(_MRI_NEGATIVE)
    bx_pos = bx.isin(_BIOPSY_POSITIVE)
    bx_neg = bx.isin(_BIOPSY_NEGATIVE)

    valid = (mri_pos | mri_neg) & (bx_pos | bx_neg)

    tp = (mri_pos & bx_pos & valid).sum()
    fp = (mri_pos & bx_neg & valid).sum()
    fn = (mri_neg & bx_pos & valid).sum()
    tn = (mri_neg & bx_neg & valid).sum()

    return pd.DataFrame(
        [[int(tp), int(fp)], [int(fn), int(tn)]],
        index=["MRI+", "MRI-"],
        columns=["Biopsy+", "Biopsy-"],
    )


def diagnostic_accuracy(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compute diagnostic accuracy metrics with 95% CIs.

    Uses biopsy as the reference standard.

    Returns
    -------
    DataFrame with columns: metric, value, ci_lower, ci_upper.
    """
    ct = contingency_table(df)
    tp = ct.loc["MRI+", "Biopsy+"]
    fp = ct.loc["MRI+", "Biopsy-"]
    fn = ct.loc["MRI-", "Biopsy+"]
    tn = ct.loc["MRI-", "Biopsy-"]
    n = tp + fp + fn + tn

    metrics = {}

    # Sensitivity = TP / (TP + FN)
    denom = tp + fn
    metrics["Sensitivity"] = (tp / denom if denom else np.nan, denom)

    # Specificity = TN / (TN + FP)
    denom = tn + fp
    metrics["Specificity"] = (tn / denom if denom else np.nan, denom)

    # PPV = TP / (TP + FP)
    denom = tp + fp
    metrics["PPV"] = (tp / denom if denom else np.nan, denom)

    # NPV = TN / (TN + FN)
    denom = tn + fn
    metrics["NPV"] = (tn / denom if denom else np.nan, denom)

    # Accuracy = (TP + TN) / N
    metrics["Accuracy"] = ((tp + tn) / n if n else np.nan, n)

    rows = []
    for name, (val, denom) in metrics.items():
        k = int(round(val * denom)) if not np.isnan(val) else 0
        lo, hi = _clopper_pearson(k, int(denom), alpha)
        rows.append({
            "metric": name,
            "value": val,
            "ci_lower": lo,
            "ci_upper": hi,
        })

    # Cohen's kappa
    po = (tp + tn) / n if n else np.nan
    pe = (
        ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        if n else np.nan
    )
    kappa = (po - pe) / (1 - pe) if (not np.isnan(pe) and pe != 1) else np.nan

    # Kappa SE and CI (approximation)
    if n > 0 and not np.isnan(kappa):
        se_kappa = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
        rows.append({
            "metric": "Cohen's kappa",
            "value": kappa,
            "ci_lower": kappa - 1.96 * se_kappa,
            "ci_upper": kappa + 1.96 * se_kappa,
        })
    else:
        rows.append({
            "metric": "Cohen's kappa",
            "value": kappa,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        })

    result = pd.DataFrame(rows)
    result.attrs["contingency"] = ct
    result.attrs["counts"] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return result
