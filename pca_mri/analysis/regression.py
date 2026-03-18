"""
pca_mri.analysis.regression — logistic regression for MRI recurrence prediction.

Functions
---------
prepare_outcome(df)              Binary outcome from rec_mri-result.
univariate_screen(df)            Univariate logistic regression for all candidate predictors.
build_multivariable_model(df)    Multivariable logistic regression with VIF check.
bootstrap_auc(df, n_boot)        Bootstrap-validated AUC for the final model.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Constants
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

# Candidate predictors: (column, label, type)
# type: "continuous", "binary", "categorical", "ordinal"
CANDIDATE_PREDICTORS: list[tuple[str, str, str]] = [
    ("tx-age", "Age at treatment", "continuous"),
    ("psa-val", "PSA at diagnosis", "continuous"),
    ("tx-gleason_total", "Gleason score", "ordinal"),
    ("tx-t_stage", "Clinical T-stage", "ordinal"),
    ("tx-biopsy_positive_ratio", "Positive biopsy core ratio", "continuous"),
    ("psa-capra_total", "CAPRA score", "continuous"),
    ("capra-risk_group", "CAPRA risk group", "categorical"),
    ("tx-type", "Treatment type", "categorical"),
    ("tx-adt", "ADT use", "binary"),
    ("bf-time_to_bf-days", "Time to biochemical failure (days)", "continuous"),
    ("psa_diff-rec_mri-days", "PSA difference to recurrence MRI", "continuous"),
    ("psa_dt-rec_mri-months", "PSA-DT to recurrence MRI (months)", "continuous"),
    ("psa-nadir_05", "PSA nadir < 0.5", "binary"),
    ("tx-d28_vol_d90", "D90 coverage volume (%)", "continuous"),
    ("mri_1-prostate_vol", "Prostate volume at MRI 1 (cc)", "continuous"),
    ("mri_1-psa", "PSA at MRI 1 (ng/mL)", "continuous"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def prepare_outcome(df: pd.DataFrame, col: str = "rec_mri-result") -> pd.Series:
    """Return a binary Series: 1 = MRI positive, 0 = MRI negative, NaN otherwise."""
    norm = _norm(df[col])
    outcome = pd.Series(np.nan, index=df.index, dtype=float)
    outcome[norm.isin(_MRI_POSITIVE)] = 1.0
    outcome[norm.isin(_MRI_NEGATIVE)] = 0.0
    return outcome


def _encode_predictor(
    df: pd.DataFrame, col: str, ptype: str
) -> pd.DataFrame | pd.Series:
    """Encode a single predictor for logistic regression.

    Returns a DataFrame of dummy columns (categorical) or a single Series.
    """
    if ptype == "continuous":
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return s.rename(col)

    if ptype == "ordinal":
        s = pd.to_numeric(df[col], errors="coerce")
        # If numeric coercion fails (e.g. T-stage strings), apply ordinal mapping
        if s.isna().sum() > df[col].isna().sum():
            ordinal_maps = {
                "tx-t_stage": {
                    "T1a": 1, "T1b": 2, "T1c": 3,
                    "T2a": 4, "T2b": 5, "T2c": 6,
                    "T3a": 7, "T3b": 8, "T4": 9,
                },
            }
            if col in ordinal_maps:
                s = df[col].map(ordinal_maps[col])
            else:
                # Fallback: factorize
                codes, _ = pd.factorize(df[col], sort=True)
                s = pd.Series(codes, index=df.index, dtype=float)
                s[s < 0] = np.nan
        return s.rename(col)

    if ptype == "binary":
        norm = _norm(df[col])
        mapping = {
            "oui": 1, "non": 0,
            "true": 1, "false": 0,
            "yes": 1, "no": 1,
            "positive": 1, "positif": 1,
            "negative": 0, "négative": 0, "négatif": 0,
        }
        s = norm.map(mapping)
        # try numeric fallback
        if s.isna().all():
            s = pd.to_numeric(df[col], errors="coerce")
        return s.rename(col)

    if ptype == "categorical":
        norm = _norm(df[col])
        if col == "tx-type":
            norm = df[col].map(_TX_SHORT).fillna(df[col])
        else:
            norm = df[col].astype(str).str.strip()
        dummies = pd.get_dummies(norm, prefix=col, drop_first=True, dtype=float)
        return dummies

    raise ValueError(f"Unknown predictor type: {ptype!r}")


# ---------------------------------------------------------------------------
# Univariate logistic regression
# ---------------------------------------------------------------------------


def univariate_screen(
    df: pd.DataFrame,
    outcome_col: str = "rec_mri-result",
    predictors: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Run univariate logistic regression for each candidate predictor.

    Returns
    -------
    DataFrame with columns: predictor, label, or, ci_lower, ci_upper,
    p_value, p_adj (Benjamini-Hochberg), n, n_events.
    """
    if predictors is None:
        predictors = CANDIDATE_PREDICTORS

    y = prepare_outcome(df, outcome_col)
    rows: list[dict[str, Any]] = []

    for col, label, ptype in predictors:
        if col not in df.columns:
            continue

        X = _encode_predictor(df, col, ptype)
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # combine and drop missing
        data = pd.concat([y.rename("_y"), X], axis=1).dropna()
        if len(data) < 10 or data["_y"].nunique() < 2:
            continue

        y_fit = data["_y"]
        X_fit = sm.add_constant(data.drop(columns="_y"))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y_fit, X_fit).fit(disp=0, maxiter=100)
        except Exception:
            continue

        # Extract OR and p-value for each non-constant predictor
        for param in model.params.index:
            if param == "const":
                continue
            coef = model.params[param]
            ci = model.conf_int().loc[param]
            # Cap extreme coefficients to avoid overflow in exp()
            or_val = np.exp(np.clip(coef, -20, 20))
            ci_lo = np.exp(np.clip(ci[0], -20, 20))
            ci_hi = np.exp(np.clip(ci[1], -20, 20))
            rows.append({
                "predictor": col,
                "param": param,
                "label": label,
                "or": or_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": model.pvalues[param],
                "n": len(data),
                "n_events": int(y_fit.sum()),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        _, p_adj, _, _ = multipletests(result["p_value"], method="fdr_bh")
        result["p_adj"] = p_adj
    return result


# ---------------------------------------------------------------------------
# Multivariable logistic regression
# ---------------------------------------------------------------------------


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute variance inflation factors for a design matrix (without constant)."""
    if X.shape[1] < 2:
        return pd.DataFrame({"variable": X.columns, "VIF": [1.0] * X.shape[1]})
    X_c = sm.add_constant(X).astype(float)
    vifs = []
    for i, col in enumerate(X_c.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X_c.values, i)
        except Exception:
            vif = np.nan
        vifs.append({"variable": col, "VIF": vif})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)


def build_multivariable_model(
    df: pd.DataFrame,
    predictor_cols: list[tuple[str, str, str]] | None = None,
    outcome_col: str = "rec_mri-result",
    vif_threshold: float = 5.0,
) -> dict[str, Any]:
    """Build a multivariable logistic regression model.

    Parameters
    ----------
    df:              Analysis-ready DataFrame.
    predictor_cols:  List of (column, label, type) tuples. If None, uses
                     predictors significant at p < 0.05 from univariate screen.
    outcome_col:     Outcome column name.
    vif_threshold:   Maximum acceptable VIF (default 5.0).

    Returns
    -------
    Dict with keys:
        model          — fitted statsmodels Logit result
        summary        — DataFrame of OR, CI, p-value per predictor
        vif            — VIF table
        n              — sample size used
        n_events       — number of MRI+ cases
        aic            — AIC
        pseudo_r2      — McFadden pseudo-R²
        hosmer_lemeshow — (statistic, p_value) tuple
        roc_auc        — apparent AUC
    """
    from sklearn.metrics import roc_auc_score

    y = prepare_outcome(df, outcome_col)

    # If no predictors specified, use significant univariate results or
    # fall back to clinically-guided CAPRA components
    if predictor_cols is None:
        uni = univariate_screen(df, outcome_col)
        sig = uni[uni["p_value"] < 0.05]["predictor"].unique().tolist()
        predictor_cols = [
            (col, label, ptype)
            for col, label, ptype in CANDIDATE_PREDICTORS
            if col in sig
        ]
        # Fallback: clinically guided predictors (CAPRA components + time to BF)
        if not predictor_cols:
            _FALLBACK = {
                "tx-age", "psa-val", "tx-gleason_total", "tx-t_stage",
                "tx-biopsy_positive_ratio", "bf-time_to_bf-days",
            }
            predictor_cols = [
                (col, label, ptype)
                for col, label, ptype in CANDIDATE_PREDICTORS
                if col in _FALLBACK
            ]

    if not predictor_cols:
        raise ValueError("No predictors selected for multivariable model.")

    # Encode all predictors
    X_parts = []
    for col, label, ptype in predictor_cols:
        if col not in df.columns:
            continue
        encoded = _encode_predictor(df, col, ptype)
        if isinstance(encoded, pd.Series):
            encoded = encoded.to_frame()
        X_parts.append(encoded)

    X = pd.concat(X_parts, axis=1)
    data = pd.concat([y.rename("_y"), X], axis=1).dropna()

    if len(data) < 10 or data["_y"].nunique() < 2:
        raise ValueError(
            f"Insufficient data after dropping missing values: N={len(data)}, "
            f"events={int(data['_y'].sum()) if len(data) else 0}."
        )

    y_fit = data["_y"]
    X_fit = data.drop(columns="_y")

    # Drop zero-variance columns (can arise from dummies after dropna)
    X_fit = X_fit.loc[:, X_fit.nunique() > 1]

    # VIF check and iterative removal
    vif_table = compute_vif(X_fit)
    removed = []
    while (
        len(vif_table) > 0
        and vif_table["VIF"].max() > vif_threshold
        and len(X_fit.columns) > 1
    ):
        worst = vif_table.iloc[0]["variable"]
        removed.append(worst)
        X_fit = X_fit.drop(columns=worst)
        data = data.drop(columns=worst)
        if X_fit.shape[1] == 0:
            break
        vif_table = compute_vif(X_fit)

    X_fit_c = sm.add_constant(X_fit)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y_fit, X_fit_c).fit(disp=0, maxiter=200)

    # Summary table
    summary_rows = []
    for param in model.params.index:
        if param == "const":
            continue
        or_val = np.exp(model.params[param])
        ci = model.conf_int().loc[param]
        summary_rows.append({
            "predictor": param,
            "or": or_val,
            "ci_lower": np.exp(ci[0]),
            "ci_upper": np.exp(ci[1]),
            "p_value": model.pvalues[param],
        })
    summary = pd.DataFrame(summary_rows)

    # Hosmer-Lemeshow test
    hl = _hosmer_lemeshow(y_fit, model.predict(X_fit_c))

    # AUC
    try:
        auc = roc_auc_score(y_fit, model.predict(X_fit_c))
    except Exception:
        auc = np.nan

    return {
        "model": model,
        "summary": summary,
        "vif": vif_table,
        "removed_vif": removed,
        "n": len(data),
        "n_events": int(y_fit.sum()),
        "aic": model.aic,
        "pseudo_r2": model.prsquared,
        "hosmer_lemeshow": hl,
        "roc_auc": auc,
    }


def _hosmer_lemeshow(
    y_true: pd.Series, y_pred: pd.Series, g: int = 10
) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit test.

    Returns (chi2_statistic, p_value).
    """
    data = pd.DataFrame({"y": y_true.values, "p": y_pred.values})
    try:
        data["group"] = pd.qcut(data["p"], q=g, duplicates="drop")
    except ValueError:
        return (np.nan, np.nan)

    agg = data.groupby("group", observed=True).agg(
        obs=("y", "sum"),
        n=("y", "count"),
        pred=("p", "sum"),
    )

    hl_stat = (
        ((agg["obs"] - agg["pred"]) ** 2)
        / (agg["pred"] * (1 - agg["pred"] / agg["n"]))
    ).sum()

    n_groups = len(agg)
    p_val = 1 - sp_stats.chi2.cdf(hl_stat, max(n_groups - 2, 1))
    return (float(hl_stat), float(p_val))


# ---------------------------------------------------------------------------
# Bootstrap internal validation
# ---------------------------------------------------------------------------


def bootstrap_auc(
    df: pd.DataFrame,
    predictor_cols: list[tuple[str, str, str]],
    outcome_col: str = "rec_mri-result",
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap-validated AUC for the multivariable model.

    Returns
    -------
    Dict with: apparent_auc, corrected_auc, optimism, ci_lower, ci_upper.
    """
    from sklearn.metrics import roc_auc_score

    y = prepare_outcome(df, outcome_col)
    X_parts = []
    for col, _, ptype in predictor_cols:
        if col not in df.columns:
            continue
        encoded = _encode_predictor(df, col, ptype)
        if isinstance(encoded, pd.Series):
            encoded = encoded.to_frame()
        X_parts.append(encoded)
    X = pd.concat(X_parts, axis=1)
    full = pd.concat([y.rename("_y"), X], axis=1).dropna()

    y_full = full["_y"]
    X_full = sm.add_constant(full.drop(columns="_y"))

    # Apparent AUC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_full = sm.Logit(y_full, X_full).fit(disp=0, maxiter=200)
    apparent_auc = roc_auc_score(y_full, model_full.predict(X_full))

    rng = np.random.RandomState(random_state)
    optimisms = []

    for _ in range(n_boot):
        idx = rng.choice(len(full), size=len(full), replace=True)
        boot = full.iloc[idx]
        y_b = boot["_y"]
        X_b = sm.add_constant(boot.drop(columns="_y"))

        if y_b.nunique() < 2:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_b = sm.Logit(y_b, X_b).fit(disp=0, maxiter=200)
            auc_boot = roc_auc_score(y_b, m_b.predict(X_b))
            auc_orig = roc_auc_score(y_full, m_b.predict(X_full))
            optimisms.append(auc_boot - auc_orig)
        except Exception:
            continue

    optimism = np.mean(optimisms) if optimisms else np.nan
    corrected = apparent_auc - optimism if not np.isnan(optimism) else np.nan

    # Bootstrap CI for corrected AUC
    boot_aucs = [apparent_auc - o for o in optimisms]
    ci_lo = np.percentile(boot_aucs, 2.5) if boot_aucs else np.nan
    ci_hi = np.percentile(boot_aucs, 97.5) if boot_aucs else np.nan

    return {
        "apparent_auc": apparent_auc,
        "corrected_auc": corrected,
        "optimism": optimism,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_successful_boots": len(optimisms),
    }
