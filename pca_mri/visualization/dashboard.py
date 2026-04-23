"""
pca_mri.visualization.dashboard — interactive results dashboard.

Produces a multi-panel Plotly figure summarising all statistical analysis
results in a single view, suitable for embedding in a Jupyter notebook.

Functions
---------
build_dashboard(df)    Returns a dict of Plotly figures keyed by panel name.
show_dashboard(df)     Renders the full dashboard inline in a notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from pca_mri.analysis import diagnostic, regression

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS = {
    "mri_pos": "#EF553B",   # red
    "mri_neg": "#636EFA",   # blue
    "biopsy_pos": "#EF553B",
    "biopsy_neg": "#636EFA",
    "tp": "#2CA02C",        # green
    "fp": "#FF7F0E",        # orange
    "fn": "#D62728",        # dark red
    "tn": "#1F77B4",        # dark blue
    "sig": "#EF553B",
    "nonsig": "#BABBBD",
    "ldr": "#636EFA",
    "hdr": "#EF553B",
    "rt": "#00CC96",
}


# ---------------------------------------------------------------------------
# Panel builders — each returns a single go.Figure
# ---------------------------------------------------------------------------


def panel_prevalence(df: pd.DataFrame) -> go.Figure:
    """Bar chart: MRI+ prevalence overall and by treatment type."""
    overall = diagnostic.prevalence(df)
    by_tx = diagnostic.prevalence_by_subgroup(df, by="tx-type")

    groups = list(by_tx["group"]) + ["Overall"]
    prevs = list(by_tx["prevalence"]) + [overall["prevalence"]]
    ci_lo = list(by_tx["ci_lower"]) + [overall["ci_lower"]]
    ci_hi = list(by_tx["ci_upper"]) + [overall["ci_upper"]]
    ns = list(by_tx["n_positive"].astype(str) + "/" + by_tx["n_total"].astype(str)) + [
        f"{overall['n_positive']}/{overall['n_total']}"
    ]

    colors = [COLORS["ldr"], COLORS["hdr"], COLORS["rt"], "#AB63FA"]
    # Pad if fewer groups
    while len(colors) < len(groups):
        colors.append("#AB63FA")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups,
        y=[p * 100 for p in prevs],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[(h - p) * 100 for h, p in zip(ci_hi, prevs)],
            arrayminus=[(p - l) * 100 for l, p in zip(ci_lo, prevs)],
        ),
        marker_color=colors[:len(groups)],
        text=[f"{p*100:.1f}%<br>({n})" for p, n in zip(prevs, ns)],
        textposition="outside",
        hovertemplate="%{x}<br>Prevalence: %{y:.1f}%<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>",
        customdata=list(zip([l*100 for l in ci_lo], [h*100 for h in ci_hi])),
    ))
    fig.update_layout(
        title="MRI-Positive Recurrence Prevalence",
        yaxis_title="Prevalence (%)",
        yaxis_range=[0, 105],
        template="plotly_dark",
        height=400,
        showlegend=False,
    )
    return fig


    def panel_prevalence_by_biopsy(df: pd.DataFrame) -> go.Figure:
        """Bar chart: MRI+ prevalence stratified by biopsy result."""
        by_biopsy = diagnostic.prevalence_by_subgroup(df, by="biopsy-result")
        
        groups = list(by_biopsy["group"])
        prevs = list(by_biopsy["prevalence"])
        ci_lo = list(by_biopsy["ci_lower"])
        ci_hi = list(by_biopsy["ci_upper"])
        ns = list(by_biopsy["n_positive"].astype(str) + "/" + by_biopsy["n_total"].astype(str))
        
        colors = [COLORS["biopsy_pos"], COLORS["biopsy_neg"]]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=groups,
            y=[p * 100 for p in prevs],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[(h - p) * 100 for h, p in zip(ci_hi, prevs)],
                arrayminus=[(p - l) * 100 for l, p in zip(ci_lo, prevs)],
            ),
            marker_color=colors[:len(groups)],
            text=[f"{p*100:.1f}%<br>({n})" for p, n in zip(prevs, ns)],
            textposition="outside",
            hovertemplate="%{x}<br>Prevalence: %{y:.1f}%<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>",
            customdata=list(zip([l*100 for l in ci_lo], [h*100 for h in ci_hi])),
        ))
        fig.update_layout(
            title="MRI-Positive Recurrence Prevalence by Biopsy Result",
            yaxis_title="Prevalence (%)",
            yaxis_range=[0, 105],
            template="plotly_dark",
            height=400,
            showlegend=False,
        )
        return fig


    def panel_prevalence_by_treatment(df: pd.DataFrame) -> go.Figure:
        """Bar chart: MRI+ prevalence stratified by treatment type."""
        by_tx = diagnostic.prevalence_by_subgroup(df, by="tx-type")
        
        groups = list(by_tx["group"])
        prevs = list(by_tx["prevalence"])
        ci_lo = list(by_tx["ci_lower"])
        ci_hi = list(by_tx["ci_upper"])
        ns = list(by_tx["n_positive"].astype(str) + "/" + by_tx["n_total"].astype(str))
        
        colors = [COLORS["ldr"], COLORS["hdr"], COLORS["rt"]]
        while len(colors) < len(groups):
            colors.append("#AB63FA")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=groups,
            y=[p * 100 for p in prevs],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[(h - p) * 100 for h, p in zip(ci_hi, prevs)],
                arrayminus=[(p - l) * 100 for l, p in zip(ci_lo, prevs)],
            ),
            marker_color=colors[:len(groups)],
            text=[f"{p*100:.1f}%<br>({n})" for p, n in zip(prevs, ns)],
            textposition="outside",
            hovertemplate="%{x}<br>Prevalence: %{y:.1f}%<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>",
            customdata=list(zip([l*100 for l in ci_lo], [h*100 for h in ci_hi])),
        ))
        fig.update_layout(
            title="MRI-Positive Recurrence Prevalence by Treatment Type",
            yaxis_title="Prevalence (%)",
            yaxis_range=[0, 105],
            template="plotly_dark",
            height=400,
            showlegend=False,
        )
        return fig


    def panel_prevalence_by_capra(df: pd.DataFrame) -> go.Figure:
        """Bar chart: MRI+ prevalence stratified by CAPRA risk group."""
        by_capra = diagnostic.prevalence_by_subgroup(df, by="psa-capra_group")
        
        groups = list(by_capra["group"])
        prevs = list(by_capra["prevalence"])
        ci_lo = list(by_capra["ci_lower"])
        ci_hi = list(by_capra["ci_upper"])
        ns = list(by_capra["n_positive"].astype(str) + "/" + by_capra["n_total"].astype(str))
        
        colors = [COLORS["ldr"], COLORS["sig"], COLORS["hdr"]]
        while len(colors) < len(groups):
            colors.append("#AB63FA")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=groups,
            y=[p * 100 for p in prevs],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[(h - p) * 100 for h, p in zip(ci_hi, prevs)],
                arrayminus=[(p - l) * 100 for l, p in zip(ci_lo, prevs)],
            ),
            marker_color=colors[:len(groups)],
            text=[f"{p*100:.1f}%<br>({n})" for p, n in zip(prevs, ns)],
            textposition="outside",
            hovertemplate="%{x}<br>Prevalence: %{y:.1f}%<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>",
            customdata=list(zip([l*100 for l in ci_lo], [h*100 for h in ci_hi])),
        ))
        fig.update_layout(
            title="MRI-Positive Recurrence Prevalence by CAPRA Risk Group",
            yaxis_title="Prevalence (%)",
            yaxis_range=[0, 105],
            template="plotly_dark",
            height=400,
            showlegend=False,
        )
        return fig


def panel_contingency(df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap: MRI vs biopsy 2x2 table."""
    ct = diagnostic.contingency_table(df)
    z = ct.values
    labels = [
        [f"TP<br>N={z[0,0]}", f"FP<br>N={z[0,1]}"],
        [f"FN<br>N={z[1,0]}", f"TN<br>N={z[1,1]}"],
    ]
    total = z.sum()
    pct = [
        [f"{z[0,0]/total*100:.0f}%", f"{z[0,1]/total*100:.0f}%"],
        [f"{z[1,0]/total*100:.0f}%", f"{z[1,1]/total*100:.0f}%"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Biopsy +", "Biopsy -"],
        y=["MRI +", "MRI -"],
        colorscale=[[0, "#1a1a2e"], [1, "#636EFA"]],
        showscale=False,
        text=[[f"{labels[i][j]}<br>({pct[i][j]})" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        textfont=dict(size=16),
        hovertemplate="MRI: %{y}<br>Biopsy: %{x}<br>N = %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="MRI vs Biopsy Agreement (2x2 Table)",
        xaxis_title="Biopsy Result (Reference Standard)",
        yaxis_title="MRI Result (Index Test)",
        template="plotly_dark",
        height=380,
        yaxis_autorange="reversed",
    )
    return fig


def panel_diagnostic_metrics(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart with 95% CIs for diagnostic accuracy metrics."""
    acc = diagnostic.diagnostic_accuracy(df)
    acc = acc[acc["metric"] != "Cohen's kappa"].reset_index(drop=True)

    fig = go.Figure()
    for _, row in acc.iterrows():
        color = COLORS["tp"] if row["value"] > 0.5 else COLORS["fn"]
        ci_lower = row["ci_lower"]
        ci_upper = row["ci_upper"]
        has_ci = not (np.isnan(ci_lower) or np.isnan(ci_upper))
        fig.add_trace(go.Bar(
            y=[row["metric"]],
            x=[row["value"]],
            orientation="h",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ci_upper - row["value"]],
                arrayminus=[row["value"] - ci_lower],
            ) if has_ci else None,
            marker_color=color,
            showlegend=False,
            hovertemplate=f"{row['metric']}: {row['value']:.3f} [95% CI: {ci_lower:.3f}-{ci_upper:.3f}]<extra></extra>",
        ))

    # Right-aligned annotation column in paper coordinates so labels never
    # overlap the bars or error bars regardless of each metric's value.
    for _, row in acc.iterrows():
        ci_lower = row["ci_lower"]
        ci_upper = row["ci_upper"]
        has_ci = not (np.isnan(ci_lower) or np.isnan(ci_upper))
        label = (
            f"{row['value']:.2f} [{ci_lower:.2f}–{ci_upper:.2f}]"
            if has_ci else f"{row['value']:.2f}"
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.01,
            y=row["metric"],
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color="#e0e0e0", size=11),
        )

    fig.update_layout(
        title=dict(text="Diagnostic Accuracy of MRI (vs Biopsy)", font=dict(color="#f2f5fa")),
        xaxis_title="Value",
        xaxis=dict(range=[0, 1.1], gridcolor="#283442", zerolinecolor="#283442",
                   tickfont=dict(color="#c8d4e3"), title_font=dict(color="#c8d4e3")),
        yaxis=dict(gridcolor="#283442", tickfont=dict(color="#c8d4e3")),
        template="plotly_dark",
        paper_bgcolor="rgb(17,17,17)",
        plot_bgcolor="rgb(17,17,17)",
        height=350,
        barmode="group",
        margin=dict(l=20, r=220, t=60, b=40),
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#808080", opacity=0.5)
    return fig


_TX_TYPE_LABELS = {
    "tx-type_RT":  "  RT vs HDRT (ref)",
    "tx-type_LDR": "  LDR vs HDRT (ref)",
}


def _format_or_ci(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value == 0 or (abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-2)):
        return f"{value:.2e}"
    return f"{value:.2f}"


def panel_forest_univariate(df: pd.DataFrame) -> go.Figure:
    """Forest plot of univariate logistic regression odds ratios."""
    uni = regression.univariate_screen(df)
    if uni.empty:
        fig = go.Figure()
        fig.add_annotation(text="No univariate results available", showarrow=False)
        return fig

    uni = uni.copy()
    uni["or_display"] = uni["or"].clip(lower=0.1, upper=50)
    uni["ci_upper_display"] = uni["ci_upper"].clip(lower=0.1, upper=50)
    uni["ci_lower_display"] = uni["ci_lower"].clip(lower=0.1, upper=50)

    def _display_label(row) -> str:
        if row["param"] in _TX_TYPE_LABELS:
            return _TX_TYPE_LABELS[row["param"]]
        if row["label"] == row["param"] or "tx-type" not in row["param"]:
            return str(row["label"])
        return str(row["param"])

    uni["display_label"] = uni.apply(_display_label, axis=1)

    # Sort: non-tx-type by p-value (descending), then tx-type variables at the end
    uni["is_tx_type"] = uni["param"].str.startswith("tx-type_", na=False)
    uni = uni.sort_values(["is_tx_type", "p_value"], ascending=[True, False]).reset_index(drop=True)

    fig = go.Figure()
    for idx, (_, row) in enumerate(uni.iterrows()):
        sig = row["p_value"] < 0.05
        color = COLORS["sig"] if sig else COLORS["nonsig"]

        fig.add_trace(go.Scatter(
            x=[row["or_display"]],
            y=[row["display_label"]],
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["ci_upper_display"] - row["or_display"]],
                arrayminus=[row["or_display"] - row["ci_lower_display"]],
                color="#c8d4e3",
                thickness=1.5,
                width=6,
            ),
            mode="markers",
            marker=dict(size=10, color=color, symbol="diamond",
                        line=dict(width=1, color="#ffffff")),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['display_label'].strip()}</b><br>"
                f"OR: {row['or']:.2f} [{row['ci_lower']:.2f}-{row['ci_upper']:.2f}]<br>"
                f"p = {row['p_value']:.3f} (adj: {row['p_adj']:.3f})<br>"
                f"N = {row['n']}<extra></extra>"
            ),
        ))

    # Right-margin annotations: OR, 95% CI, and p-value for each predictor.
    for idx, (_, row) in enumerate(uni.iterrows()):
        or_txt = _format_or_ci(row["or"])
        lo_txt = _format_or_ci(row["ci_lower"])
        hi_txt = _format_or_ci(row["ci_upper"])
        p_val = row["p_value"]
        p_txt = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
        text = f"OR {or_txt} ({lo_txt}–{hi_txt}), p = {p_txt}"
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.01,
            y=row["display_label"],
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color="#e0e0e0", size=11),
        )

    fig.add_vline(x=1, line_dash="dash", line_color="#808080", opacity=0.6)
    fig.update_layout(
        title=dict(text="Univariate Logistic Regression — Odds Ratios for MRI+ Recurrence",
                   font=dict(color="#f2f5fa"), x=0.05, xanchor="left"),
        xaxis=dict(
            title=dict(text="Odds Ratio (log scale)", font=dict(color="#c8d4e3")),
            type="log",
            range=[np.log10(0.08), np.log10(60)],
            gridcolor="#283442",
            zerolinecolor="#283442",
            tickfont=dict(color="#c8d4e3", size=11),
            linecolor="#506784",
            showline=True,
            mirror=True,
        ),
        yaxis=dict(
            gridcolor="#283442",
            tickfont=dict(color="#c8d4e3", size=11),
            automargin=True,
            linecolor="#506784",
            showline=True,
            mirror=True,
        ),
        template="plotly_dark",
        paper_bgcolor="rgb(17,17,17)",
        plot_bgcolor="rgb(17,17,17)",
        height=max(420, len(uni) * 38),
        margin=dict(l=20, r=350, t=80, b=70),
    )

    # Significance legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color=COLORS["sig"], symbol="diamond"),
                             name="p < 0.05"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color=COLORS["nonsig"], symbol="diamond"),
                             name="p ≥ 0.05"))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, y=-0.12,
            xanchor="center", yanchor="top",
            bgcolor="rgba(17,17,17,0.6)",
            bordercolor="#506784", borderwidth=1,
            font=dict(color="#e0e0e0", size=11),
        ),
    )
    return fig


# Friendly labels for raw column names used in the multivariable model
_PREDICTOR_LABELS = {
    "tx-age": "Age at treatment",
    "psa-val": "PSA at diagnosis",
    "tx-gleason_total": "Gleason score",
    "tx-t_stage": "Clinical T-stage",
    "tx-biopsy_positive_ratio": "Positive biopsy core ratio",
    "bf-time_to_bf-days": "Time to biochemical failure (days)",
    "psa-capra_total": "CAPRA score",
    "tx-adt": "ADT use",
    "psa-nadir_05": "PSA nadir < 0.5",
    "mri_1-prostate_vol": "Prostate volume at MRI 1 (cc)",
    "mri_1-psa": "PSA at MRI 1 (ng/mL)",
    "psa_diff-rec_mri-days": "PSA difference to recurrence MRI",
    "psa_dt-rec_mri-months": "PSA-DT to recurrence MRI (months)",
}


def panel_multivariable_summary(mv_result: dict) -> go.Figure:
    """Forest plot for the multivariable model results."""
    summary = mv_result["summary"]
    if summary.empty:
        fig = go.Figure()
        fig.add_annotation(text="No multivariate results", showarrow=False)
        return fig

    summary = summary.copy()
    summary["or_display"] = summary["or"].clip(lower=0.1, upper=50)
    summary["ci_upper_display"] = summary["ci_upper"].clip(lower=0.1, upper=50)
    summary["ci_lower_display"] = summary["ci_lower"].clip(lower=0.1, upper=50)
    summary = summary.sort_values("p_value", ascending=False)

    summary["display_label"] = summary["predictor"].map(
        lambda p: _PREDICTOR_LABELS.get(p, p)
    )

    fig = go.Figure()
    for _, row in summary.iterrows():
        sig = row["p_value"] < 0.05
        color = COLORS["sig"] if sig else COLORS["nonsig"]
        fig.add_trace(go.Scatter(
            x=[row["or_display"]],
            y=[row["display_label"]],
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["ci_upper_display"] - row["or_display"]],
                arrayminus=[row["or_display"] - row["ci_lower_display"]],
                color="#c8d4e3",
                thickness=1.5,
                width=6,
            ),
            mode="markers",
            marker=dict(size=12, color=color, symbol="diamond",
                        line=dict(width=1, color="#ffffff")),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['display_label']}</b><br>"
                f"OR: {row['or']:.2f} [{row['ci_lower']:.2f}-{row['ci_upper']:.2f}]<br>"
                f"p = {row['p_value']:.3f}<extra></extra>"
            ),
        ))

    # Right-margin OR (CI), p-value annotations — matches Figure 3 styling.
    for _, row in summary.iterrows():
        or_txt = _format_or_ci(row["or"])
        lo_txt = _format_or_ci(row["ci_lower"])
        hi_txt = _format_or_ci(row["ci_upper"])
        p_val = row["p_value"]
        p_txt = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.01,
            y=row["display_label"],
            text=f"OR {or_txt} ({lo_txt}–{hi_txt}), p = {p_txt}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color="#e0e0e0", size=11),
        )

    fig.add_vline(x=1, line_dash="dash", line_color="#808080", opacity=0.6)

    # Model stats annotation (bottom-left, paper coordinates)
    auc = mv_result.get("roc_auc", np.nan)
    r2 = mv_result.get("pseudo_r2", np.nan)
    hl_stat, hl_p = mv_result.get("hosmer_lemeshow", (np.nan, np.nan))
    stats_text = (
        f"AUC = {auc:.3f} | Pseudo-R² = {r2:.3f}<br>"
        f"Hosmer-Lemeshow: χ² = {hl_stat:.2f}, p = {hl_p:.3f}<br>"
        f"N = {mv_result['n']}, events = {mv_result['n_events']}"
    )
    fig.add_annotation(
        x=0.01, y=-0.22, xref="paper", yref="paper",
        text=stats_text, showarrow=False,
        font=dict(size=11, color="#e0e0e0"), align="left",
        xanchor="left", yanchor="top",
        bgcolor="rgba(17,17,17,0.6)", bordercolor="#506784", borderwidth=1,
    )

    fig.update_layout(
        title=dict(text="Multivariate Logistic Regression — Adjusted Odds Ratios",
                   font=dict(color="#f2f5fa")),
        xaxis=dict(
            title=dict(text="Adjusted Odds Ratio (log scale)",
                       font=dict(color="#c8d4e3")),
            type="log",
            range=[np.log10(0.08), np.log10(60)],
            gridcolor="#283442",
            zerolinecolor="#283442",
            tickfont=dict(color="#c8d4e3"),
        ),
        yaxis=dict(
            gridcolor="#283442",
            tickfont=dict(color="#c8d4e3"),
            automargin=True,
        ),
        template="plotly_dark",
        paper_bgcolor="rgb(17,17,17)",
        plot_bgcolor="rgb(17,17,17)",
        height=max(420, len(summary) * 50),
        margin=dict(l=20, r=320, t=60, b=130),
    )

    # Significance legend (matches Figure 3)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=COLORS["sig"], symbol="diamond"),
                             name="p < 0.05"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=COLORS["nonsig"], symbol="diamond"),
                             name="p ≥ 0.05"))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, y=-0.12,
            xanchor="center", yanchor="top",
            bgcolor="rgba(17,17,17,0.6)",
            bordercolor="#506784", borderwidth=1,
            font=dict(color="#e0e0e0"),
        ),
    )
    return fig


def panel_roc(mv_result: dict) -> go.Figure:
    """ROC curve from the multivariable model."""
    from sklearn.metrics import roc_curve

    model = mv_result["model"]
    y_true = model.model.endog
    y_pred = model.predict()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = mv_result.get("roc_auc", np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color=COLORS["mri_pos"], width=2.5),
        name=f"ROC (AUC = {auc:.3f})",
        fill="tozeroy",
        fillcolor="rgba(239, 85, 59, 0.2)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Random (AUC = 0.5)",
    ))
    fig.update_layout(
        title="ROC Curve — Multivariable Model",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        template="plotly_dark",
        height=420,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        legend=dict(x=0.55, y=0.1),
    )
    return fig


def panel_calibration(mv_result: dict, n_bins: int = 5) -> go.Figure:
    """Calibration plot: predicted vs observed probability."""
    model = mv_result["model"]
    y_true = pd.Series(model.model.endog)
    y_pred = pd.Series(model.predict())

    data = pd.DataFrame({"y": y_true.values, "p": y_pred.values})
    try:
        data["bin"] = pd.qcut(data["p"], q=n_bins, duplicates="drop")
    except ValueError:
        data["bin"] = pd.cut(data["p"], bins=n_bins)

    cal = data.groupby("bin", observed=True).agg(
        pred_mean=("p", "mean"),
        obs_mean=("y", "mean"),
        n=("y", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cal["pred_mean"], y=cal["obs_mean"],
        mode="markers+lines",
        marker=dict(size=cal["n"] * 1.5 + 5, color=COLORS["tp"]),
        name="Observed",
        hovertemplate="Predicted: %{x:.2f}<br>Observed: %{y:.2f}<br>N = %{marker.size:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Perfect calibration",
    ))
    fig.update_layout(
        title="Calibration Plot",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Proportion",
        template="plotly_dark",
        height=400,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
    )
    return fig


def panel_predictor_distributions(df: pd.DataFrame, top_predictors: list[str] | None = None) -> go.Figure:
    """Box plots of top predictors split by MRI result."""
    outcome = regression.prepare_outcome(df)
    df_plot = df.copy()
    df_plot["MRI Result"] = outcome.map({1.0: "MRI+", 0.0: "MRI-"})
    df_plot = df_plot.dropna(subset=["MRI Result"])

    if top_predictors is None:
        top_predictors = ["tx-age", "psa-val", "psa-capra_total", "bf-time_to_bf-days"]

    # Filter to existing continuous columns
    top_predictors = [c for c in top_predictors if c in df.columns]
    if not top_predictors:
        fig = go.Figure()
        fig.add_annotation(text="No predictors available", showarrow=False)
        return fig

    fig = make_subplots(
        rows=1, cols=len(top_predictors),
        subplot_titles=top_predictors,
        horizontal_spacing=0.08,
    )

    for i, col in enumerate(top_predictors, 1):
        for mri_val, color in [("MRI+", COLORS["mri_pos"]), ("MRI-", COLORS["mri_neg"])]:
            vals = pd.to_numeric(
                df_plot.loc[df_plot["MRI Result"] == mri_val, col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()
            fig.add_trace(go.Box(
                y=vals,
                name=mri_val,
                marker_color=color,
                showlegend=(i == 1),
                legendgroup=mri_val,
                boxmean=True,
            ), row=1, col=i)

    fig.update_layout(
        title="Key Predictors by MRI Result",
        template="plotly_dark",
        height=420,
        boxmode="group",
    )
    return fig


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def build_dashboard(
    df: pd.DataFrame,
    run_multivariable: bool = True,
    mv_predictors: list[tuple[str, str, str]] | None = None,
) -> dict[str, go.Figure]:
    """Build all dashboard panels.

    Parameters
    ----------
    df:                 Analysis-ready DataFrame (BF cohort with MRI + biopsy).
    run_multivariable:  Whether to fit the multivariable model (can be slow).
    mv_predictors:      Override predictors for the multivariable model.

    Returns
    -------
    Dict mapping panel name -> Plotly Figure.
    """
    panels = {}

    # --- Objective 1: Prevalence ---
    panels["prevalence"] = panel_prevalence(df)

    # --- Objective 2: Diagnostic accuracy ---
    panels["contingency"] = panel_contingency(df)
    panels["diagnostic_metrics"] = panel_diagnostic_metrics(df)

    # --- Objective 3a: Univariate ---
    panels["forest_univariate"] = panel_forest_univariate(df)

    # --- Predictor distributions ---
    panels["predictor_distributions"] = panel_predictor_distributions(df)

    # --- Objective 3b: Multivariable ---
    if run_multivariable:
        try:
            mv = regression.build_multivariable_model(
                df, predictor_cols=mv_predictors
            )
            panels["forest_multivariable"] = panel_multivariable_summary(mv)
            panels["roc"] = panel_roc(mv)
            panels["calibration"] = panel_calibration(mv)
            panels["_mv_result"] = mv  # stash for downstream use (not a figure)
        except Exception as e:
            print(f"Multivariable model failed: {e}")

    return panels


def show_dashboard(
    df: pd.DataFrame,
    run_multivariable: bool = True,
    mv_predictors: list[tuple[str, str, str]] | None = None,
) -> dict[str, go.Figure]:
    """Build and display the full dashboard in a Jupyter notebook.

    Returns the panels dict for further inspection.
    """
    panels = build_dashboard(df, run_multivariable, mv_predictors)

    display_order = [
        "prevalence",
        "contingency",
        "diagnostic_metrics",
        "predictor_distributions",
        "forest_univariate",
        "forest_multivariable",
        "roc",
        "calibration",
    ]

    for name in display_order:
        if name in panels and isinstance(panels[name], go.Figure):
            panels[name].show()

    return panels
