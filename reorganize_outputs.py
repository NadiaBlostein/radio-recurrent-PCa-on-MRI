"""
Reorganize html_data into publication-ready filenames and export PNG versions.
- Renames/copies HTML files to match manuscript figure/table schema
- Splits multi-tab HTML files (t1-descriptive_stats, fig3-stratification)
- Extracts embedded Plotly figures and base64 PNGs to produce png_data/
"""

import os
import re
import json
import shutil
import base64
from pathlib import Path

import plotly.graph_objects as go

ROOT    = Path(__file__).parent
HTML_DIR = ROOT / "html_data"
PNG_DIR  = ROOT / "png_data"
PNG_DIR.mkdir(exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────

STANDALONE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  {styles}
</head>
<body>
  {body}
</body>
</html>
"""

BASE_STYLES = """<style>
    body { margin:0; padding:20px; background:#111111;
           font-family:"Open Sans",verdana,arial,sans-serif; }
    h2 { color:white; margin-bottom:15px; }
    h3 { color:white; margin-top:30px; }
    .table-frame { margin-top:10px; }
    .table-frame iframe { width:100%; border:none; min-height:120px; background:#1a1a2e; }
    img { max-width:100%; display:block; margin:10px auto; }
  </style>"""


def wrap_standalone(title, body_html):
    return STANDALONE_TEMPLATE.format(title=title, styles=BASE_STYLES, body=body_html)


def extract_balanced(text, start, open_c, close_c):
    depth = 0
    for i, c in enumerate(text[start:]):
        if c == open_c:  depth += 1
        elif c == close_c: depth -= 1
        if depth == 0:
            return text[start:start+i+1], start+i+1
    return None, -1


def plotly_panel_from_script(script_text):
    """Return (data, layout) parsed from a Plotly.newPlot() call in a script block."""
    idx = script_text.find("Plotly.newPlot")
    if idx == -1:
        return None, None
    arr_start = script_text.index("[", idx)
    data_str, after_data = extract_balanced(script_text, arr_start, "[", "]")
    rest = script_text[after_data:]
    obj_offset = re.search(r"\{", rest).start()
    layout_str, _ = extract_balanced(rest, obj_offset, "{", "}")
    data   = json.loads(data_str)
    layout = json.loads(layout_str)
    layout.pop("template", None)   # strip deprecated template to avoid version errors
    return data, layout


def save_plotly_png(data, layout, png_path, width=1400, height=900):
    try:
        fig = go.Figure(data=data, layout=layout)
        fig.write_image(str(png_path), width=width, height=height)
        print(f"  [plotly]  {png_path.name}")
        return True
    except Exception as e:
        print(f"  [plotly ERR] {png_path.name}: {e}")
        return False


def save_base64_png(section_html, png_path):
    m = re.search(
        r'<img[^>]+src=["\']data:image/(?:png|jpeg|jpg);base64,([^"\']+)["\']',
        section_html, re.DOTALL
    )
    if m:
        png_path.write_bytes(base64.b64decode(m.group(1)))
        print(f"  [base64]  {png_path.name}")
        return True
    return False


def extract_table_html_from_script(html_content):
    """
    Extract the first inline HTML table string from a script block.
    Handles two patterns:
      - var html = "...";          (t2-missing_data style)
      - d.write("...");            (composite iframe style)
    Returns the decoded HTML string, or None.
    """
    import json as _json
    decoder = _json.JSONDecoder()
    for var_prefix in ("var html = ", "d.write("):
        idx = html_content.find(var_prefix)
        if idx == -1:
            continue
        str_start = idx + len(var_prefix)
        try:
            value, _ = decoder.raw_decode(html_content, str_start)
            if isinstance(value, str):
                return value
        except Exception:
            pass
    return None


def html_table_to_plotly_png(table_html, png_path, width=1200, height=None):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(table_html, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        print(f"  [skip]    No table rows found for {png_path.name}")
        return False
    headers   = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    body_rows = [[c.get_text(strip=True) for c in r.find_all(["th", "td"])] for r in rows[1:]]
    col_data  = [[] for _ in headers]
    for row in body_rows:
        for i, val in enumerate(row):
            if i < len(col_data):
                col_data[i].append(val)
    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color="#0f0f23",
                    font=dict(color="#e0e0e0", size=12), align="left", line_color="#333"),
        cells=dict(values=col_data,  fill_color="#1a1a2e",
                   font=dict(color="#e0e0e0", size=11), align="left", line_color="#333"),
    )])
    fig.update_layout(paper_bgcolor="#111111", plot_bgcolor="#111111",
                      margin=dict(l=20, r=20, t=40, b=20))
    h = height or max(400, 30 * len(body_rows) + 120)
    try:
        fig.write_image(str(png_path), width=width, height=h)
        print(f"  [table]   {png_path.name}")
        return True
    except Exception as e:
        print(f"  [table ERR] {png_path.name}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. Figure 1 — Flow Diagram  (fig1_sankey.html)
# ══════════════════════════════════════════════════════════════════════════════

src = HTML_DIR / "fig1_sankey.html"
dst = HTML_DIR / "Figure_1_Flow_Diagram.html"
shutil.copy2(src, dst)
print(f"Copied  {src.name} → {dst.name}")

content = src.read_text(encoding="utf-8")
scripts = re.findall(r"<script[^>]*>(.*?)</script>", content, re.DOTALL)
for s in scripts:
    if "Plotly.newPlot" in s:
        data, layout = plotly_panel_from_script(s)
        save_plotly_png(data, layout, PNG_DIR / "Figure_1_Flow_Diagram.png")
        break


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tables 1, 2, S1, S2  —  split from t1-descriptive_stats.html
# ══════════════════════════════════════════════════════════════════════════════

t1_content = (HTML_DIR / "t1-descriptive_stats.html").read_text(encoding="utf-8")
m = re.search(r"var tables = (\{.*?\});", t1_content, re.DOTALL)
tables_dict = json.loads(m.group(1))

TABLE_MAP = {
    "rec_mri-result":   ("Table_1_Descriptive_Stats_MRI_Result",
                         "Table 1 — Descriptive Statistics by MRI Recurrence Result"),
    "biopsy-result":    ("Table_2_Descriptive_Stats_Biopsy_Result",
                         "Table 2 — Descriptive Statistics by Biopsy Result"),
    "tx-type":          ("Table_S1_Descriptive_Stats_Treatment_Type",
                         "Table S1 — Descriptive Statistics by Treatment Type"),
    "capra-risk_group": ("Table_S2_Descriptive_Stats_CAPRA_Group",
                         "Table S2 — Descriptive Statistics by CAPRA Risk Group"),
}

for key, (stem, title) in TABLE_MAP.items():
    table_html = tables_dict[key]
    body = (
        f'<h2>{title}</h2>\n'
        f'<div class="table-frame"><iframe id="table-iframe"></iframe></div>\n'
        f'<script>(function(){{\n'
        f'  var f=document.getElementById("table-iframe");\n'
        f'  var d=f.contentDocument||f.contentWindow.document;\n'
        f'  d.open(); d.write({json.dumps(table_html)}); d.close();\n'
        f'}})();</script>'
    )
    html_path = HTML_DIR / f"{stem}.html"
    html_path.write_text(wrap_standalone(title, body), encoding="utf-8")
    print(f"Created {html_path.name}")
    html_table_to_plotly_png(table_html, PNG_DIR / f"{stem}.png", width=1400)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Table S3 — Missing data report  (t2-missing_data.html)
# ══════════════════════════════════════════════════════════════════════════════

src = HTML_DIR / "t2-missing_data.html"
dst = HTML_DIR / "Table_S3_Missing_Data_Report.html"
shutil.copy2(src, dst)
print(f"Copied  {src.name} → {dst.name}")
t2_raw = src.read_text(encoding="utf-8")
t2_table_html = extract_table_html_from_script(t2_raw) or t2_raw
html_table_to_plotly_png(t2_table_html, PNG_DIR / "Table_S3_Missing_Data_Report.png", width=1000)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Table S4 + Figures S1/S2/S3 — split from fig3-stratification.html
# ══════════════════════════════════════════════════════════════════════════════

fig3_content = (HTML_DIR / "fig3-stratification.html").read_text(encoding="utf-8")
sec_positions = [m.start() for m in re.finditer(r'<div class="section">', fig3_content)]

def get_section(positions, content, idx):
    s = positions[idx]
    e = positions[idx+1] if idx+1 < len(positions) else len(content)
    return content[s:e]

SECTION_MAP = [
    (0, "Table_S4_Crosstab_Biopsy_vs_MRI",
        "Table S4 — Cross-tabulation: Biopsy Result vs MRI Detection"),
    (1, "Figure_S1_BF_to_MRI_Lag",
        "Figure S1 — BF-to-MRI Recurrence Lag (Biopsy-Positive Patients)"),
    (2, "Figure_S2_Serial_MRI_Followup",
        "Figure S2 — Serial MRI Follow-up Visits per Patient"),
    (3, "Figure_S3_PSA_Kinetics",
        "Figure S3 — PSA Kinetics at Recurrence MRI"),
]

for sec_idx, stem, title in SECTION_MAP:
    sec_html = get_section(sec_positions, fig3_content, sec_idx)
    html_path = HTML_DIR / f"{stem}.html"
    html_path.write_text(wrap_standalone(title, f'<h2>{title}</h2>\n{sec_html}'), encoding="utf-8")
    print(f"Created {html_path.name}")

    png_path = PNG_DIR / f"{stem}.png"

    # Try base64 image first (sections B/C/D)
    if save_base64_png(sec_html, png_path):
        continue

    # Section A: table embedded via inline script
    table_html = extract_table_html_from_script(sec_html)
    if table_html:
        html_table_to_plotly_png(table_html, png_path, width=900)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Figure S6 — Key Predictors Explorer  (fig2-explorer.html)
# ══════════════════════════════════════════════════════════════════════════════

src = HTML_DIR / "fig2-explorer.html"
dst = HTML_DIR / "Figure_S6_Key_Predictors_by_MRI_Result.html"
shutil.copy2(src, dst)
print(f"Copied  {src.name} → {dst.name}")
# Interactive explorer — no single representative static panel; skip PNG


# ══════════════════════════════════════════════════════════════════════════════
# 6. Figures 2, 3, 4, S4, S5 — individual panels from fig4-dashboard.html
#    (do NOT modify fig4-dashboard.html itself)
# ══════════════════════════════════════════════════════════════════════════════

fig4_content = (HTML_DIR / "fig4-dashboard.html").read_text(encoding="utf-8")
fig4_scripts = re.findall(r"<script[^>]*>(.*?)</script>", fig4_content, re.DOTALL)

# Map panel title keywords → (html_stem, png_stem, panel_description)
# Determined by inspecting script titles:
#   Script 1: "MRI-Positive Recurrence Prevalence"         → Figure 2 (part A)
#   Script 2: "MRI vs Biopsy Agreement (2x2 Table)"       → Figure 2 (part B)
#   Script 3: "Diagnostic Accuracy of MRI (vs Biopsy)"    → Figure 2 (part C, main)
#   Script 4: "Univariate Logistic Regression …"          → Figure 3
#   Script 5: "Key Predictors by MRI Result"              → Figure S6 (skip — already mapped)
#   Script 6: "Multivariable Logistic Regression …"       → Figure 4
#   Script 7: "ROC Curve …"                               → Figure S4
#   Script 8: "Calibration Plot"                          → Figure S5

PANEL_TITLE_STEM = {
    "diagnostic accuracy":           "Figure_2_Diagnostic_Accuracy",
    "univariate logistic regression": "Figure_3_Univariate_Logistic_Regression",
    "multivariate logistic regression": "Figure_4_Multivariate_Logistic_Regression",
    "multivariable logistic regression": "Figure_4_Multivariate_Logistic_Regression",
    # "roc curve" handled separately below as a dark-themed standalone
    "calibration plot":              "Figure_S5_Calibration_Plot",
}

DASHBOARD_STEMS = [
    "Figure_2_Diagnostic_Accuracy",
    "Figure_3_Univariate_Logistic_Regression",
    "Figure_4_Multivariate_Logistic_Regression",
    # Figure_S4_ROC_Curve handled separately below as a dark-themed standalone
    "Figure_S5_Calibration_Plot",
]

# Extract individual panels and create standalone HTML + PNG files
exported = set()
for s in fig4_scripts:
    if "Plotly.newPlot" not in s:
        continue
    try:
        data, layout = plotly_panel_from_script(s)
    except Exception:
        continue
    title_val = layout.get("title", {})
    if isinstance(title_val, dict):
        title_val = title_val.get("text", "")
    title_lower = str(title_val).lower()

    stem = None
    for keyword, candidate in PANEL_TITLE_STEM.items():
        if keyword in title_lower and candidate not in exported:
            stem = candidate
            break

    if stem:
        # Save PNG
        save_plotly_png(data, layout, PNG_DIR / f"{stem}.png")
        exported.add(stem)

        # Create standalone dark-themed HTML for this figure
        BG          = "rgb(17,17,17)"
        TITLE_TXT   = "#f2f5fa"

        fig = go.Figure(data=data, layout=layout)
        # Ensure dark theme settings
        fig.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color="#c8d4e3"),
        )

        plotly_div = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True},
        )

        title_text = title_val if title_val else stem.replace("Figure_", "").replace("_", " ")
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_text}</title>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      background: #111111;
      font-family: "Open Sans", verdana, arial, sans-serif;
    }}
    h1 {{
      color: white;
      text-align: center;
      margin-bottom: 5px;
    }}
  </style>
</head>
<body>
  <h1>{title_text}</h1>
  {plotly_div}
</body>
</html>
"""
        (HTML_DIR / f"{stem}.html").write_text(html_content, encoding="utf-8")
        print(f"Created {stem}.html (standalone)")

# Report any panels not exported
missing = set(DASHBOARD_STEMS) - exported
if missing:
    print(f"  [warn]    No panel matched for: {missing}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figure S4 — ROC Curve (dark-themed standalone, HTML + PNG in sync)
# ══════════════════════════════════════════════════════════════════════════════

FIG_S4_STEM  = "Figure_S4_ROC_Curve"
FIG_S4_TITLE = "ROC Curve — Multivariable Model"
FIG_S4_SUBTITLE = (
    "Receiver operating characteristic curve for the multivariable logistic "
    "regression model predicting MRI-positive recurrence."
)

# Pull the ROC panel data out of the dashboard (same source-of-truth as before)
roc_data = None
for s in fig4_scripts:
    if "Plotly.newPlot" not in s:
        continue
    try:
        d, l = plotly_panel_from_script(s)
    except Exception:
        continue
    title_val = l.get("title", {})
    if isinstance(title_val, dict):
        title_val = title_val.get("text", "")
    if "roc curve" in str(title_val).lower():
        roc_data = d
        break

if roc_data is None:
    print(f"  [warn]    Could not extract ROC panel for {FIG_S4_STEM}")
else:
    # Dark-theme colors (matches the rest of the dashboard/figure set)
    BG          = "rgb(17,17,17)"
    AXIS_GRID   = "#283442"
    AXIS_LINE   = "#506784"
    AXIS_TXT    = "#c8d4e3"
    TITLE_TXT   = "#f2f5fa"
    LEGEND_TXT  = "#e0e0e0"
    LEGEND_BD   = "#506784"
    ROC_COLOR   = "#EF553B"
    ROC_FILL    = "rgba(239, 85, 59, 0.25)"
    DIAG_COLOR  = "#8892a6"

    # Preserve ordering of traces exactly as in the dashboard panel:
    #   0: ROC curve (with AUC), 1: Random-chance diagonal
    auc_value = 0.675
    for t in roc_data:
        nm = t.get("name", "")
        if nm.startswith("ROC"):
            # extract AUC from the existing name to stay in sync with the analysis
            import re as _re
            m = _re.search(r"AUC\s*=\s*([\d.]+)", nm)
            if m:
                auc_value = float(m.group(1))

    roc_trace  = next(t for t in roc_data if str(t.get("name", "")).startswith("ROC"))
    rand_trace = next(t for t in roc_data if "Random" in str(t.get("name", "")))

    fig_s4 = go.Figure()
    fig_s4.add_trace(go.Scatter(
        x=roc_trace["x"], y=roc_trace["y"],
        mode="lines",
        line=dict(color=ROC_COLOR, width=2.8, shape="hv"),
        fill="tozeroy",
        fillcolor=ROC_FILL,
        name=f"ROC (AUC = {auc_value:.3f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig_s4.add_trace(go.Scatter(
        x=rand_trace["x"], y=rand_trace["y"],
        mode="lines",
        line=dict(color=DIAG_COLOR, dash="dash", width=1.8),
        name="Random (AUC = 0.500)",
        hoverinfo="skip",
    ))

    fig_s4.update_layout(
        title=dict(
            text=FIG_S4_TITLE,
            font=dict(color=TITLE_TXT, size=18),
            x=0.05,
        ),
        xaxis=dict(
            title=dict(text="False Positive Rate (1 − Specificity)",
                       font=dict(color=AXIS_TXT, size=13)),
            tickfont=dict(color=AXIS_TXT, size=12),
            range=[0, 1],
            gridcolor=AXIS_GRID,
            linecolor=AXIS_LINE,
            zerolinecolor=AXIS_GRID,
            showline=True, mirror=True,
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate (Sensitivity)",
                       font=dict(color=AXIS_TXT, size=13)),
            tickfont=dict(color=AXIS_TXT, size=12),
            range=[0, 1.02],
            gridcolor=AXIS_GRID,
            linecolor=AXIS_LINE,
            zerolinecolor=AXIS_GRID,
            showline=True, mirror=True,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=AXIS_TXT),
        legend=dict(
            x=0.98, y=0.04,
            xanchor="right", yanchor="bottom",
            bgcolor="rgba(17,17,17,0.75)",
            bordercolor=LEGEND_BD, borderwidth=1,
            font=dict(color=LEGEND_TXT, size=12),
        ),
        annotations=[dict(
            x=0.02, y=0.98, xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            showarrow=False,
            align="left",
            bgcolor="rgba(17,17,17,0.75)",
            bordercolor=LEGEND_BD, borderwidth=1,
            font=dict(color=LEGEND_TXT, size=12),
            text=f"AUC = {auc_value:.3f}",
        )],
        height=550,
        margin=dict(l=70, r=40, t=70, b=70),
        showlegend=True,
    )

    # ── Write standalone dark-themed HTML (matches Figure_4 styling) ──
    plotly_div_s4 = fig_s4.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )
    html_s4 = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{FIG_S4_TITLE}</title>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      background: #111111;
      font-family: "Open Sans", verdana, arial, sans-serif;
    }}
    h1 {{
      color: white;
      text-align: center;
      margin-bottom: 5px;
    }}
    .subtitle {{
      color: #999;
      text-align: center;
      margin-bottom: 20px;
    }}
  </style>
</head>
<body>
  <h1>{FIG_S4_TITLE}</h1>
  <p class="subtitle">{FIG_S4_SUBTITLE}</p>
  {plotly_div_s4}
</body>
</html>
"""
    (HTML_DIR / f"{FIG_S4_STEM}.html").write_text(html_s4, encoding="utf-8")
    print(f"Created {FIG_S4_STEM}.html (dark-themed standalone)")

    # ── Write matching dark-themed PNG from the SAME figure object ──
    try:
        fig_s4.write_image(str(PNG_DIR / f"{FIG_S4_STEM}.png"), width=1400, height=900)
        print(f"  [plotly]  {FIG_S4_STEM}.png")
    except Exception as e:
        print(f"  [plotly ERR] {FIG_S4_STEM}.png: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\nDone.")
print(f"\nHTML files in html_data/:")
for f in sorted(HTML_DIR.iterdir()):
    if f.suffix == ".html":
        marker = "  NEW " if not f.name.startswith(("fig", "t1", "t2")) else "  orig"
        print(f"{marker}  {f.name}")

print(f"\nPNG files in png_data/:")
for f in sorted(PNG_DIR.iterdir()):
    print(f"       {f.name}")
