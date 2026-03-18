"""Export Plotly figures and styled tables as standalone HTML files.

All exported pages use a consistent dark-themed style matching the project's
existing ``export_html.py`` look and feel.

Usage
-----
>>> from pca_mri.visualization import export_html
>>> export_html.save_figure(fig, "My Title", "html_data/fig1.html")
>>> export_html.save_table(styled, "My Table", "html_data/table1.html",
...                        subtitle="Some description.")
>>> export_html.save_descriptive_stats_tables(
...     {"tx-type": t1_a_styled, "biopsy-result": t1_b_styled},
...     "html_data/descriptive_stats.html",
... )
>>> export_html.save_interactive_explorer(df, "html_data/explorer.html")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

# ── Shared CSS (matches project dark-theme styling) ──────────────────────────
_CSS = """
    body {
      margin: 0;
      padding: 20px;
      background: #111111;
      font-family: "Open Sans", verdana, arial, sans-serif;
    }
    h1 {
      color: white;
      text-align: center;
      margin-bottom: 5px;
    }
    h2 {
      color: white;
      margin-bottom: 15px;
    }
    .subtitle {
      color: #999;
      text-align: center;
      margin-bottom: 20px;
    }
    .cards {
      display: flex;
      justify-content: center;
      gap: 20px;
      margin-top: 20px;
      flex-wrap: wrap;
    }
    .section {
      border-top: 1px solid #333;
      margin-top: 40px;
      padding-top: 20px;
    }
    .desc-text {
      color: #999;
      margin-bottom: 15px;
    }
    label.strat-label {
      color: white;
      display: block;
      margin-bottom: 5px;
    }
    select#strat-select {
      background: #1e1e1e;
      color: white;
      border: 1px solid #555;
      border-radius: 4px;
      padding: 8px 12px;
      font-size: 14px;
      min-width: 320px;
      margin-bottom: 20px;
    }
    .table-frame {
      background: #1a1a2e;
      border-radius: 8px;
      padding: 10px;
      overflow-x: auto;
    }
    .table-frame iframe {
      width: 100%;
      border: none;
      min-height: 600px;
      background: #1a1a2e;
    }
"""


def _ensure_dir(path: str | Path) -> None:
    """Create parent directories for *path* if they don't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ─── Public API ──────────────────────────────────────────────────────────────


def save_figure(
    fig,
    title: str,
    output_path: str | Path,
    *,
    subtitle: str | None = None,
    cards: Sequence[tuple[str, str, str]] | None = None,
    height: int = 650,
) -> None:
    """Export a Plotly figure as a standalone HTML file.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to export.
    title : str
        Page heading (``<h1>``).
    output_path : str | Path
        Destination file path (e.g. ``"html_data/fig1.html"``).
    subtitle : str, optional
        Grey subtitle rendered below the heading.
    cards : list of (label, value, color) tuples, optional
        Summary cards rendered below the figure.
    height : int
        Figure height in pixels (default 650).
    """
    _ensure_dir(output_path)

    fig.update_layout(height=height, width=None, autosize=True)
    plotly_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    subtitle_html = (
        f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    )

    cards_html = ""
    if cards:
        cards_html = '<div class="cards">\n' + "\n".join(
            f"""<div style="background:#1e1e1e; border-left:4px solid {color};
                 border-radius:6px; padding:12px 20px; min-width:110px; text-align:center;">
              <div style="color:{color}; font-size:28px; font-weight:bold;">{value}</div>
              <div style="color:#999; font-size:12px; margin-top:4px;">{label}</div>
            </div>"""
            for label, value, color in cards
        ) + "\n</div>"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>{title}</h1>
  {subtitle_html}
  {plotly_div}
  {cards_html}
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Saved: {output_path}")


def save_table(
    styled_or_html,
    title: str,
    output_path: str | Path,
    *,
    subtitle: str | None = None,
) -> None:
    """Export a styled DataFrame (or raw HTML string) as a standalone HTML file.

    The table is rendered inside an iframe to isolate its styles.

    Parameters
    ----------
    styled_or_html : pandas.io.formats.style.Styler | str
        A ``Styler`` object (calls ``.to_html()``) or a raw HTML string.
    title : str
        Page heading (``<h2>``).
    output_path : str | Path
        Destination file path.
    subtitle : str, optional
        Grey description text rendered above the table.
    """
    _ensure_dir(output_path)

    table_html = (
        styled_or_html
        if isinstance(styled_or_html, str)
        else styled_or_html.to_html()
    )

    subtitle_html = (
        f'<p class="desc-text">{subtitle}</p>' if subtitle else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h2>{title}</h2>
  {subtitle_html}
  <div class="table-frame">
    <iframe id="table-iframe"></iframe>
  </div>

  <script>
    (function() {{
      var html = {json.dumps(table_html)};
      var iframe = document.getElementById('table-iframe');
      var doc = iframe.contentDocument || iframe.contentWindow.document;
      doc.open();
      doc.write(html);
      doc.close();
    }})();
  </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Saved: {output_path}")


def save_table_with_dropdown(
    tables: dict[str, object],
    title: str,
    output_path: str | Path,
    *,
    labels: dict[str, str] | None = None,
    subtitle: str | None = None,
    dropdown_label: str = "Stratify by:",
) -> None:
    """Export multiple styled tables with a dropdown selector.

    Parameters
    ----------
    tables : dict[str, Styler | str]
        Mapping of key → styled table (or HTML string).  The first key is
        shown by default.
    title : str
        Page heading (``<h2>``).
    output_path : str | Path
        Destination file path.
    labels : dict[str, str], optional
        Human-readable labels for each key shown in the dropdown.  Defaults
        to using the keys themselves.
    subtitle : str, optional
        Grey description text above the dropdown.
    dropdown_label : str
        Label text for the dropdown (default ``"Stratify by:"``).
    """
    _ensure_dir(output_path)

    labels = labels or {k: k for k in tables}

    table_html_map: dict[str, str] = {}
    for key, styled in tables.items():
        table_html_map[key] = (
            styled if isinstance(styled, str) else styled.to_html()
        )

    tables_json = json.dumps(table_html_map)

    dropdown_options_html = "\n".join(
        f'      <option value="{key}">{lbl}</option>'
        for key, lbl in labels.items()
    )

    subtitle_html = (
        f'<p class="desc-text">{subtitle}</p>' if subtitle else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h2>{title}</h2>
  {subtitle_html}

  <label class="strat-label" for="strat-select">{dropdown_label}</label>
  <select id="strat-select">
{dropdown_options_html}
  </select>

  <div class="table-frame">
    <iframe id="table-iframe"></iframe>
  </div>

  <script>
    var tables = {tables_json};

    function updateTable() {{
      var key = document.getElementById('strat-select').value;
      var iframe = document.getElementById('table-iframe');
      var html = tables[key] || '<p>No data</p>';
      var doc = iframe.contentDocument || iframe.contentWindow.document;
      doc.open();
      doc.write(html);
      doc.close();
    }}

    document.getElementById('strat-select').addEventListener('change', updateTable);
    updateTable();
  </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Saved: {output_path}")


def save_interactive_explorer(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    cat_cols: list[tuple[str, str]] | None = None,
    cont_cols: list[tuple[str, str]] | None = None,
    title: str = "Distribution Explorer",
    subtitle: str | None = (
        "Select a categorical and continuous variable to explore "
        "their distributions."
    ),
    kde_height: int = 420,
    bar_height: int = 380,
) -> None:
    """Export the interactive KDE + bar-chart explorer as standalone HTML.

    Pre-renders every (categorical × continuous) combination as a Plotly
    JSON spec and embeds two JavaScript dropdowns that swap figures
    client-side via ``Plotly.react()``.  No Python server required.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned patient dataframe.
    output_path : str | Path
        Destination HTML file.
    cat_cols : list of (col, label) tuples, optional
        Categorical variables for the dropdown.  Defaults to all
        ``_CATEGORICAL`` columns present in *df*.
    cont_cols : list of (col, label) tuples, optional
        Continuous variables for the dropdown.  Defaults to all
        ``_CONTINUOUS`` columns present in *df*.
    title : str
        Page heading.
    subtitle : str, optional
        Grey description text below the heading.
    kde_height : int
        Height of the KDE figure in pixels.
    bar_height : int
        Height of the bar chart figure in pixels.
    """
    from pca_mri.visualization.descriptive_plots import (
        plot_kde,
        plot_category_bar,
        plot_histogram,
    )
    from pca_mri.analysis.descriptive import _CATEGORICAL, _CONTINUOUS

    _ensure_dir(output_path)

    if cat_cols is None:
        cat_cols = [(col, lbl) for col, lbl in _CATEGORICAL if col in df.columns]
    if cont_cols is None:
        cont_cols = [(col, lbl) for col, lbl in _CONTINUOUS if col in df.columns]

    # Pre-render bar figures: key = "cat_col|cont_col" → Plotly JSON
    bar_specs: dict[str, str] = {}
    for cat_col, _ in cat_cols:
        for cont_col, cont_label in cont_cols:
            fig = plot_category_bar(df, cat_col, cont_col, height=bar_height)
            # Add mean values to hover text for each category
            for trace in fig.data:
                categories = trace.x
                means = [df[df[cat_col] == cat][cont_col].mean() for cat in categories]
                trace.customdata = means
                trace.hovertemplate = f"<b>%{{x}}</b><br>Mean {cont_label}: %{{customdata:.2f}}<extra></extra>"
            bar_specs[f"{cat_col}|{cont_col}"] = fig.to_json()

    # Pre-render histogram figures: key = "cat_col|cont_col" → Plotly JSON
    hist_specs: dict[str, str] = {}
    for cat_col, _ in cat_cols:
        for cont_col, _ in cont_cols:
            fig = plot_histogram(df, cat_col, cont_col, height=kde_height)
            hist_specs[f"{cat_col}|{cont_col}"] = fig.to_json()


    # Pre-render KDE figures: key = "cat_col|cont_col" → Plotly JSON
    kde_specs: dict[str, str] = {}
    for cat_col, _ in cat_cols:
        for cont_col, _ in cont_cols:
            fig = plot_kde(df, cat_col, cont_col, height=kde_height)
            kde_specs[f"{cat_col}|{cont_col}"] = fig.to_json()

    # Build dropdown HTML
    cat_options_html = "\n".join(
        f'      <option value="{col}">{lbl}</option>'
        for col, lbl in cat_cols
    )
    cont_options_html = "\n".join(
        f'      <option value="{col}">{lbl}</option>'
        for col, lbl in cont_cols
    )

    subtitle_html = (
        f'<p class="desc-text">{subtitle}</p>' if subtitle else ""
    )

    # Wrap each Plotly JSON string so it stays as raw JSON in the JS object
    # kde_specs values are already JSON strings from fig.to_json(), so we
    # embed them as: "key": <raw json> inside a JS object literal.
    def _js_object(specs: dict[str, str]) -> str:
        entries = []
        for key, val in specs.items():
            entries.append(f"    {json.dumps(key)}: {val}")
        return "{\n" + ",\n".join(entries) + "\n  }"

    bar_js = _js_object(bar_specs)
    hist_js = _js_object(hist_specs)
    kde_js = _js_object(kde_specs)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>{_CSS}
    .controls {{
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }}
    .controls label {{
      color: white;
      display: block;
      margin-bottom: 5px;
    }}
    .controls select {{
      background: #1e1e1e;
      color: white;
      border: 1px solid #555;
      border-radius: 4px;
      padding: 8px 12px;
      font-size: 14px;
      min-width: 320px;
    }}
    #bar-plot, #hist-plot, #kde-plot {{
      margin-bottom: 30px;
    }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  {subtitle_html}

  <div class="controls">
    <div>
      <label for="cat-select">Categorical variable:</label>
      <select id="cat-select">
{cat_options_html}
      </select>
    </div>
    <div>
      <label for="cont-select">Continuous variable:</label>
      <select id="cont-select">
{cont_options_html}
      </select>
    </div>
  </div>

  <div id="bar-plot"></div>
  <div id="hist-plot"></div>
  <div id="kde-plot"></div>

  <script>
    var kdeSpecs = {kde_js};
    var barSpecs = {bar_js};
    var histSpecs = {hist_js};

    function update() {{
      var cat  = document.getElementById('cat-select').value;
      var cont = document.getElementById('cont-select').value;
      var comboKey = cat + '|' + cont;

      // Update bar plot (depends on both categorical and continuous)
      var barSpec = barSpecs[comboKey];
      if (barSpec) {{
        Plotly.react('bar-plot', barSpec.data, barSpec.layout);
      }}

      // Update histogram plot
      var histSpec = histSpecs[comboKey];
      if (histSpec) {{
        Plotly.react('hist-plot', histSpec.data, histSpec.layout);
      }}

      // Update KDE plot
      var kdeSpec = kdeSpecs[comboKey];
      if (kdeSpec) {{
        Plotly.react('kde-plot', kdeSpec.data, kdeSpec.layout);
      }}
    }}

    document.getElementById('cat-select').addEventListener('change', update);
    document.getElementById('cont-select').addEventListener('change', update);
    update();
  </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Saved: {output_path}")


def save_composite(
    items: Sequence[tuple[str, object]],
    output_path: str | Path,
    *,
    title: str = "",
    subtitle: str | None = None,
) -> None:
    """Export a sequence of figures and/or tables into a single HTML file.

    Each item is rendered as its own section, stacked vertically with a
    clear heading.  Supported object types:

    * **Plotly figure** (``plotly.graph_objects.Figure``) — embedded via
      ``to_html(full_html=False)``.
    * **Matplotlib figure** (``matplotlib.figure.Figure``) — rasterised to
      a base-64 PNG ``<img>`` tag.
    * **Pandas Styler** (``pd.io.formats.style.Styler``) — rendered inside
      an auto-resizing ``<iframe>`` to isolate table CSS.
    * **Raw HTML string** — inserted verbatim.

    Parameters
    ----------
    items : sequence of (heading, object) tuples
        Each tuple contains a section heading (``str``, may be empty) and
        the renderable object.
    output_path : str | Path
        Destination HTML file.
    title : str
        Page-level ``<h2>`` heading rendered at the top.
    subtitle : str, optional
        Grey description text below the page heading.
    """
    import base64
    import io as _io

    _ensure_dir(output_path)

    subtitle_html = (
        f'<p class="desc-text">{subtitle}</p>' if subtitle else ""
    )

    sections: list[str] = []
    plotly_needed = False
    iframe_counter = 0

    for heading, obj in items:
        heading_html = f'<h3 style="color:white; margin-top:30px;">{heading}</h3>' if heading else ""

        # ── Plotly figure ──
        try:
            import plotly.graph_objects as go
            if isinstance(obj, go.Figure):
                plotly_needed = True
                div = obj.to_html(full_html=False, include_plotlyjs=False)
                sections.append(f'{heading_html}\n{div}')
                continue
        except ImportError:
            pass

        # ── Matplotlib figure ──
        try:
            import matplotlib.figure
            if isinstance(obj, matplotlib.figure.Figure):
                buf = _io.BytesIO()
                obj.savefig(buf, format="png", bbox_inches="tight",
                            facecolor=obj.get_facecolor(), dpi=150)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                buf.close()
                sections.append(
                    f'{heading_html}\n'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%; height:auto; display:block; '
                    f'margin:10px 0;" />'
                )
                continue
        except ImportError:
            pass

        # ── Pandas Styler ──
        if hasattr(obj, "to_html") and not isinstance(obj, str):
            table_html = obj.to_html()
            iframe_id = f"composite-iframe-{iframe_counter}"
            iframe_counter += 1
            sections.append(
                f'{heading_html}\n'
                f'<div class="table-frame">'
                f'<iframe id="{iframe_id}" style="width:100%; border:none; '
                f'min-height:120px; background:#1a1a2e;"></iframe></div>\n'
                f'<script>(function(){{'
                f'var f=document.getElementById("{iframe_id}");'
                f'var d=f.contentDocument||f.contentWindow.document;'
                f'd.open();d.write({json.dumps(table_html)});d.close();'
                f'function resize(){{f.style.height=d.body.scrollHeight+40+"px";}}'
                f'f.onload=resize;setTimeout(resize,200);'
                f'}})()</script>'
            )
            continue

        # ── Raw HTML string ──
        if isinstance(obj, str):
            sections.append(f'{heading_html}\n{obj}')
            continue

        raise TypeError(
            f"Unsupported item type {type(obj).__name__!r}. Expected a Plotly "
            f"figure, matplotlib figure, pandas Styler, or HTML string."
        )

    plotly_js = (
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        if plotly_needed else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  {plotly_js}
  <style>{_CSS}</style>
</head>
<body>
  <h2>{title}</h2>
  {subtitle_html}
  {"".join(f'<div class="section">{s}</div>' for s in sections)}
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Saved: {output_path}")


# Default stratification labels matching export_html_v1.py Section 2
_DEFAULT_STRAT_LABELS: dict[str, str] = {
    "tx-type":          "Stratified by Treatment Type",
    "rec_mri-result":   "Stratified by MRI Recurrence Result",
    "biopsy-result":    "Stratified by Biopsy Result",
    "capra-risk_group": "Stratified by CAPRA Risk Group",
}


def save_descriptive_stats_tables(
    tables: dict[str, object],
    output_path: str | Path,
    *,
    labels: dict[str, str] | None = None,
    title: str = "Descriptive Statistics",
    subtitle: str = (
        "Clinical characteristics table with heatmap colouring "
        "(row-normalised). Select a stratification variable below."
    ),
    dropdown_label: str = "Stratify by:",
) -> None:
    """Export descriptive-statistics tables as a standalone HTML page.

    Reproduces the "Section 2: Descriptive Statistics" layout from the
    project's ``export_html_v1.py`` — a dark-themed page with a dropdown
    selector that switches between stratified tables rendered in an iframe.

    Sections 1 (Sankey / figures) and 3 (missing-data report) are **not**
    included; use :func:`save_figure` or :func:`save_table` for those.

    Parameters
    ----------
    tables : dict[str, Styler | str]
        Mapping of stratification key → styled table (a ``pandas Styler``
        object or a pre-rendered HTML string).  The first key is shown by
        default.  Example keys: ``"tx-type"``, ``"biopsy-result"``.
    output_path : str | Path
        Destination HTML file (e.g. ``"html_data/descriptive_stats.html"``).
        Parent directories are created automatically.
    labels : dict[str, str], optional
        Human-readable labels shown in the dropdown for each key.  Defaults
        to :data:`_DEFAULT_STRAT_LABELS` for recognised keys; unrecognised
        keys fall back to the key string itself.
    title : str
        Page heading (default ``"Descriptive Statistics"``).
    subtitle : str
        Grey description text rendered above the dropdown.
    dropdown_label : str
        Label for the ``<select>`` element (default ``"Stratify by:"``).
    """
    _ensure_dir(output_path)

    # Resolve labels: use caller-supplied, then defaults, then raw keys
    if labels is None:
        labels = {
            k: _DEFAULT_STRAT_LABELS.get(k, k) for k in tables
        }

    # Convert Styler objects → HTML strings
    table_html_map: dict[str, str] = {}
    for key, styled in tables.items():
        table_html_map[key] = (
            styled if isinstance(styled, str) else styled.to_html()
        )

    tables_json = json.dumps(table_html_map)

    dropdown_options_html = "\n".join(
        f'      <option value="{key}">{lbl}</option>'
        for key, lbl in labels.items()
    )

    subtitle_html = (
        f'<p class="desc-text">{subtitle}</p>' if subtitle else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h2>{title}</h2>
  {subtitle_html}

  <label class="strat-label" for="strat-select">{dropdown_label}</label>
  <select id="strat-select">
{dropdown_options_html}
  </select>

  <div class="table-frame">
    <iframe id="table-iframe"></iframe>
  </div>

  <script>
    var tables = {tables_json};

    function updateTable() {{
      var key = document.getElementById('strat-select').value;
      var iframe = document.getElementById('table-iframe');
      var html = tables[key] || '<p>No data</p>';
      var doc = iframe.contentDocument || iframe.contentWindow.document;
      doc.open();
      doc.write(html);
      doc.close();
    }}

    document.getElementById('strat-select').addEventListener('change', updateTable);
    updateTable();
  </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Saved: {output_path}")
