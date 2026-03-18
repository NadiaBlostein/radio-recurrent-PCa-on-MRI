"""
pca_mri.visualization.sankey — Patient exclusion flow (Sankey diagram).

Traces how datapoints were systematically excluded through the data-cleanup
pipeline, from raw dataset (N=110) to the final analytic cohort (N=68).
"""

from __future__ import annotations

import plotly.graph_objects as go


def plot_exclusion_sankey() -> go.Figure:
    """Build a Plotly Sankey diagram of the data-cleanup exclusion flow.

    The numbers are derived from the cleanup pipeline in data-cleanup.ipynb:
      - Raw dataset: 110 patients
      - 3 duplicate patients removed → 107
      - 39 patients without biopsy data excluded → 68
      - Of 68: 48 biopsy-positive, 20 biopsy-negative
      - Of 68: 52 MRI-positive, 16 MRI-negative (cross-tabulated with biopsy)

    Nodes at each stage are aligned in the same vertical column so that
    kept and excluded groups appear side-by-side.

    Returns
    -------
    go.Figure
        Plotly Figure with the Sankey diagram.
    """

    # ── Node definitions with explicit (x, y) positions ──
    # Column 0 (x=0.01): Raw dataset
    # Column 1 (x=0.25): After dedup (kept) + Duplicates (excluded)
    # Column 2 (x=0.50): Analytic cohort (kept) + No biopsy (excluded)
    # Column 3 (x=0.75): Biopsy+, Biopsy-
    # Column 4 (x=0.99): MRI+, MRI-

    labels = [
        "Raw dataset<br>(N=110)",                          # 0
        "After deduplication<br>(N=107)",                   # 1
        "Duplicates removed<br>(N=3)",                     # 2
        "Analytic cohort<br>(N=68)",                        # 3
        "No biopsy data<br>excluded (N=39)",                # 4
        "Biopsy-positive<br>(N=48)",                        # 5
        "Biopsy-negative<br>(N=20)",                        # 6
        "MRI-positive<br>(N=52)",                           # 7
        "MRI-negative<br>(N=16)",                           # 8
    ]

    node_colors = [
        "#608da2",   # 0 raw — blue
        "#608da2",   # 1 after dedup — same blue
        "#888888",   # 2 duplicates — gray (excluded)
        "#3A86FF",   # 3 analytic cohort — different blue
        "#888888",   # 4 no biopsy — gray (excluded)
        "#00CC96",   # 5 biopsy+ — green
        "#EF553B",   # 6 biopsy- — red
        "#00CC96",   # 7 MRI+ — green
        "#EF553B",   # 8 MRI- — red
    ]

    # x positions (columns); y positions (vertical order within column)
    # Plotly requires values in (0,1) exclusive — use small offsets from edges
    x_pos = [0.01,  0.25, 0.25,  0.50, 0.50,  0.75, 0.75,  0.99, 0.99]
    y_pos = [0.50,  0.35, 0.95,  0.35, 0.90,  0.25, 0.75,  0.30, 0.85]

    # ── Link definitions (source → target, value) ──
    # Cross-tab: Bx+ → MRI+ 36, Bx+ → MRI− 12, Bx− → MRI+ 16, Bx− → MRI− 4
    sources =    [0,  0,  1,  1,  3,  3,  5,  5,  6,  6]
    targets =    [1,  2,  3,  4,  5,  6,  7,  8,  7,  8]
    values  =    [107, 3, 68, 39, 48, 20, 36, 12, 16,  4]

    link_colors = [
        "rgba(96,141,162,0.4)",   # 0→1 kept (blue)
        "rgba(136,136,136,0.4)",  # 0→2 duplicates (gray)
        "rgba(58,134,255,0.4)",   # 1→3 analytic cohort (blue)
        "rgba(136,136,136,0.4)",  # 1→4 no biopsy (gray)
        "rgba(0,204,150,0.4)",    # 3→5 biopsy+ (green)
        "rgba(239,85,59,0.4)",    # 3→6 biopsy- (red)
        "rgba(0,204,150,0.4)",    # 5→7 biopsy+ → MRI+ (green)
        "rgba(239,85,59,0.4)",    # 5→8 biopsy+ → MRI- (red)
        "rgba(0,204,150,0.4)",    # 6→7 biopsy- → MRI+ (green)
        "rgba(239,85,59,0.4)",    # 6→8 biopsy- → MRI- (red)
    ]

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="#555", width=1),
            label=labels,
            color=node_colors,
            x=x_pos,
            y=y_pos,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Flow Diagram of Patient Selection and Cohort Derivation",
            font=dict(size=16, color="white"),
        ),
        font=dict(size=12, color="white"),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        height=600,
        width=1050,
        annotations=[
            dict(
                text=(
                    "<b>Key:</b> "
                    "<span style='color:#888888'>\u25a0</span> Excluded  "
                    "<span style='color:#608da2'>\u25a0</span> Raw  "
                    "<span style='color:#3A86FF'>\u25a0</span> Analytic cohort  "
                    "<span style='color:#00CC96'>\u25a0</span> Positive  "
                    "<span style='color:#EF553B'>\u25a0</span> Negative"
                ),
                showarrow=False,
                xref="paper", yref="paper",
                x=1.0, y=1.0,
                xanchor="right", yanchor="top",
                font=dict(size=11),
            ),
        ],
    )

    return fig
