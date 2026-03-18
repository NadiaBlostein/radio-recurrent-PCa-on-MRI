"""
pca_mri.visualization — publication-ready figures for the PCa MRI project.

Modules
-------
distributions       Age, PSA, Gleason, T-stage, CAPRA distributions.
timelines           Time-to-event histograms and MRI follow-up counts.
kinetics            PSA doubling-time and per-patient PSA trajectory plots.
descriptive_plots   Standalone KDE density and stacked bar chart figures.
interactive         Dropdown-driven KDE + stacked bar chart explorer.
dashboard           Interactive Plotly dashboard for full analysis results.
sankey              Patient exclusion flow (Sankey diagram).
export_html         Export figures and tables as standalone HTML files.
"""
from pca_mri.visualization import (
    distributions, timelines, kinetics, descriptive_plots,
    interactive, dashboard, sankey, export_html,
)

__all__ = [
    "distributions", "timelines", "kinetics", "descriptive_plots",
    "interactive", "dashboard", "sankey", "export_html",
]
