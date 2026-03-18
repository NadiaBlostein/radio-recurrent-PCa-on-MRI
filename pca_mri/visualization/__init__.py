"""
pca_mri.visualization — publication-ready figures for the PCa MRI project.

Modules
-------
distributions    Age, PSA, Gleason, T-stage, CAPRA distributions.
timelines        Time-to-event histograms and MRI follow-up counts.
kinetics         PSA doubling-time and per-patient PSA trajectory plots.
interactive      Dropdown-driven KDE + stacked bar chart explorer.
dashboard        Interactive Plotly dashboard for full analysis results.
sankey           Patient exclusion flow (Sankey diagram).
"""
from pca_mri.visualization import distributions, timelines, kinetics, interactive, dashboard, sankey

__all__ = ["distributions", "timelines", "kinetics", "interactive", "dashboard", "sankey"]
