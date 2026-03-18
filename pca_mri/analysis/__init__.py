"""
pca_mri.analysis — descriptive statistics, stratification, diagnostics, regression.

Modules
-------
descriptive      table1(), missingness_summary(), capra_summary()
stratification   split_by_tx_type(), split_by_recurrence()
diagnostic       prevalence(), contingency_table(), diagnostic_accuracy()
regression       univariate_screen(), build_multivariable_model(), bootstrap_auc()
"""
from pca_mri.analysis import descriptive, stratification, diagnostic, regression

__all__ = ["descriptive", "stratification", "diagnostic", "regression"]
