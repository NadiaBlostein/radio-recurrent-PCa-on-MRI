"""
pca_mri — analysis library for radio-recurrent prostate cancer on MRI.

Modules:
    io              — load/save raw and cleaned data files
    preprocessing   — column cleaning, patient-level flags, derived features
    analysis        — descriptive statistics, stratification
    visualization   — timelines, distributions, PSA kinetics
"""
from pca_mri import io
from pca_mri import preprocessing
from pca_mri import analysis
from pca_mri import visualization

__all__ = ["io", "preprocessing", "analysis", "visualization"]
