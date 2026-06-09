"""Counterfactual Recommendation via Doubly Calibrated Estimator (DCE).

Skeleton based on arXiv:2403.00817 (WWW 2024 oral).
Official PyTorch implementation: https://github.com/WonbinKweon/DCE_WWW2024
"""

from .model import CalibratedPropensityModel, dce_dr_loss

__all__ = ["CalibratedPropensityModel", "dce_dr_loss"]
