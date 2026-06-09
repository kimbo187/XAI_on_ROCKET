"""
XAI on ROCKET — SHAP Companion Script
=====================================

This file documents the SHAP integration point used in the thesis work.
The main runnable baseline in this repository is:

    notebooks/XAI_on_ROCKET_without_SHAP.py

To run this companion script, install the optional SHAP dependencies first:

    pip install -r requirements-shap.txt

The SHAP workflow can be computationally heavier than the intrinsic baseline,
especially when many ROCKET kernels are used.
"""

from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP is optional and is not installed by default. "
        "Install it with: pip install -r requirements-shap.txt"
    ) from exc

from src.xai_methods import map_kernels_to_input_with_shap


def compute_kernel_shap_values(model, background_data: np.ndarray, samples: np.ndarray):
    """Compute SHAP values for a trained classifier.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible model, for example RidgeClassifierCV.
    background_data:
        Background transformed ROCKET features used by SHAP.
    samples:
        Transformed samples to explain.

    Returns
    -------
    shap_values:
        SHAP values for the provided samples.

    Notes
    -----
    For ROCKET, SHAP is normally applied to the transformed feature space.
    The helper `map_kernels_to_input_with_shap` can then be used to connect
    SHAP-weighted kernel windows back to the raw time-series signal.
    """
    explainer = shap.KernelExplainer(model.decision_function, background_data)
    return explainer.shap_values(samples)


def map_shap_weighted_kernels(kernel_parameters, input_ts, shap_values):
    """Map selected kernels back to raw time steps with SHAP weighting."""
    return map_kernels_to_input_with_shap(kernel_parameters, input_ts, shap_values)


if __name__ == "__main__":
    print("This is a companion SHAP integration script.")
    print("Run notebooks/XAI_on_ROCKET_without_SHAP.py first to train ROCKET and select kernels.")
    print("Then use compute_kernel_shap_values(...) and map_shap_weighted_kernels(...).")
