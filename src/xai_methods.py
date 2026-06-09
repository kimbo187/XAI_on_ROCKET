"""
xai_methods.py
==============
Intrinsic XAI utilities for the ROCKET time-series classifier.

Provides:
  - Feature-map construction (padded / unpadded)
  - PPV and MV contribution attribution per time step
  - Mutual-information-based kernel importance ranking
  - Kernel parameter extraction
  - Kernel-to-input mapping (with and without SHAP)

References
----------
Rastegar, A. (2023). Explaining ROCKET time series classifier model.
Meng, H., Wagner, C., & Triguero, I. (2023). Explaining time series
    classifiers through meaningful perturbation and optimisation.
    Information Sciences, 645, 119334.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_regression


# ──────────────────────────────────────────────────────────────────────────────
# Kernel dilation helper
# ──────────────────────────────────────────────────────────────────────────────

def dilate_kernel(weights: np.ndarray, dilation: int) -> np.ndarray:
    """
    Expand kernel weights by inserting zeros according to the dilation factor.

    Parameters
    ----------
    weights : np.ndarray, shape (kernel_length,)
    dilation : int

    Returns
    -------
    kernel_dilated : np.ndarray, shape ((kernel_length - 1) * dilation + 1,)
    """
    length = len(weights)
    receptive_field = (length - 1) * dilation + 1
    kernel_dilated = np.zeros(receptive_field)
    for j in range(length):
        kernel_dilated[j * dilation] = weights[j]
    return kernel_dilated


# ──────────────────────────────────────────────────────────────────────────────
# Feature-map helpers
# ──────────────────────────────────────────────────────────────────────────────

def _feature_map_kernel_pad(X, weights, length, bias, dilation, padding):
    """Feature map for a kernel *with* padding (symmetric)."""
    input_length    = len(X)
    receptive_field = (length - 1) * dilation + 1
    half_pad        = (receptive_field - 1) // 2
    X_padded        = np.concatenate([np.zeros(half_pad), X, np.zeros(half_pad)])

    kernel_dilated      = dilate_kernel(weights, dilation)
    fm                  = np.zeros(input_length)
    indices_contributors = []
    values_contributors  = []

    for t in range(input_length):
        fm[t] = bias + np.inner(kernel_dilated, X_padded[t : t + receptive_field])
        contributors = t + np.nonzero(kernel_dilated)[0] - half_pad
        indices_contributors.append(contributors)
        values_contributors.append(fm[t])

    return fm, indices_contributors, values_contributors


def _feature_map_kernel_nopad(X, weights, length, bias, dilation, padding):
    """Feature map for a kernel *without* padding."""
    input_length    = len(X)
    receptive_field = (length - 1) * dilation + 1
    kernel_dilated  = dilate_kernel(weights, dilation)

    fm                  = np.zeros(input_length)
    indices_contributors = []
    values_contributors  = []

    for t in range(input_length - receptive_field + 1):
        fm[t] = bias + np.inner(kernel_dilated, X[t : t + receptive_field])
        contributors = t + np.nonzero(kernel_dilated)[0]
        indices_contributors.append(contributors)
        values_contributors.append(fm[t])

    return fm, indices_contributors, values_contributors


# ──────────────────────────────────────────────────────────────────────────────
# Contribution attribution
# ──────────────────────────────────────────────────────────────────────────────

def calc_features_contribution(
    indices_contributors: list,
    values_contributors: list,
    length_ts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-time-step PPV and MV contribution scores for a single kernel.

    Parameters
    ----------
    indices_contributors : list of arrays
        For each convolution position, the input indices it "sees".
    values_contributors : list of floats
        Convolution output at each position.
    length_ts : int
        Length of the time series.

    Returns
    -------
    contributions_ppv_feat : np.ndarray, shape (length_ts,)
        Normalised PPV attribution per time step.
    contributions_mv_feat : np.ndarray, shape (length_ts,)
        Normalised MV attribution per time step.
    """
    contributions = np.full((length_ts, length_ts), np.nan)

    for i, (indices, value) in enumerate(zip(indices_contributors, values_contributors)):
        for j in indices:
            if 0 <= j < length_ts:
                contributions[i, j] = value

    # PPV: binary – did this time step contribute to a positive activation?
    contributions_ppv = np.where(contributions > 0, 1.0, 0.0)
    ppv_feat = np.any(contributions_ppv > 0, axis=0).astype(float)
    if ppv_feat.sum() > 0:
        ppv_feat /= ppv_feat.sum()

    # MV: which time step contributed to the maximum activation?
    mv_feat = np.nanmax(contributions, axis=0)
    mv_feat = np.where(mv_feat == np.nanmax(mv_feat), 1.0, 0.0)
    if mv_feat.sum() > 0:
        mv_feat /= mv_feat.sum()

    return ppv_feat, mv_feat


def feature_map_xai(
    X: np.ndarray,
    kernels: tuple,
    kernel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute feature maps and contribution scores for a set of kernel indices.

    Parameters
    ----------
    X : np.ndarray, shape (time_series_length,)
        A single time series instance.
    kernels : tuple
        Output of ``generate_kernels``.
    kernel_indices : np.ndarray of int
        Indices of the kernels to visualise.

    Returns
    -------
    fm_array : np.ndarray, shape (n_kernels, ts_length)
    contributions_ppv : np.ndarray, shape (n_kernels, ts_length)
    contributions_mv  : np.ndarray, shape (n_kernels, ts_length)
    """
    weights, lengths, biases, dilations, paddings = kernels

    fm_list, ppv_list, mv_list = [], [], []

    for j in kernel_indices:
        a1  = int(np.sum(lengths[:j]))
        b1  = a1 + int(lengths[j])
        w   = weights[a1:b1]

        if paddings[j] == 0:
            fm, idx_contrib, val_contrib = _feature_map_kernel_nopad(
                X, w, lengths[j], biases[j], dilations[j], paddings[j]
            )
        else:
            fm, idx_contrib, val_contrib = _feature_map_kernel_pad(
                X, w, lengths[j], biases[j], dilations[j], paddings[j]
            )

        ppv_feat, mv_feat = calc_features_contribution(idx_contrib, val_contrib, len(X))

        fm_list.append(fm)
        ppv_list.append(ppv_feat)
        mv_list.append(mv_feat)

    return (
        np.array(fm_list),
        np.vstack(ppv_list),
        np.vstack(mv_list),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Kernel selection via mutual information
# ──────────────────────────────────────────────────────────────────────────────

def select_kernels_with_mutual_info(
    contributions_ppv: np.ndarray,
    contributions_mv:  np.ndarray,
    num_kernels: int,
    y: np.ndarray,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank kernels by importance = sum(contributions) × mutual_information(contributions, y).

    Parameters
    ----------
    contributions_ppv : np.ndarray, shape (n_kernels, ts_length)
    contributions_mv  : np.ndarray, shape (n_kernels, ts_length)
    num_kernels : int
        Total number of kernels (used with threshold to set the top-k count).
    y : np.ndarray
        Target labels (encoded as integers).
    threshold : float
        Fraction of top kernels to return.

    Returns
    -------
    important_ppv : np.ndarray of int – indices of top PPV kernels
    important_mv  : np.ndarray of int – indices of top MV kernels
    """
    k = max(1, int(num_kernels * threshold))

    mi_ppv = np.array([
        mutual_info_regression(contributions_ppv[i].reshape(-1, 1), y)[0]
        for i in range(contributions_ppv.shape[0])
    ])
    mi_mv = np.array([
        mutual_info_regression(contributions_mv[i].reshape(-1, 1), y)[0]
        for i in range(contributions_mv.shape[0])
    ])

    score_ppv = contributions_ppv.sum(axis=1) * mi_ppv
    score_mv  = contributions_mv.sum(axis=1)  * mi_mv

    return (
        np.argsort(score_ppv)[::-1][:k],
        np.argsort(score_mv)[::-1][:k],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Kernel parameter extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_kernel_parameters(
    important_kernel_indices: np.ndarray,
    kernels: tuple,
) -> list[tuple]:
    """
    Extract full parameter tuples for the selected important kernels.

    Parameters
    ----------
    important_kernel_indices : array of int
    kernels : tuple from generate_kernels

    Returns
    -------
    list of (length, weights, bias, dilation, padding) tuples
    """
    weights, lengths, biases, dilations, paddings = kernels
    params = []

    # The weight vector stores all kernel weights concatenated together.
    # Therefore the start position for kernel idx is the cumulative sum of
    # all previous kernel lengths, not the position within the selected list.
    cumulative_starts = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(int)

    for idx in np.asarray(important_kernel_indices, dtype=int):
        if idx < 0 or idx >= len(lengths):
            continue
        start = int(cumulative_starts[idx])
        klen = int(lengths[idx])
        end = start + klen
        params.append((
            klen,
            weights[start:end].copy(),
            float(biases[idx]),
            int(dilations[idx]),
            int(paddings[idx]),
        ))
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Kernel → input mapping
# ──────────────────────────────────────────────────────────────────────────────

def map_kernels_to_input(
    kernel_parameters: list[tuple],
    input_ts: np.ndarray,
) -> list[list[tuple]]:
    """
    Map important kernels back to their receptive fields in the raw time series.

    Parameters
    ----------
    kernel_parameters : list of (length, weights, bias, dilation, padding)
    input_ts : np.ndarray, shape (ts_length,)

    Returns
    -------
    mapped_kernels : list of lists of (time_step, kernel_value)
    """
    input_length   = len(input_ts)
    mapped_kernels = []

    for klen, kweights, kbias, kdilation, _ in kernel_parameters:
        receptive_field = (klen - 1) * kdilation + 1
        kernel_dilated  = dilate_kernel(kweights, kdilation)
        mappings        = []

        for t in range(input_length - receptive_field + 1):
            val = kbias + np.inner(kernel_dilated, input_ts[t : t + receptive_field])
            mappings.append((t, val))

        mapped_kernels.append(mappings)

    return mapped_kernels


def map_kernels_to_input_with_shap(
    kernel_parameters: list[tuple],
    input_ts: np.ndarray,
    shap_values: np.ndarray,
) -> list[list[tuple]]:
    """
    Identical to ``map_kernels_to_input`` but also computes the cumulative SHAP
    contribution for each kernel's receptive-field window.

    Parameters
    ----------
    kernel_parameters : list of (length, weights, bias, dilation, padding)
    input_ts : np.ndarray, shape (1, ts_length)  *or*  (ts_length,)
    shap_values : np.ndarray, shape (ts_length,)

    Returns
    -------
    mapped_kernels : list of lists of (time_step, kernel_value, shap_contribution)
    """
    # Accept both (1, L) and (L,) input shapes
    ts = input_ts[0] if input_ts.ndim == 2 else input_ts
    input_length   = len(ts)
    mapped_kernels = []

    for klen, kweights, kbias, kdilation, _ in kernel_parameters:
        receptive_field = (klen - 1) * kdilation + 1
        kernel_dilated  = dilate_kernel(kweights, kdilation)
        mappings        = []

        for t in range(input_length - receptive_field + 1):
            val  = kbias + np.inner(kernel_dilated, ts[t : t + receptive_field])
            shap = float(np.sum(shap_values[t : t + receptive_field]))
            mappings.append((t, val, shap))

        mapped_kernels.append(mappings)

    return mapped_kernels
