"""
pipeline.py
===========
High-level helpers: data loading, feature-impact segmentation,
classifier training/evaluation, and metric computation.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from scipy.stats import bootstrap

from sktime.datasets import load_UCR_UEA_dataset

from .rocket_core import generate_kernels, apply_kernels
from .xai_methods import (
    feature_map_xai,
    select_kernels_with_mutual_info,
    get_kernel_parameters,
    map_kernels_to_input,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a UCR/UEA time-series dataset via sktime.

    Parameters
    ----------
    name : str – e.g. ``"GunPoint"``

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, y_train = load_UCR_UEA_dataset(name=name, split="train", return_type="numpy2d")
    X_test,  y_test  = load_UCR_UEA_dataset(name=name, split="test",  return_type="numpy2d")
    return X_train, X_test, y_train, y_test


def encode_labels(
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Label-encode class strings to integers."""
    le = LabelEncoder()
    return le.fit_transform(y_train), le.transform(y_test), le


# ──────────────────────────────────────────────────────────────────────────────
# ROCKET pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_rocket_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    num_kernels: int = 300,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Generate kernels, transform train and test sets.

    Returns
    -------
    X_train_t, X_test_t : transformed feature matrices
    kernels              : raw kernel tuple (for later XAI use)
    """
    kernels   = generate_kernels(X_train.shape[1], num_kernels)
    X_train_t = apply_kernels(X_train, kernels)
    X_test_t  = apply_kernels(X_test,  kernels)
    return X_train_t, X_test_t, kernels


def train_ridge_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RidgeClassifierCV:
    """Train a RidgeClassifierCV (one-vs-rest, L2 regularisation)."""
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), fit_intercept=True)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf:    RidgeClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> float:
    """Return accuracy and optionally print it."""
    acc = clf.score(X_test, y_test)
    if verbose:
        print(f"Accuracy: {acc:.4f}")
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Feature-impact segmentation (intrinsic XAI)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_feature_impacts(
    coefs:           np.ndarray,
    X_test_transform: np.ndarray,
) -> np.ndarray:
    """
    Element-wise product of classifier coefficients and transformed features.

    Returns
    -------
    feature_impacts : np.ndarray, shape (n_examples, n_features)
    """
    return coefs * X_test_transform


def select_and_segment_features(
    feature_impacts: np.ndarray,
    n_top_coef:  int,
    kernel_size: int,
    ts_idx:      int,
) -> dict[str, np.ndarray]:
    """
    Split the most influential features into four groups:

    - ``below_pos``  : positively contributing features with index < kernel_size
    - ``above_pos``  : positively contributing features with index >= kernel_size
    - ``below_neg``  : negatively contributing features with index < kernel_size
    - ``above_neg``  : negatively contributing features with index >= kernel_size

    All index arrays are halved (integer) to convert from feature-space to
    time-series-space indices (PPV and MV are interleaved in ROCKET output).

    Returns
    -------
    dict with keys ``below_pos``, ``above_pos``, ``below_neg``, ``above_neg``
    (np.ndarray of int)
    """
    impacts = feature_impacts[ts_idx]

    top_pos = np.argsort(impacts)[::-1][:n_top_coef]
    top_neg = np.argsort(impacts)[:n_top_coef]

    def split_and_halve(indices):
        below = indices[indices <  kernel_size]
        above = indices[indices >= kernel_size]
        return (below // 2).astype(int), (above // 2).astype(int)

    below_pos, above_pos = split_and_halve(top_pos)
    below_neg, above_neg = split_and_halve(top_neg)

    return {
        "below_pos": below_pos,
        "above_pos": above_pos,
        "below_neg": below_neg,
        "above_neg": above_neg,
    }


def calculate_contributions(
    coefs:            np.ndarray,
    feature_indices:  np.ndarray,
    ts_idx:           int,
    X_test_transform: np.ndarray,
) -> np.ndarray:
    """Signed contributions of selected features for a single instance."""
    valid = feature_indices[
        (feature_indices >= 0) & (feature_indices < len(coefs)) &
        (feature_indices < X_test_transform.shape[1])
    ]
    return np.abs(coefs[valid]) * X_test_transform[ts_idx, valid]


# ──────────────────────────────────────────────────────────────────────────────
# XAI evaluation metrics: Faithfulness and Robustness
# ──────────────────────────────────────────────────────────────────────────────

def _add_noise_segment(
    x:           np.ndarray,
    start:       int,
    end:         int,
    noise_level: float = 0.1,
) -> np.ndarray:
    x_mod = x.copy()
    x_mod[start:end] += np.random.normal(0, noise_level, end - start)
    return x_mod


def calculate_faithfulness(
    model:             RidgeClassifierCV,
    X_test_transformed: np.ndarray,
    important_segments: list[tuple[int, int]],
    noise_level:       float = 0.1,
) -> tuple[float, float]:
    """
    Faithfulness: average drop in model confidence when the identified
    important segment is perturbed with Gaussian noise.

    Higher (more positive) values indicate that the explanation correctly
    identifies influential segments.

    Returns
    -------
    mean  : float
    margin_of_error : float (95 % bootstrap CI half-width)
    """
    drops = []
    for i, x in enumerate(X_test_transformed):
        if i >= len(important_segments):
            break
        start, end = important_segments[i]
        if not (0 <= start < end <= len(x)):
            continue
        x_mod   = _add_noise_segment(x, start, end, noise_level)
        drop    = (model.decision_function([x])[0] -
                   model.decision_function([x_mod])[0])
        drops.append(drop)

    if not drops:
        return float("nan"), float("nan")

    mean = float(np.mean(drops))
    if len(drops) > 1:
        res = bootstrap((np.array(drops),), np.mean,
                        confidence_level=0.95, n_resamples=1000)
        moe = (res.confidence_interval.high - res.confidence_interval.low) / 2
    else:
        moe = float("nan")
    return mean, moe


def calculate_robustness(
    kernel_parameters: list[tuple],
    input_ts:          np.ndarray,
    num_perturbations: int   = 10,
    noise_level:       float = 0.1,
    threshold:         float = 0.1,
) -> dict[str, float]:
    """
    Robustness: how stable are the kernel mappings under small input noise?

    Parameters
    ----------
    kernel_parameters : from get_kernel_parameters
    input_ts          : raw time series, shape (ts_length,)
    num_perturbations : number of noise trials
    noise_level       : std-dev of Gaussian noise
    threshold         : max absolute change below which explanations are 'unchanged'

    Returns
    -------
    dict with keys ``mean``, ``margin_of_error``,
    ``unchanged_proportion_mean``, ``unchanged_proportion_moe``
    """
    base_mappings  = map_kernels_to_input(kernel_parameters, input_ts)
    robustness_vals, unchanged_props = [], []

    for _ in range(num_perturbations):
        noisy_ts   = input_ts + np.random.normal(0, noise_level, input_ts.shape)
        pert_maps  = map_kernels_to_input(kernel_parameters, noisy_ts)

        diffs = [
            np.mean(np.abs(
                np.array([v for _, v in base]) -
                np.array([v for _, v in pert])
            ))
            for base, pert in zip(base_mappings, pert_maps)
            if base and pert
        ]
        diffs = [d for d in diffs if not np.isnan(d)]
        if diffs:
            robustness_vals.append(float(np.mean(diffs)))

        unch = sum(
            1 for base, pert in zip(base_mappings, pert_maps)
            if base and pert and np.all(
                np.abs(
                    np.array([v for _, v in base]) -
                    np.array([v for _, v in pert])
                ) < threshold
            )
        )
        unchanged_props.append(unch / max(len(base_mappings), 1))

    def _bootstrap_moe(arr):
        if len(arr) < 2:
            return float("nan")
        res = bootstrap((np.array(arr),), np.mean,
                        confidence_level=0.95, n_resamples=1000)
        return (res.confidence_interval.high - res.confidence_interval.low) / 2

    return {
        "mean":                     float(np.mean(robustness_vals)),
        "margin_of_error":          _bootstrap_moe(robustness_vals),
        "unchanged_proportion_mean": float(np.mean(unchanged_props)),
        "unchanged_proportion_moe": _bootstrap_moe(unchanged_props),
    }
