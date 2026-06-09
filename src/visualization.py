"""
visualization.py
================
Saliency-map and kernel-impact plots for the ROCKET XAI pipeline.

Reference:
    Meng, H., Wagner, C., & Triguero, I. (2023). Explaining time series
    classifiers through meaningful perturbation and optimisation.
    Information Sciences, 645, 119334.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional


def plot_kernel_impact(
    input_ts:       np.ndarray,
    mapped_kernels: list[list[tuple]],
    title:          str  = "Kernel Impact on Time Series",
    figsize:        tuple = (15, 5),
    threshold:      Optional[float] = None,
    save_path:      Optional[str]   = None,
) -> plt.Figure:
    """
    Saliency-map: overlay kernel receptive-field highlights on the raw time series.

    Each kernel is assigned a distinct colour (viridis palette).
    Receptive-field windows above ``threshold`` are shaded; individual
    contributing time steps are marked with scatter dots.

    Parameters
    ----------
    input_ts       : shape (ts_length,) – raw time series values
    mapped_kernels : output of map_kernels_to_input or map_kernels_to_input_with_shap
    title          : plot title
    figsize        : matplotlib figure size
    threshold      : if given, only highlight windows where |kernel_value| > threshold
    save_path      : optional file path to save the figure

    Returns
    -------
    fig : matplotlib Figure
    """
    ts = input_ts[0] if input_ts.ndim == 2 else input_ts
    time_steps  = np.arange(len(ts))
    num_kernels = len(mapped_kernels)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_steps, ts, color="black", linewidth=1.2, label="Input Time Series", zorder=5)

    cmap    = plt.get_cmap("viridis")
    handles = []

    for i, mappings in enumerate(mapped_kernels):
        colour = cmap(i / max(num_kernels - 1, 1))
        first  = True
        for entry in mappings:
            t, val = entry[0], entry[1]
            if threshold is not None and abs(val) <= threshold:
                continue
            ax.axvspan(t, t + 1, color=colour, alpha=0.25, zorder=2)
            lbl = f"Kernel {i + 1} Contribution" if first else None
            sc  = ax.scatter(t, ts[t], color=colour, edgecolor="black",
                             s=30, zorder=6, label=lbl)
            if first:
                handles.append(sc)
                first = False

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_shap_kernel_comparison(
    input_ts:               np.ndarray,
    mapped_kernels_no_shap: list[list[tuple]],
    mapped_kernels_shap:    list[list[tuple]],
    title_suffix:           str  = "",
    figsize:                tuple = (18, 5),
    save_path:              Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison: kernels WITH SHAP (left) vs WITHOUT SHAP (right).

    Parameters
    ----------
    input_ts               : raw time series, shape (ts_length,) or (1, ts_length)
    mapped_kernels_shap    : from map_kernels_to_input_with_shap
    mapped_kernels_no_shap : from map_kernels_to_input
    title_suffix           : appended to each subplot title
    save_path              : optional save path
    """
    ts = input_ts[0] if input_ts.ndim == 2 else input_ts

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, mappings, label in zip(
        axes,
        [mapped_kernels_shap, mapped_kernels_no_shap],
        ["With SHAP", "Without SHAP"],
    ):
        ax.plot(np.arange(len(ts)), ts, color="black", linewidth=1.2,
                label="Input Time Series", zorder=5)
        cmap = plt.get_cmap("viridis")
        n    = max(len(mappings) - 1, 1)

        for i, kernel_maps in enumerate(mappings):
            colour = cmap(i / n)
            # SHAP-enhanced mappings have 3 elements; plain have 2
            has_shap = len(kernel_maps[0]) == 3 if kernel_maps else False
            for entry in kernel_maps:
                t, val = entry[0], entry[1]
                shap_w = abs(entry[2]) if has_shap else 1.0
                alpha  = min(0.6, 0.1 + shap_w * 0.5) if has_shap else 0.25
                ax.axvspan(t, t + 1, color=colour, alpha=alpha, zorder=2)

        ax.set_title(f"{label} – {title_suffix}")
        ax.set_xlabel("Time Steps")

    axes[0].set_ylabel("Value")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_results_table(
    results: dict[str, dict[str, float]],
    title:   str = "Faithfulness & Robustness Summary",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a summary table of Faithfulness and Robustness scores.

    Parameters
    ----------
    results : nested dict  {group_name: {"faithfulness": ..., "faithfulness_moe": ...,
                                         "robustness": ..., "robustness_moe": ...}}
    """
    groups   = list(results.keys())
    col_lbls = ["Faithfulness (mean ± err)", "Robustness (mean ± err)"]

    cell_text = []
    for g in groups:
        r = results[g]
        cell_text.append([
            f"{r['faithfulness']:.4f} ± {r['faithfulness_moe']:.4f}",
            f"{r['robustness']:.4f} ± {r['robustness_moe']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.5 * len(groups)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=groups,
        colLabels=col_lbls,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax.set_title(title, fontsize=12, pad=20)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
