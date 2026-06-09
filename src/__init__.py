"""
xai_rocket
==========
Explainability toolkit for the ROCKET time-series classifier.
"""

from .rocket_core  import generate_kernels, apply_kernel, apply_kernels
from .xai_methods  import (
    dilate_kernel,
    feature_map_xai,
    select_kernels_with_mutual_info,
    get_kernel_parameters,
    map_kernels_to_input,
    map_kernels_to_input_with_shap,
)
from .pipeline import (
    load_dataset,
    encode_labels,
    build_rocket_features,
    train_ridge_classifier,
    evaluate_classifier,
    calculate_feature_impacts,
    select_and_segment_features,
    calculate_contributions,
    calculate_faithfulness,
    calculate_robustness,
)
from .visualization import (
    plot_kernel_impact,
    plot_shap_kernel_comparison,
    plot_results_table,
)

__all__ = [
    "generate_kernels", "apply_kernel", "apply_kernels",
    "dilate_kernel", "feature_map_xai",
    "select_kernels_with_mutual_info", "get_kernel_parameters",
    "map_kernels_to_input", "map_kernels_to_input_with_shap",
    "load_dataset", "encode_labels", "build_rocket_features",
    "train_ridge_classifier", "evaluate_classifier",
    "calculate_feature_impacts", "select_and_segment_features",
    "calculate_contributions", "calculate_faithfulness", "calculate_robustness",
    "plot_kernel_impact", "plot_shap_kernel_comparison", "plot_results_table",
]
