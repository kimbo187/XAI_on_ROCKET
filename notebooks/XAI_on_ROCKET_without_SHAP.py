"""
XAI on ROCKET — Without SHAP (Intrinsic XAI)
=============================================
Master's Thesis · Stockholm University, Spring 2024
Authors : Kamal Tikabo & Pamodou Touray
Supervisor : Zhendong Wang

This notebook reproduces the "without-SHAP" baseline described in the thesis.
It runs the full ROCKET pipeline on the GunPoint dataset, applies intrinsic
XAI attribution, and evaluates explanations using Faithfulness and Robustness.

Run in order:
    1. Install dependencies  (Cell 0)
    2. Imports & seeds        (Cell 1)
    3. Data loading           (Cell 2)
    4. ROCKET training        (Cell 3)
    5. Feature segmentation   (Cell 4)
    6. Feature maps & XAI     (Cell 5)
    7. Kernel importance      (Cell 6)
    8. Visualisation          (Cell 7)
    9. Faithfulness           (Cell 8)
    10. Robustness            (Cell 9)
"""

# ============================================================
# Cell 0 – Install dependencies
# ============================================================
# Run once in a fresh environment:
#   pip install sktime numba numpy scipy scikit-learn matplotlib

# ============================================================
# Cell 1 – Imports & reproducibility seeds
# ============================================================

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from scipy.stats import bootstrap

from src.rocket_core import generate_kernels, apply_kernels
from src.xai_methods import (
    feature_map_xai,
    select_kernels_with_mutual_info,
    get_kernel_parameters,
    map_kernels_to_input,
)
from src.pipeline import (
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
from src.visualization import plot_kernel_impact, plot_results_table

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Environment ready.")

# ============================================================
# Cell 2 – Load GunPoint dataset
# ============================================================

X_train, X_test, y_train, y_test = load_dataset("GunPoint")
y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

print(f"Train shape : {X_train.shape}")
print(f"Test  shape : {X_test.shape}")
print(f"Classes     : {label_encoder.classes_}")

# ============================================================
# Cell 3 – ROCKET: generate kernels, transform, train, evaluate
# ============================================================

NUM_KERNELS = 300  # thesis used 300; paper default is 10 000

X_train_t, X_test_t, kernels = build_rocket_features(
    X_train, X_test, num_kernels=NUM_KERNELS
)

classifier = train_ridge_classifier(X_train_t, y_train_enc)
accuracy   = evaluate_classifier(classifier, X_test_t, y_test_enc)

# ============================================================
# Cell 4 – Feature-impact segmentation
# ============================================================

# Flatten coefficients (binary ridge → shape (1, n_features) or (n_classes, n_features))
coefs = classifier.coef_.reshape(-1)
if classifier.coef_.shape[0] > 1:
    coefs = coefs[:X_test_t.shape[1]]  # take first class row if multiclass

PERCENTAGE_TOP = 0.20
N_TOP          = int(len(coefs) * PERCENTAGE_TOP)
KERNEL_SIZE    = 100   # threshold separating "below" from "above" index groups
TS_IDX         = 51    # test-set instance to explain

feature_impacts = calculate_feature_impacts(coefs, X_test_t)
predicted_class = classifier.predict(X_test_t[TS_IDX].reshape(1, -1))
print(f"Predicted class for instance {TS_IDX}: {predicted_class}")

groups = select_and_segment_features(feature_impacts, N_TOP, KERNEL_SIZE, TS_IDX)

for name, idx_arr in groups.items():
    print(f"  {name:12s}: {len(idx_arr)} kernels")

# ============================================================
# Cell 5 – Feature maps and XAI attribution
# ============================================================

fm_results = {}
for group_name, kernel_indices in groups.items():
    if len(kernel_indices) == 0:
        print(f"  Skipping {group_name} (no kernels selected).")
        continue
    fm, ppv_contrib, mv_contrib = feature_map_xai(
        X_test[TS_IDX], kernels, kernel_indices
    )
    fm_results[group_name] = {
        "fm":         fm,
        "ppv":        ppv_contrib,
        "mv":         mv_contrib,
        "n_kernels":  len(kernel_indices),
    }
    print(f"  {group_name}: feature map shape {fm.shape}")

# ============================================================
# Cell 6 – Select most important kernels via mutual information
# ============================================================

important_results = {}
for group_name, res in fm_results.items():
    imp_ppv, imp_mv = select_kernels_with_mutual_info(
        res["ppv"], res["mv"],
        num_kernels=res["n_kernels"],
        y=y_test_enc,
        threshold=0.6,
    )
    kparams = get_kernel_parameters(imp_ppv, kernels)
    mapped  = map_kernels_to_input(kparams, X_test[TS_IDX])
    important_results[group_name] = {
        "kernel_params": kparams,
        "mapped":        mapped,
    }
    print(f"  {group_name}: {len(kparams)} important kernels identified")

# ============================================================
# Cell 7 – Saliency-map visualisation
# ============================================================

input_ts_2d = X_test[TS_IDX].reshape(1, -1)

for group_name, res in important_results.items():
    fig = plot_kernel_impact(
        input_ts=input_ts_2d,
        mapped_kernels=res["mapped"],
        title=f"Kernel Impact — {group_name} (without SHAP)",
    )
    plt.show()

# ============================================================
# Cell 8 – Faithfulness evaluation
# ============================================================

RECEPTIVE_FIELD = 10
eval_results    = {}

for group_name, res in important_results.items():
    mapped = res["mapped"]
    if not mapped:
        continue

    # Build a segment list from the kernel mappings
    segments = [
        (start, start + RECEPTIVE_FIELD)
        for kernel_mapping in mapped
        for start, _ in kernel_mapping
    ]
    # Pad / trim to match test-set size
    n_test = len(X_test_t)
    if len(segments) < n_test:
        reps     = n_test // len(segments)
        remainder = n_test % len(segments)
        segments = segments * reps + segments[:remainder]
    else:
        segments = segments[:n_test]

    faith_mean, faith_moe = calculate_faithfulness(
        model=classifier,
        X_test_transformed=X_test_t,
        important_segments=segments,
    )
    eval_results[group_name] = {
        "faithfulness":     faith_mean,
        "faithfulness_moe": faith_moe,
    }
    print(f"  [{group_name}] Faithfulness: {faith_mean:.4f} ± {faith_moe:.4f}")

# ============================================================
# Cell 9 – Robustness evaluation
# ============================================================

for group_name, res in important_results.items():
    kparams = res["kernel_params"]
    if not kparams:
        continue

    rob = calculate_robustness(
        kernel_parameters=kparams,
        input_ts=X_test[TS_IDX],
        num_perturbations=10,
    )
    eval_results[group_name]["robustness"]     = rob["mean"]
    eval_results[group_name]["robustness_moe"] = rob["margin_of_error"]
    print(
        f"  [{group_name}] Robustness: {rob['mean']:.4f} ± {rob['margin_of_error']:.4f}  "
        f"| unchanged: {rob['unchanged_proportion_mean']:.3f}"
    )

# ============================================================
# Cell 10 – Summary table
# ============================================================

fig = plot_results_table(eval_results, title="Faithfulness & Robustness — Without SHAP")
plt.show()
print("\nDone.")
