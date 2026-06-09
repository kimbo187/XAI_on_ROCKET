# Bridging the Gap: Enhancing Explainability in ROCKET

> **Master's Thesis Project · Stockholm University · Spring 2024**  
> Kamal Tikabo & Pamodou Touray  
> Department of Computer and Systems Sciences (DSV)  
> Supervisor: Zhendong Wang

---

## Overview

This repository contains a cleaned and refactored implementation from the master's thesis **"Bridging the Gap: Enhancing Explainability in ROCKET"**.

The project explores how **Explainable Artificial Intelligence (XAI)** can make the **ROCKET** time-series classifier more transparent. ROCKET is fast and accurate, but it relies on many randomly generated convolutional kernels, which makes its decision process difficult to interpret.

The main implementation in this repository focuses on the **intrinsic, without-SHAP baseline**. It maps influential ROCKET kernels back to the original time-series signal and evaluates the quality of explanations using **Faithfulness** and **Robustness**.

A companion SHAP integration script is also included to show how SHAP-based explanations can be connected to the same kernel-mapping pipeline.

---

## Why this project matters

Time-series classifiers are used in domains such as healthcare, finance, signal processing and activity recognition. In these areas, model accuracy is not enough. Users also need to understand why a model made a prediction.

ROCKET transforms time-series data using random convolutional kernels. This makes the model efficient, but the connection between the final prediction and the original signal is not obvious. This project investigates how to bridge that gap by tracing important transformed features back to raw time steps.

---

## Repository structure

```text
xai_rocket/
│
├── src/
│   ├── __init__.py
│   ├── rocket_core.py          # ROCKET kernel generation and transformation
│   ├── xai_methods.py          # Intrinsic XAI, kernel mapping, SHAP mapping helper
│   ├── pipeline.py             # Dataset loading, training, evaluation and metrics
│   └── visualization.py        # Saliency-style plots and result tables
│
├── notebooks/
│   ├── XAI_on_ROCKET_without_SHAP.py
│   └── XAI_on_ROCKET_with_SHAP.py
│
├── docs/
│   └── thesis_context.md
│
├── results/
│   ├── accuracy_summary.txt
│   └── faithfulness_robustness_summary.csv
│
├── run_experiment.py
├── requirements.txt
├── requirements-shap.txt
├── LICENSE
└── README.md
```

---

## Method summary

### 1. ROCKET transformation

ROCKET generates random 1D convolutional kernels with different:

- kernel lengths,
- weights,
- biases,
- dilations,
- paddings.

Each kernel produces two features:

| Feature | Meaning |
|---|---|
| **PPV** | Proportion of Positive Values |
| **MV** | Maximum Value |

The transformed feature matrix is then used to train a linear classifier, in this case `RidgeClassifierCV`.

### 2. Intrinsic XAI pipeline

The intrinsic explanation pipeline works by:

1. training ROCKET + Ridge classifier,
2. calculating feature impact as classifier coefficient × transformed feature value,
3. separating influential features into positive and negative groups,
4. mapping important kernel indices back to kernel parameters,
5. mapping receptive fields back to the original time-series signal,
6. visualising influential time steps with saliency-style plots,
7. evaluating explanations with Faithfulness and Robustness.

### 3. SHAP companion integration

The thesis also investigated SHAP-based explanations. In this repository, the SHAP integration is provided as an optional companion script:

```text
notebooks/XAI_on_ROCKET_with_SHAP.py
```

Install optional SHAP dependencies with:

```bash
pip install -r requirements-shap.txt
```

---

## Evaluation metrics

| Metric | Purpose |
|---|---|
| **Faithfulness** | Measures whether perturbing important segments changes model confidence. |
| **Robustness** | Measures whether explanations remain stable under small input noise. |

Higher Faithfulness means the explanation has identified truly influential segments. Lower Robustness values mean the explanation is more stable under perturbation.

---

## Reported thesis results

The thesis experiments reported that ROCKET achieved **up to 100% accuracy** on the GunPoint dataset under the tested configuration.

The reported Faithfulness and Robustness summaries are included in:

```text
results/faithfulness_robustness_summary.csv
results/accuracy_summary.txt
```

Exact numerical results may vary slightly depending on random seed, dependency versions and kernel generation.

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/kimbo187/XAI_on_ROCKET.git
cd XAI_on_ROCKET
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the optional SHAP companion script:

```bash
pip install -r requirements-shap.txt
```

### 4. Run the intrinsic baseline

```bash
python run_experiment.py
```

Or run the script directly:

```bash
python notebooks/XAI_on_ROCKET_without_SHAP.py
```

You can also open the notebook version:

```text
notebooks/XAI_on_ROCKET_without_SHAP.ipynb
```

---

## Skills demonstrated

This project demonstrates:

- time-series classification,
- ROCKET kernel transformation,
- Explainable AI for time-series models,
- model evaluation with Faithfulness and Robustness,
- NumPy-based feature analysis,
- Numba-accelerated computation,
- scikit-learn model training,
- scientific experimentation,
- code refactoring and documentation.

---

## Thesis connection

This repository is connected to the master's thesis **"Bridging the Gap: Enhancing Explainability in ROCKET"**. The thesis investigated how intrinsic XAI and SHAP-inspired explanations can improve transparency in ROCKET-based time-series classification.

More context is available in:

```text
docs/thesis_context.md
```

---

## References

- Dempster, A., Petitjean, F., & Webb, G. I. (2020). *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels*. Data Mining and Knowledge Discovery.
- Rastegar, A. (2023). *Explaining ROCKET time series classifier model*.
- Tikabo, K., & Touray, P. (2024). *Bridging the Gap: Enhancing Explainability in ROCKET*. Master's thesis, Stockholm University.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
