# Thesis Context

This repository is based on the master thesis **"Bridging the Gap: Enhancing Explainability in ROCKET"**, written by Kamal Tikabo and Pamodou Touray at the Department of Computer and Systems Sciences, Stockholm University, Spring 2024.

## What the thesis investigated

The thesis studied how explainability can be improved for **ROCKET** (RandOm Convolutional KErnel Transform), a fast and accurate time-series classification method that transforms raw time series using thousands of randomly generated convolutional kernels.

ROCKET is efficient, but its reliance on random kernels makes it difficult to understand why a prediction was made. The thesis therefore explored how Explainable Artificial Intelligence (XAI) methods can help make ROCKET's internal behaviour more transparent.

## Why ROCKET is hard to explain

ROCKET generates many random kernels with different lengths, weights, biases, dilations and paddings. Each kernel produces two transformed features:

- **PPV**: Proportion of Positive Values
- **MV**: Maximum Value

A linear classifier is then trained on these transformed features. Although the classifier is linear, the connection between the transformed features and the original time-series signal is not immediately interpretable.

## What this repository focuses on

This repository refactors the code used to investigate intrinsic explainability for ROCKET. The implementation focuses on:

- generating and applying ROCKET kernels,
- training a Ridge classifier on transformed features,
- calculating feature impacts from classifier coefficients,
- grouping influential features by positive/negative impact,
- mapping important kernels back to raw time-series locations,
- visualising kernel impact as saliency-style plots,
- evaluating explanations using Faithfulness and Robustness.

## SHAP context

The thesis also discusses and evaluates SHAP-based explanations. This repository includes utilities and a companion script showing where SHAP values can be integrated, but the main executable pipeline is the intrinsic, without-SHAP baseline.

## Evaluation metrics

### Faithfulness

Faithfulness measures whether the explanation points to segments that actually influence the classifier. Important segments are perturbed, and the resulting change in model confidence is measured.

### Robustness

Robustness measures how stable explanations are under small input perturbations. A robust explanation should not change drastically when minor noise is added to the input signal.

## Portfolio purpose

The purpose of this repository is to present the thesis implementation in a cleaner, more reusable and recruiter-friendly format. It is intended to demonstrate applied machine learning, time-series classification, model interpretability, scientific experimentation and Python refactoring skills.
