"""
rocket_core.py
==============
Core ROCKET kernel generation and transformation functions.

Based on the original ROCKET paper:
    Dempster, A., Petitjean, F., & Webb, G. I. (2020).
    ROCKET: Exceptionally fast and accurate time series classification
    using random convolutional kernels.
    Data Mining and Knowledge Discovery, 34(5), 1454–1495.
    https://doi.org/10.1007/s10618-020-00701-z
"""

import numpy as np
from numba import njit, prange


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length: int, num_kernels: int):
    """
    Generate random convolutional kernels for the ROCKET method.

    Each kernel has randomly initialised length (7, 9, or 11), weights
    drawn from N(0,1) and mean-centred, a uniform bias in [-1, 1],
    an exponentially distributed dilation, and a random binary padding flag.

    Parameters
    ----------
    input_length : int
        Length of the input time series.
    num_kernels : int
        Number of kernels to generate.

    Returns
    -------
    weights : np.ndarray of float64
        Concatenated weight vectors for all kernels.
    lengths : np.ndarray of int32
        Length of each kernel (7, 9, or 11).
    biases : np.ndarray of float64
        Bias term for each kernel.
    dilations : np.ndarray of int32
        Dilation factor for each kernel.
    paddings : np.ndarray of int32
        Padding amount for each kernel (0 = no padding).
    """
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights  = np.zeros(lengths.sum(), dtype=np.float64)
    biases   = np.zeros(num_kernels,   dtype=np.float64)
    dilations = np.zeros(num_kernels,  dtype=np.int32)
    paddings  = np.zeros(num_kernels,  dtype=np.int32)

    a1 = 0
    for i in range(num_kernels):
        _length  = lengths[i]
        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()   # zero-mean

        biases[i] = np.random.uniform(-1, 1)

        max_exp  = np.log2((input_length - 1) / (_length - 1))
        dilation = np.int32(2 ** np.random.uniform(0, max_exp))
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    """
    Apply a single convolutional kernel to an input time series.

    Computes two aggregate features:
      - PPV : Proportion of Positive Values  (fraction of positive convolution outputs)
      - MV  : Maximum Value                  (max convolution output)

    Parameters
    ----------
    X : np.ndarray of float64, shape (input_length,)
    weights, length, bias, dilation, padding : kernel parameters

    Returns
    -------
    ppv : float  – proportion of positive outputs
    max_val : float – maximum output value
    """
    input_length  = len(X)
    output_length = (input_length + 2 * padding) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF
    end  = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum  = bias
        index = i
        for j in range(length):
            if 0 <= index < input_length:
                _sum += weights[j] * X[index]
            index += dilation

        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(
    "float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))",
    parallel=True,
    fastmath=True,
)
def apply_kernels(X, kernels):
    """
    Apply all kernels to every example in X (parallelised via Numba).

    Parameters
    ----------
    X : np.ndarray, shape (n_examples, n_features)
    kernels : tuple returned by generate_kernels

    Returns
    -------
    _X : np.ndarray, shape (n_examples, num_kernels * 2)
        Concatenated [PPV, MV] features for every kernel.
    """
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels     = len(lengths)
    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float64)

    for i in prange(num_examples):
        a1 = a2 = 0
        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2
            _X[i, a2:b2] = apply_kernel(
                X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j]
            )
            a1, a2 = b1, b2

    return _X
