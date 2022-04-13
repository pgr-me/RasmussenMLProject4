#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, utils.py

This module provides miscellaneous utility functions that support the core perceptrons of this program.

"""
# Third party libraries
import numba as nb
import numpy as np
from numba import njit


def bias(X):
    bias_term = np.ones([len(X), 1])
    return np.hstack([X, bias_term])

def cross_entropy(Y: np.array, Yhat: np.array) -> float:
    return -np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))

@njit
def mse(Y, Yhat):
    return np.sum(np.square(np.subtract(Y, Yhat))) / len(Y)




def sigmoid(output):
    return 1 / (1 + np.exp(-output))

def sigmoid_update(weights):
    return weights * (1 - weights)


@njit
def shuffle_indices(n: int) -> nb.int64[:]:
    """
    Shuffle a zero-indexed index.
    :param n: Number of elements in array
    :return: Array of randomly ordered indices
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices