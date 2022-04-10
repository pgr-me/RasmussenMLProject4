#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, utils.py

This module provides miscellaneous utility functions that support the core perceptrons of this program.

"""
# Third party libraries
import numpy as np
from numba import njit


@njit
def mse(Y, Yhat):
    return np.sum(np.square(np.subtract(Y, Yhat))) / len(Y)


def cross_entropy(Y: np.array, Yhat: np.array) -> float:
    return -np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))


def sigmoid(output):
    return 1 / (1 + np.exp(-output))
