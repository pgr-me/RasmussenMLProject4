#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, regression_perceptron.py

This module provides the KNN class, the base class of KNNClassifier and KNNRegressor classes.

"""
# Third party libraries
import numpy as np
from numba import njit
import numba as nb
import pandas as pd


def accuracy(Y, Yhat):
    maxes = np.broadcast_to(np.max(Yhat, axis=1), Yhat.T.shape).T
    pred = np.greater_equal(Yhat, maxes).astype(np.uint8)
    pred_correct = (Y == pred).sum(axis=1) == Y.shape[1]
    return np.sum(pred_correct) / len(pred_correct)


def dummy_categorical_label(data: pd.DataFrame, label: str) -> tuple:
    """
    Dummy categorical label.
    :param data: Dataframe
    :param label: Label column
    :return: Tuple of data with dummied labels and label cols
    """

    dummied_labels: pd.DataFrame = pd.get_dummies(data[label].astype("category"), prefix=label)
    # Sort dummied cols in ascending order
    label_cols = sorted(list(dummied_labels))
    return dummied_labels[label_cols], label_cols


def gradient(eta, Y, Yhat, X):
    return eta * (Y - Yhat).T.dot(X)


def predict_output(w, X):
    return w.dot(X.T).T


def normalize_output(output):
    return (np.exp(output.T) / np.sum(np.exp(output.T), axis=0)).T


def predict(w, X):
    output = predict_output(w, X)
    normed_output = normalize_output(output)
    return normed_output


def cross_entropy(Y: np.array, Yhat: np.array) -> float:
    return -np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))


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


@njit
def train_classification_perceptron(Y: nb.float64[:], X: nb.float64[:], eta: float = 0.1, thresh: float = 0.01, max_iter: int = 500):
    """
    Train online perceptron for regression.
    :param Y: Target
    :param X: Features
    :param eta: Learning rate
    :param thresh: MSE fractional improvement threshold
    :param max_iter: Number of iterations
    :return: Trained weights
    """
    w = (np.random.rand(1, X.shape[1]) - 0.5) * 2 / 100
    mse = np.abs(0.00001)
    iteration = 0
    while True:
        # Randomly shuffle indices
        indices = shuffle_indices(Y.shape[0])

        # Update weights
        for index in indices:
            w = w + gradient(Y, X, w, index, eta)

        # Compute MSE and improvement from the latest iteration
        Yhat = predict(X, w)
        new_mse = compute_mse(Y, Yhat)
        delta = 1 - (mse - new_mse) / mse
        mse = new_mse
        iteration = iteration + 1

        # Return if delta improvement attained or max iterations reached
        if (delta <= thresh) or (iteration >= max_iter):
            return w
