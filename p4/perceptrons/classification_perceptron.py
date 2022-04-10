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


