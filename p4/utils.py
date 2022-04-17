#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, p4/utils.py

This module provides miscellaneous utility functions.

"""
# Third party libraries
import numpy as np
import pandas as pd


def accuracy(Y: np.array, Yhat: np.array) -> float:
    """
    Compute accuracy of prediction given true values.
    :param Y: Vector of true values
    :param Yhat: Vector of predicted values
    :return: Accuracy
    """
    maxes = np.broadcast_to(np.max(Yhat, axis=1), Yhat.T.shape).T
    pred = np.greater_equal(Yhat, maxes).astype(np.uint8)
    pred_correct = (Y == pred).sum(axis=1) == Y.shape[1]
    return np.sum(pred_correct) / len(pred_correct)


def bias(X: np.array) -> np.array:
    """
    Add an intercept (bias) to a feature set.
    :param X: Feature matrix
    :return: Feature matrix + column of ones
    """
    bias_term = np.ones([len(X), 1])
    return np.hstack([X, bias_term])
    #return np.hstack([bias_term, X])


def cross_entropy(Y: np.array, Yhat: np.array) -> float:
    """
    Compute the cross entropy from true and predicted values
    :param Y: True values
    :param Yhat: Predicted values
    :return: Cross entropy
    """
    return -np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))


def mse(Y: np.array, Yhat: np.array) -> float:
    """
    Compute the mean square error (MSE).
    :param Y: Truth
    :param Yhat: Predicted
    :return: MSE
    """
    return np.sum(np.square(np.subtract(Y, Yhat))) / len(Y)


def sigmoid(output: np.array) -> np.array:
    """
    Compute the sigmoid of an array.
    :param output: Output to be transformed
    :return: Sigmoid-transformed output
    """
    return 1 / (1 + np.exp(-output))


def sigmoid_update(weights: np.array) -> np.array:
    """
    Helper function used in backpropagation.
    :param weights: Weights array
    :return: Update array
    """
    return weights * (1 - weights)


def shuffle_indices(n: int) -> np.array:
    """
    Shuffle a zero-indexed index.
    :param n: Number of elements in array
    :return: Array of randomly ordered indices
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices


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
