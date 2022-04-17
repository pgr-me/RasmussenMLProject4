#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, mlp.py

This module provides the MLP class for multi-layer perceptron training, validation, and prediction routines.

Sources: Lecture notes, Alpaydin, https://zerowithdot.com/mlp-backpropagation/, and
https://brilliant.org/wiki/backpropagation/
"""
# Third party imports
import numpy as np

# Local imports
from p4.utils import accuracy, bias, cross_entropy, mse, shuffle_indices


class MLP:
    """
    Supports classification and regression.
    """

    def __init__(self, layers: list, D: int, eta: float, problem_class: str, n_runs: int = 200, name: str = None):
        self.layers = layers
        self.D = D  # Number of input dimensions
        self.eta = eta  # Learning rate
        self.problem_class = problem_class
        self.n_runs = n_runs
        self.name = name
        self.Yhat = None
        self.n_layers = len(self.layers)
        self.val_scores = []
        self.tr_scores = []
        self.val_acc = []
        self.tr_acc = []

        for layer in self.layers:
            setattr(self, layer.name, layer)

    def __repr__(self):
        return f"{len(self.layers)}-layer MLP"

    def backpropagate(self, X, Y, Yhat):
        """
        Backpropagate from output layer to input layer.
        :param X: Inputs
        :param Y: True values
        :param Yhat: Predicted values
        """
        backprop_error = Y - Yhat
        weight_changes = {}
        for index in reversed(range(1, self.n_layers)):
            layer = self.layers[index]
            preceding_layer = self.layers[index - 1]
            preceding_Z = preceding_layer.Z
            weight_change = backprop_error.T.dot(bias(preceding_Z))
            weight_changes[index] = weight_change
            backprop_error = layer.backprop_error(backprop_error, preceding_Z)
        weight_changes[0] = backprop_error.T.dot(bias(X))

        for index, weight_change in weight_changes.items():
            layer = self.layers[index]
            layer.W = layer.W - self.eta * weight_change

    def initialize_weights(self):
        """
        Initialize weights for each layer.
        """
        n_input_units = self.D
        self.layers[0].initialize_weights()
        for layer in self.layers[1:]:
            layer.n_input_units = n_input_units
            layer.initialize_weights()
            n_input_units = layer.n_units

    def predict(self, X) -> np.array:
        """
        Predict forward from input layer to output layer.
        :param X: Input feature matrix
        :return: Predicted output (i.e., Yhat)
        """
        Z = self.layers[0].predict(X)
        for layer in self.layers[1:]:
            Z = layer.predict(Z)
        self.Yhat = Z
        return self.Yhat

    def score(self, Y, Yhat) -> float:
        """
        If classification, compute cross entropy; otherwise compute MSE.
        :param Y: True values
        :param Yhat: Predicted values
        :return: Score
        """
        if self.problem_class == "classification":
            return cross_entropy(Y, Yhat)
        return mse(Y, Yhat)

    def train(self, Y_tr, X_tr, Y_val=None, X_val=None):
        """
        Train the MLP.
        :param Y_tr: Training true values
        :param X_tr: Training feature matrix
        :param Y_val: Validation (or test) true values
        :param X_val: Validation (or test) feature matrix
        """
        run_validation = (Y_val is not None) and (X_val is not None)
        for run in range(self.n_runs):
            indices = shuffle_indices(len(X_tr))
            Yhat_tr = self.predict(X_tr[indices, :])
            self.tr_scores.append(self.score(Y_tr[indices, :], Yhat_tr))
            if self.problem_class == "classification":
                self.tr_acc.append(accuracy(Y_tr, Yhat_tr))
            self.backpropagate(X_tr[indices, :], Yhat_tr, Y_tr[indices, :])
            if run_validation:
                Yhat_val = self.predict(X_val)
                self.val_scores.append(self.score(Y_val, Yhat_val))
                if self.problem_class == "classification":
                    self.val_acc.append(accuracy(Y_val, Yhat_val))
