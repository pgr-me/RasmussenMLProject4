#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, self.py

This module provides the MLP class for multi-layer perceptron training, validation, and prediction routines.

Sources: Lecture notes, Alpaydin, https://zerowithdot.com/mlp-backpropagation/, and
https://brilliant.org/wiki/backpropagation/
"""
# Local imports
from p4.utils import bias, cross_entropy, mse, shuffle_indices


class MLP:
    def __init__(self, layers: list, D: int, eta: float, problem_class: str, n_runs: int = 200):
        self.layers = layers
        self.D = D  # Number of input dimensions
        self.eta = eta  # Learning rate
        self.problem_class = problem_class
        self.n_runs = n_runs
        self.Yhat = None
        self.n_layers = len(self.layers)
        self.scores = []

        for layer in self.layers:
            setattr(self, layer.name, layer)

    def __repr__(self):
        return f"{len(self.layers)}-layer MLP"

    def backpropagate(self, X, Y, Yhat):
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
        n_input_units = self.D
        self.layers[0].initialize_weights()
        for layer in self.layers[1:]:
            layer.n_input_units = n_input_units
            layer.initialize_weights()
            n_input_units = layer.n_units

    def predict(self, X):
        Z = self.layers[0].predict(X)
        for layer in self.layers[1:]:
            Z = layer.predict(Z)
        self.Yhat = Z
        return self.Yhat

    def score(self, Y, Yhat):
        if self.problem_class == "classification":
            return cross_entropy(Y, Yhat)
        return mse(Y, Yhat)

    def train(self, Y, X, Y_val=None, X_val=None):
        self.initialize_weights()
        run_validation = (Y_val is not None) and (X_val is not None)
        for run in range(self.n_runs):
            indices = shuffle_indices(len(X))
            Yhat = self.predict(X[indices, :])
            self.backpropagate(X[indices, :], Yhat, Y[indices, :])
            if run_validation:
                Yhat_val = self.predict(X_val)
                score = self.score(Y_val, Yhat_val)
                self.scores.append(score)
