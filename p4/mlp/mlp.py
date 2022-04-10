#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, mlp.py

This module provides the MLP class for multi-layer perceptron training, validation, and prediction routines.

"""
# Standard library imports
import typing as t
# Third party libraries
import numpy as np
from numba import njit
import numba as nb
import pandas as pd
# Local imports
from p4.utils import cross_entropy, mse


class MLP:
    def __init__(self, layers: list, D: int, eta: float, problem_class: str):
        self.layers = layers
        self.D = D  # Number of input dimensions
        self.eta = eta  # Learning rate
        self.problem_class = problem_class
        self.Yhat = None

        for layer in self.layers:
            setattr(self, layer.name, layer)

    def __repr__(self):
        return f"{len(self.layers)}-layer MLP"

    def initialize_weights(self):
        n_input_units = self.D
        self.layers[0].initialize_weights()
        for layer in self.layers[1:]:
            layer.n_input_units = n_input_units
            layer.initialize_weights()
            print(layer, layer.n_input_units, layer.n_units)
            n_input_units = layer.n_units

    def predict(self, X):
        Z = self.layers[0].predict(X)
        for layer in self.layers[1:]:
            print(layer)
            Z = layer.predict(Z)
        self.Yhat = Z
        return self.Yhat

    def score(self, Y, Yhat):
        if self.problem_class == "classification":
            return cross_entropy(Y, Yhat)
        return mse(Y, Yhat)