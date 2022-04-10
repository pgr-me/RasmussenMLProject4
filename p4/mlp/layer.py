#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, layer.py

This module provides the Layer class for multi-layer perceptron training, validation, and prediction routines.

"""
import typing as t
# Third party libraries
import numpy as np

from p4.utils import sigmoid


class Layer:
    def __init__(self, name: str, n_units: int, n_input_units: t.Union[int, None], apply_sigmoid: bool = True):
        self.name = name
        self.n_input_units = n_input_units
        self.n_units = n_units
        self.apply_sigmoid = apply_sigmoid
        self.W = None
        self.X = None
        self.Z = None

    def __repr__(self):
        return f"Layer {self.name}"

    def initialize_weights(self):
        n_input_units_bias = self.n_input_units + 1
        self.W = (np.random.rand(self.n_units, n_input_units_bias) - 0.5) * 2 / 100

    def predict(self, X):
        X = np.hstack([np.ones((len(X), 1)), X])
        o = self.W.dot(X.T).T
        self.Z = sigmoid(o) if self.apply_sigmoid else o
        return self.Z
