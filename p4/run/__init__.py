#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, p4/run/__init__.py

"""
# Classification models
from p4.run.run_classification_autoencoder import run_classification_autoencoder
from p4.run.run_classification_mlp import run_classification_mlp
from p4.run.run_classification_perceptron import run_classification_perceptron

# Regression models
from p4.run.run_regression_autoencoder import run_regression_autoencoder
from p4.run.run_regression_mlp import run_regression_mlp
from p4.run.run_regression_perceptron import run_regression_perceptron