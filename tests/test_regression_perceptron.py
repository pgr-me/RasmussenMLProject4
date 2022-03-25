#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

K-Folds cross validation strategy:
    Each fold-run is its own experiment
    Assign each observation to one of five folds

    # Do validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For fold i:
        fold i is test
        fold ~i is train-val
        Split train-val into train and val (80 / 20)
        Train on train
        Predict trained model using different param sets on val
    Take best params over all fold i's: Take a mean to determine best params

    # Do testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For fold i:
         ...
        Test on best params

"""
# Standard library imports
import collections as c
from copy import deepcopy
import json
from pathlib import Path
import warnings

# Third party imports
import numpy as np
import pandas as pd
from numba import jit, njit
import numba as nb

# Local imports
from p4.preprocessing import Preprocessor
from p4.algorithms.regression_perceptron import compute_mse, train_perceptron
from p4.preprocessing.split import make_splits
from p4.preprocessing.standardization import get_standardization_params, standardize, get_standardization_cols

warnings.filterwarnings('ignore')

test_dir = Path(".").absolute()
repo_dir = test_dir.parent
p4_dir = repo_dir / "p4"
src_dir = repo_dir / "data"

# Specify k folds
k_folds = 2
val_frac = 0.2


# Load data catalog and tuning params
with open(src_dir / "data_catalog.json", "r") as file:
    data_catalog = json.load(file)
data_catalog = {k: v for k, v in data_catalog.items() if k in ["forestfires", "machine", "abalone"]}
data_catalog = {k: v for k, v in data_catalog.items() if k in ["machine"]}


def test_regression_perceptron():
    for dataset_name, dataset_meta in data_catalog.items():
        preprocessor = Preprocessor(dataset_name, dataset_meta, src_dir)
        preprocessor.load()
        preprocessor.drop()
        preprocessor.identify_features_label_id()
        preprocessor.replace()
        preprocessor.log_transform()
        preprocessor.set_data_classes()
        preprocessor.impute()
        preprocessor.dummy()
        preprocessor.set_data_classes()
        preprocessor.shuffle()

        # Extract feature and label columns
        feature_cols = preprocessor.features
        label_col = preprocessor.label
        problem_class = dataset_meta["problem_class"]
        index = preprocessor.data.index.values
        data = preprocessor.data.copy()
        if problem_class == "classification":
            data[label_col] = data[label_col].astype(int)

        # Assign folds
        data["fold"] = make_splits(data, problem_class, label_col, k_folds=k_folds, val_frac=None)

        # Iterate over each fold-run
        for fold in range(1, k_folds + 1):
            test_mask = data["fold"] == fold
            test = data.copy()[test_mask].drop(axis=1, labels="fold")
            train_val = data.copy()[~test_mask].drop(axis=1, labels="fold")
            train_val["train"] = make_splits(train_val, problem_class, label_col, k_folds=None, val_frac=val_frac)
            train_mask = train_val["train"] == 1
            train = train_val.copy()[train_mask].drop(axis=1, labels="train")
            val = train_val.copy()[~train_mask].drop(axis=1, labels="train")

            # Get standardization parameters from training-validation set
            cols = get_standardization_cols(train_val, feature_cols)
            means, std_devs = get_standardization_params(data.copy()[cols])

            # Standardize data
            test = test.drop(axis=1, labels=cols).join(standardize(test[cols], means, std_devs))
            train_val = train_val.drop(axis=1, labels=cols).join(standardize(train_val[cols], means, std_devs))
            print('yolo')
            pass

        splits = make_splits(data, problem_class, label_col, k_folds=None, val_frac=val_frac)
        testing_data = preprocessor.data.copy().sample(frac=0.2, random_state=777)
        # preprocessor.discretize()
        # testing_data = preprocessor.discretize_nontrain(testing_data)
        data = preprocessor.data.copy().sample(frac=0.8, random_state=777)

        # Extract label, features, and data classes
        label = preprocessor.label
        features = [x for x in preprocessor.features if x != label]
        data = data[[label] + features]
        data["intercept"] = 1

        eta = 0.1
        YX = data.copy().astype(np.float64).values
        Y, X = YX[:, 0].reshape(len(YX), 1), YX[:, 1:]
        weights = train_perceptron(Y, X)
        mse = compute_mse(Y, X, weights)
        print(f"{dataset_name}: {mse:.2f}")

if __name__ == "__main__":
    test_regression_perceptron()
