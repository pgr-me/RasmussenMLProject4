"""Peter Rasmussen, Programming Assignment 4, __main__.py

This program trains a simple majority predictor across six datasets to estimate the 1) class for
classification datasets or 2) mean for regression datasets. Classification datasets are scored on
accuracy and regression datasets are scored by mean squared error. Data is split into test, train,
and validation sets and the user specifies the number of folds to use.

Inputs: Six datasets obtained from the course website are used in this analysis. They are available in
the RasmussenMLProject1/data directory of this repo.

Outputs: Two outputs are generated and saved to a user-specified directory. The first is
output.csv. This provides more detailed, fold-level scoring and parameter (beta) outputs. The second,
summary.csv, provides dataset-level performance statistics.

The structure of this package is based on the Python lab0 package that Scott Almes developed for
Data Structures 605.202. Per Scott, this module "is the entry point into this program when the
module is executed as a standalone program."

"""

# standard library imports
import argparse
import logging
import os
from pathlib import Path

# local imports
from p4.run import run_classification_autoencoder, run_classification_mlp, run_classification_perceptron
from p4.run import run_regression_autoencoder, run_regression_mlp, run_regression_perceptron

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_dir", "-i", type=Path, help="Input directory"
)
parser.add_argument(
    "--dst_dir", "-o", type=Path, help="Output directory"
)
parser.add_argument(
    "--k_folds", "-k", default=5, type=int, help="Number of folds to partition data"
)
parser.add_argument(
    "--val_frac", "-v", default=0.1, type=float, help="Fraction of validation samples"
)
parser.add_argument(
    "--random_state", "-r", default=777, type=int, help="Pseudo-random seed"
)
args = parser.parse_args()


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
log_path = dir_path / "p4.log"
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)

logging.debug(f"Begin: src_dir={args.src_dir.name}, dst_dir={args.dst_dir.name}.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.debug("run_classification_autoencoder")
run_classification_autoencoder(args.src_dir, args.dst_dir, args.k_folds, args.val_frac)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.debug("run_classification_mlp")
run_classification_mlp(args.src_dir, args.dst_dir, args.k_folds, args.val_frac)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.debug("run_classification_perceptron")
run_classification_perceptron(args.src_dir, args.dst_dir, args.k_folds, args.val_frac)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.debug("run_classification_autoencoder")
run_regression_autoencoder(args.src_dir, args.dst_dir, args.k_folds, args.val_frac)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.debug("run_classification_mlp")
run_regression_mlp(args.src_dir, args.dst_dir, args.k_folds, args.val_frac)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.debug("run_classification_perceptron")
run_regression_perceptron(args.src_dir, args.dst_dir, args.k_folds, args.val_frac)

logging.debug("Finish.\n")



