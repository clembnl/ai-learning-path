"""Cell definitions for the guided notebooks. Each module exposes CELLS: list[(kind, source)]."""

SETUP = '''import sys, pathlib
ROOT = next((p for p in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents)
             if (p / "common" / "data.py").is_file() and (p / "steps").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Open the notebook from inside the downloaded repository.")
sys.path.insert(0, str(ROOT))   # shared modules from root or notebooks/
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from common.data import load_split_standardized, load
from common.metrics import *
from common.plotting import plot_loss, plot_regression_fit, plot_decision_boundaries
data, scaler = load_split_standardized()
print("train / val / test sizes:", len(data.X_train), len(data.X_val), len(data.X_test))'''
