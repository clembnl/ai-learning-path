"""Smoke tests: every step runs end to end and meets its success criterion.

Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from common.data import Standardizer, generate, load_split_standardized, split
from common.metrics import accuracy, confusion_matrix, r2


# --------------------------------------------------------------------------- data
def test_generate_shape_and_balance():
    data = generate(n_samples=1000, seed=42)
    assert data.shape == (1000, 4)
    assert 0.40 <= data[:, 3].mean() <= 0.60, "pass rate should be roughly balanced"


def test_split_sizes_and_disjoint():
    X, score, y = generate(200)[:, :2], generate(200)[:, 2], generate(200)[:, 3]
    s = split(X, score, y)
    assert len(s.X_train) == 140 and len(s.X_val) == 30 and len(s.X_test) == 30
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == 200


def test_split_rejects_bad_fractions():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        split(X, np.zeros(10), np.zeros(10), train=0.9, val=0.2)


def test_standardizer_uses_train_stats():
    data, scaler = load_split_standardized()
    assert np.allclose(data.X_train.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(data.X_train.std(axis=0), 1, atol=1e-9)
    assert scaler.mean.shape == (2,)


# ------------------------------------------------------------------------ metrics
def test_metrics_basics():
    y = np.array([0, 1, 1, 0])
    assert accuracy(y, np.array([0, 1, 0, 0])) == 0.75
    assert confusion_matrix(y, np.array([0, 1, 0, 0])).tolist() == [[2, 0], [1, 1]]
    assert r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0


# -------------------------------------------------------------------------- steps
def test_step01_gd_matches_closed_form_and_fits():
    from steps.step01_linear_regression import main

    r = main(verbose=False)
    assert np.abs(r["w"] - r["w_closed_form"]).max() < 1e-3
    assert abs(r["b"] - r["b_closed_form"]) < 1e-3
    assert r["r2"] >= 0.75


def test_step02_logistic_accuracy():
    from steps.step02_logistic_regression import main

    assert main(verbose=False)["accuracy"] >= 0.80


def test_step03_perceptron_runs_and_is_reasonable():
    from steps.step03_perceptron import main

    r = main(verbose=False)
    assert r["accuracy"] >= 0.70
    assert len(r["errors_per_epoch"]) > 0


def test_step04_sklearn_matches_numpy():
    from steps.step04_sklearn import main

    r = main(verbose=False)
    assert r["weight_gap_linear"] < 1e-3
    assert r["weight_gap_logistic"] < 5e-2
    assert abs(r["logistic_numpy"]["accuracy"] - r["logistic_sklearn"]["accuracy"]) <= 0.02


def test_step05_torch_matches_numpy():
    from steps.step02_logistic_regression import main as np_main
    from steps.step05_pytorch import main

    r = main(verbose=False)
    assert r["linear"]["r2"] >= 0.75
    assert abs(r["logistic"]["accuracy"] - np_main(verbose=False)["accuracy"]) <= 0.03


def test_step06_mlp_beats_linear_models():
    from steps.step06_neural_network import main

    r = main(verbose=False)
    assert r["MLP 2-16-1 (torch)"] >= r["logistic (numpy)"] + 0.02
    assert r["MLP 2-16-1 (torch)"] >= 0.90
