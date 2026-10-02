"""Step 6 - A small neural network (multi-layer perceptron).

WHAT YOU LEARN
    * Stacking neurons:  Linear(2,16) -> ReLU -> Linear(16,1)
      The hidden layer learns 16 "features" of the input; the output neuron is
      a logistic regression ON THOSE LEARNED FEATURES.
    * Without the non-linearity (ReLU) two stacked linear layers collapse into a
      single linear layer - the activation is what makes depth useful.
    * Backpropagation = the chain rule applied layer by layer; autograd does it.
    * The decision boundary is no longer a straight line: the network can bend
      around the sleep sweet spot that steps 2-5 could not capture.
    * Validation set for early stopping / model selection; Adam optimizer.
    * Alternative: hand-craft the feature slept^2 and feed logistic regression.
      Feature engineering by hand vs. features learned by the network.

TASK
    Same classification, compare against every previous linear model.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from common.data import Split, load_split_standardized
from common.metrics import accuracy, classification_report
from common.plotting import plot_decision_boundaries, plot_loss
from steps.step02_logistic_regression import LogisticRegressionGD
from steps.step03_perceptron import Perceptron, to_pm1
from steps.step05_pytorch import build_logistic_regression, make_loader, predict_class, to_tensor, train


def build_mlp(hidden: int = 16, seed: int = 0) -> nn.Module:
    """2 -> hidden -> 1 with a ReLU in between.

    The seed is applied here so weight initialisation is reproducible regardless of
    how much torch randomness was consumed earlier in the session (notebooks!).
    """
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def train_with_early_stopping(model: nn.Module, data: Split, epochs: int = 300, lr: float = 1e-2,
                              patience: int = 30, seed: int = 0) -> dict[str, list[float]]:
    """Train one epoch at a time and keep the weights with the lowest validation loss."""
    torch.manual_seed(seed)
    loader = make_loader(data.X_train, data.y_train)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val, best_state, bad_epochs = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        h = train(model, loader, loss_fn, opt, epochs=1, X_val=data.X_val, y_val=data.y_val)
        history["train"] += h["train"]
        history["val"] += h["val"]
        if h["val"][-1] < best_val - 1e-4:
            best_val, best_state, bad_epochs = h["val"][-1], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return history


def add_sleep_squared(X: np.ndarray) -> np.ndarray:
    """Hand-crafted feature: slept^2 lets a LINEAR model capture the sweet spot."""
    return np.column_stack([X, X[:, 1] ** 2])


def main(verbose: bool = True) -> dict:
    data, _ = load_split_standardized()

    # Earlier linear models for the final comparison
    perceptron = Perceptron(lr=0.01, epochs=50).fit(data.X_train, to_pm1(data.y_train))
    logistic_np = LogisticRegressionGD(lr=0.5, epochs=1000).fit(data.X_train, data.y_train)
    logistic_torch, _ = build_logistic_regression(data)
    logistic_fe = LogisticRegressionGD(lr=0.5, epochs=2000).fit(add_sleep_squared(data.X_train), data.y_train)

    # The neural network
    mlp = build_mlp(hidden=16)
    hist = train_with_early_stopping(mlp, data)

    results = {
        "perceptron": accuracy(data.y_test, perceptron.predict01(data.X_test)),
        "logistic (numpy)": accuracy(data.y_test, logistic_np.predict(data.X_test)),
        "logistic (torch)": accuracy(data.y_test, predict_class(logistic_torch, data.X_test)),
        "logistic + slept^2 feature": accuracy(data.y_test, logistic_fe.predict(add_sleep_squared(data.X_test))),
        "MLP 2-16-1 (torch)": accuracy(data.y_test, predict_class(mlp, data.X_test)),
    }
    if verbose:
        print(f"trained {len(hist['train'])} epochs (early stopping), best val BCE = {min(hist['val']):.4f}")
        n_params = sum(p.numel() for p in mlp.parameters())
        print(f"MLP parameters: {n_params}  (logistic regression has 3)\n")
        classification_report("MLP - test", data.y_test, predict_class(mlp, data.X_test))
        print("\n=== Test accuracy, all steps ===")
        for name, acc in results.items():
            print(f"  {name:<28} {acc:.3f}")
        plot_loss({"train BCE": hist["train"], "val BCE": hist["val"]}, "Step 6 - MLP training (early stopping)",
                  ylabel="BCE", filename="step06_loss.png")
        plot_decision_boundaries(
            data.X_test, data.y_test,
            {"perceptron": perceptron.predict01,
             "logistic regression": logistic_np.predict,
             "MLP 2-16-1": lambda X: predict_class(mlp, X)},
            "Step 6 - straight lines vs a learned curve", filename="step06_boundary.png")
    results["history"] = hist
    return results


if __name__ == "__main__":
    main()
