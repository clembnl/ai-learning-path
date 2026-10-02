"""Step 5 - The same models in PyTorch (autograd replaces our hand-written gradients).

WHAT YOU LEARN
    * Tensors, Dataset / DataLoader (mini-batches), nn.Module, loss, optimizer.
    * The canonical training loop:
          opt.zero_grad()  ->  loss = loss_fn(model(x), y)  ->  loss.backward()  ->  opt.step()
      `backward()` computes exactly the  X^T (p - y) / n  we derived by hand.
    * nn.Linear(2, 1) IS linear regression (with MSELoss) and IS logistic
      regression (with BCEWithLogitsLoss, which applies the sigmoid internally
      in a numerically stable way).
    * Mini-batch SGD: the loss curve is noisier than full-batch GD but each
      epoch does several updates.
    * model.eval() / torch.no_grad() for inference.

TASK
    Steps 1 and 2 again, in PyTorch.  Weights should match the NumPy versions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.data import Split, load_split_standardized
from common.metrics import classification_report, regression_report
from common.plotting import plot_decision_boundaries, plot_loss

torch.set_num_threads(2)  # tiny problem: more threads only add overhead


def to_tensor(a: np.ndarray) -> torch.Tensor:
    t = torch.tensor(a, dtype=torch.float32)
    return t.unsqueeze(1) if t.ndim == 1 else t  # targets become column vectors (n, 1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(to_tensor(X), to_tensor(y)), batch_size=batch_size, shuffle=shuffle)


def train(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
          epochs: int, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> dict[str, list[float]]:
    """Generic training loop reused verbatim in step 6."""
    history: dict[str, list[float]] = {"train": [], "val": []}
    Xv = to_tensor(X_val) if X_val is not None else None
    yv = to_tensor(y_val) if y_val is not None else None
    for _ in range(epochs):
        model.train()
        running, count = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()           # 1. forget previous gradients
            loss = loss_fn(model(xb), yb)   # 2. forward + loss
            loss.backward()                 # 3. autograd: dloss/dparam for every parameter
            optimizer.step()                # 4. param -= lr * grad   (for SGD)
            running += loss.item() * len(xb)
            count += len(xb)
        history["train"].append(running / count)
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                history["val"].append(loss_fn(model(Xv), yv).item())
    return history


@torch.no_grad()
def predict_regression(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(to_tensor(X)).squeeze(1).numpy()


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(to_tensor(X))).squeeze(1).numpy()


def predict_class(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return (predict_proba(model, X) >= 0.5).astype(float)


def build_linear_regression(data: Split, epochs: int = 100, lr: float = 0.05, seed: int = 0) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    model = nn.Linear(2, 1)
    hist = train(model, make_loader(data.X_train, data.score_train), nn.MSELoss(),
                 torch.optim.SGD(model.parameters(), lr=lr), epochs, data.X_val, data.score_val)
    return model, hist


def build_logistic_regression(data: Split, epochs: int = 200, lr: float = 0.1, seed: int = 0) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    model = nn.Linear(2, 1)  # logits; the sigmoid lives inside BCEWithLogitsLoss
    hist = train(model, make_loader(data.X_train, data.y_train), nn.BCEWithLogitsLoss(),
                 torch.optim.SGD(model.parameters(), lr=lr), epochs, data.X_val, data.y_val)
    return model, hist


def main(verbose: bool = True) -> dict:
    data, _ = load_split_standardized()

    lin, lin_hist = build_linear_regression(data)
    log, log_hist = build_logistic_regression(data)

    if verbose:
        print("torch linear   w =", np.round(lin.weight.detach().numpy()[0], 4), " b =", round(lin.bias.item(), 4))
        print("torch logistic w =", np.round(log.weight.detach().numpy()[0], 4), " b =", round(log.bias.item(), 4))
        print("(compare with the NumPy weights printed in steps 1 and 2)")
    reports = {
        "linear": regression_report("linear regression - torch", data.score_test, predict_regression(lin, data.X_test)),
        "logistic": classification_report("logistic regression - torch", data.y_test, predict_class(log, data.X_test)),
    }
    reports["linear"]["w"] = lin.weight.detach().numpy()[0]
    reports["logistic"]["w"] = log.weight.detach().numpy()[0]

    if verbose:
        plot_loss({"train MSE": lin_hist["train"], "val MSE": lin_hist["val"]}, "Step 5 - torch linear regression",
                  ylabel="MSE", filename="step05_linear_loss.png")
        plot_loss({"train BCE": log_hist["train"], "val BCE": log_hist["val"]}, "Step 5 - torch logistic regression",
                  ylabel="BCE", filename="step05_logistic_loss.png")
        plot_decision_boundaries(data.X_test, data.y_test, {"torch nn.Linear(2,1)": lambda X: predict_class(log, X)},
                                 "Step 5 - same straight boundary as step 2", filename="step05_boundary.png")
    return reports


if __name__ == "__main__":
    main()
