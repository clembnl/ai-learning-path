"""Step 2 - Logistic regression from scratch (NumPy).

WHAT YOU LEARN
    * Same linear core as step 1 (z = X @ w + b) but the output goes through the
      sigmoid  p = 1 / (1 + exp(-z))  so it can be read as P(passed = 1 | x).
    * Classification needs a different loss: binary cross-entropy (log loss).
      MSE on probabilities gives tiny gradients when the model is confidently
      wrong; cross-entropy does not.
    * Beautiful fact: the gradient is again  X^T (p - y) / n  - the same form as
      step 1.  Only the activation and the loss changed.
    * The decision boundary  w . x + b = 0  is a STRAIGHT LINE in feature space.
      It cannot bend around the "sleep sweet spot" - remember this for step 6.

TASK
    Predict whether the student passes (score >= 50) from hours studied / slept.
"""

from __future__ import annotations

import numpy as np

from common.data import load_split_standardized
from common.metrics import classification_report
from common.plotting import plot_decision_boundaries, plot_loss


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)  # avoid log(0)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionGD:
    """Logistic regression trained with full-batch gradient descent."""

    def __init__(self, lr: float = 0.5, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.history = []
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)           # forward: probabilities
            self.history.append(binary_cross_entropy(y, p))
            grad_w = X.T @ (p - y) / n                 # dBCE/dw  (same shape as step 1!)
            grad_b = (p - y).sum() / n
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(float)


def main(verbose: bool = True) -> dict:
    data, _ = load_split_standardized()
    model = LogisticRegressionGD(lr=0.5, epochs=1000).fit(data.X_train, data.y_train)

    if verbose:
        print("w =", np.round(model.w, 4), " b =", round(model.b, 4))
        print(f"final train BCE = {model.history[-1]:.4f}")
    report = classification_report("logistic regression (GD) - test", data.y_test, model.predict(data.X_test))

    if verbose:
        plot_loss({"train BCE": model.history}, "Step 2 - training loss", ylabel="binary cross-entropy",
                  filename="step02_loss.png")
        plot_decision_boundaries(data.X_test, data.y_test, {"logistic regression": model.predict},
                                 "Step 2 - decision boundary is a straight line", filename="step02_boundary.png")

    report["w"], report["b"], report["model"] = model.w, model.b, "logistic regression (GD)"
    return report


if __name__ == "__main__":
    main()
