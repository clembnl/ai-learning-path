"""Step 1 - Linear regression from scratch (NumPy).

WHAT YOU LEARN
    * A model is a function with parameters:  y_hat = X @ w + b
    * A loss measures how wrong the model is:  MSE = mean((y_hat - y)^2)
    * Learning = minimising the loss by following the gradient (gradient descent)
    * The learning rate controls the step size: too small = slow, too big = diverge
    * Standardising features makes gradient descent well-behaved
    * For linear regression there is an exact closed-form answer (normal equation),
      which we use to VERIFY that gradient descent converged.

TASK
    Predict the exam score (0-100) from hours studied and hours slept.
"""

from __future__ import annotations

import numpy as np

from common.data import load_split_standardized
from common.metrics import regression_report
from common.plotting import plot_loss, plot_regression_fit


class LinearRegressionGD:
    """Linear regression trained with full-batch gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 500):
        self.lr = lr
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[float] = []  # training loss per epoch

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.history = []
        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b          # 1. forward pass
            err = y_hat - y                      # 2. residuals
            self.history.append(float(np.mean(err ** 2)))
            grad_w = (2.0 / n) * X.T @ err       # 3. dL/dw
            grad_b = (2.0 / n) * err.sum()       # 3. dL/db
            self.w -= self.lr * grad_w           # 4. step against the gradient
            self.b -= self.lr * grad_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


def closed_form(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Normal equation: theta = (X^T X)^-1 X^T y, with a column of ones for the bias."""
    Xb = np.c_[np.ones(len(X)), X]
    theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return theta[1:], float(theta[0])


def learning_rate_experiment(X: np.ndarray, y: np.ndarray, lrs=(0.001, 0.1, 1.05), epochs: int = 60) -> dict[str, list[float]]:
    """Show the three regimes of the learning rate: too slow, good, diverging."""
    histories = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for lr in lrs:
            h = LinearRegressionGD(lr=lr, epochs=epochs).fit(X, y).history
            histories[f"lr={lr}"] = [min(v, 1e6) for v in h]  # cap for plotting
    return histories


def main(verbose: bool = True) -> dict:
    data, scaler = load_split_standardized()

    model = LinearRegressionGD(lr=0.1, epochs=500).fit(data.X_train, data.score_train)
    w_cf, b_cf = closed_form(data.X_train, data.score_train)

    if verbose:
        print("Gradient descent  w =", np.round(model.w, 4), " b =", round(model.b, 4))
        print("Closed form       w =", np.round(w_cf, 4), " b =", round(b_cf, 4))
        print(f"max |difference| = {max(np.abs(model.w - w_cf).max(), abs(model.b - b_cf)):.2e}")
        # weights are in standardized units; convert back to "score points per hour"
        print("Per-hour effect (original units):", np.round(model.w / scaler.std, 3),
              "(true: 5.0 for studied; slept has NO linear effect, only a curved one the model cannot see)")

    report = regression_report("linear regression (GD) - test", data.score_test, model.predict(data.X_test))

    if verbose:
        plot_loss({"train MSE": model.history}, "Step 1 - training loss", ylabel="MSE", filename="step01_loss.png")
        plot_loss(learning_rate_experiment(data.X_train, data.score_train), "Step 1 - effect of the learning rate",
                  ylabel="MSE (capped)", filename="step01_learning_rates.png", logy=True)
        plot_regression_fit(data.score_test, model.predict(data.X_test), "Step 1 - predicted vs true score (test)",
                            filename="step01_fit.png")

    report["w"], report["b"], report["w_closed_form"], report["b_closed_form"] = model.w, model.b, w_cf, b_cf
    return report


if __name__ == "__main__":
    main()
