"""Step 3 - The perceptron (Rosenblatt, 1958): a single artificial neuron.

WHAT YOU LEARN
    * A neuron = weighted sum of inputs + an activation function.
        z = w . x + b          (exactly the same linear core as steps 1 and 2)
        y_hat = step(z)        (+1 if z >= 0 else -1)
    * The perceptron learning rule updates the weights ONLY on mistakes:
        w <- w + lr * y * x ,  b <- b + lr * y
      It is an error-driven rule, not the gradient of a loss.
    * Why?  The step function has zero gradient everywhere (and is undefined at
      0), so gradient descent cannot be used.  This is THE limitation that
      motivated smooth activations (sigmoid, ReLU) and backpropagation.
    * Convergence is guaranteed only if the data is linearly separable.  Ours is
      not (noise + sleep sweet spot), so the number of errors oscillates instead
      of reaching zero.  Logistic regression, which optimises a smooth loss,
      is the "trainable" version of the same neuron.

TASK
    Same as step 2: classify pass / fail.  Labels are encoded as -1 / +1.
"""

from __future__ import annotations

import numpy as np

from common.data import load_split_standardized
from common.metrics import classification_report
from common.plotting import plot_decision_boundaries, plot_loss
from steps.step02_logistic_regression import LogisticRegressionGD


def step_function(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0, -1.0)


class Perceptron:
    """Rosenblatt perceptron with online (sample by sample) updates."""

    def __init__(self, lr: float = 0.01, epochs: int = 50, seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[int] = []  # misclassified samples per epoch

    def fit(self, X: np.ndarray, y_pm: np.ndarray) -> "Perceptron":
        """y_pm must be in {-1, +1}."""
        if not set(np.unique(y_pm)) <= {-1.0, 1.0}:
            raise ValueError("Perceptron expects labels in {-1, +1}")
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.history = []
        for _ in range(self.epochs):
            errors = 0
            for i in rng.permutation(len(X)):  # visit samples in random order
                xi, yi = X[i], y_pm[i]
                if yi * (xi @ self.w + self.b) <= 0:  # mistake -> nudge the boundary toward xi
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            self.history.append(errors)
            if errors == 0:  # linearly separable and converged
                break
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return labels in {-1, +1}."""
        return step_function(self.decision_function(X))

    def predict01(self, X: np.ndarray) -> np.ndarray:
        """Return labels in {0, 1} for comparison with the other steps."""
        return (self.predict(X) > 0).astype(float)


def to_pm1(y01: np.ndarray) -> np.ndarray:
    return np.where(y01 > 0.5, 1.0, -1.0)


def main(verbose: bool = True) -> dict:
    data, _ = load_split_standardized()
    perceptron = Perceptron(lr=0.01, epochs=50).fit(data.X_train, to_pm1(data.y_train))
    logistic = LogisticRegressionGD(lr=0.5, epochs=1000).fit(data.X_train, data.y_train)

    if verbose:
        print("perceptron w =", np.round(perceptron.w, 4), " b =", round(perceptron.b, 4))
        print("errors per epoch:", perceptron.history)
        print("-> the error count never reaches 0: the data is not linearly separable")
    report = classification_report("perceptron - test", data.y_test, perceptron.predict01(data.X_test))
    if verbose:
        classification_report("logistic regression (step 2) - test", data.y_test, logistic.predict(data.X_test))
        plot_loss({"misclassified (train)": perceptron.history}, "Step 3 - perceptron mistakes per epoch",
                  ylabel="# errors", filename="step03_errors.png")
        plot_decision_boundaries(data.X_test, data.y_test,
                                 {"perceptron": perceptron.predict01, "logistic regression": logistic.predict},
                                 "Step 3 - two linear neurons, two training rules", filename="step03_boundary.png")

    report["w"], report["b"], report["errors_per_epoch"] = perceptron.w, perceptron.b, perceptron.history
    return report


if __name__ == "__main__":
    main()
