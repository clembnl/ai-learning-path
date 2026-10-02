"""Plot helpers shared by all steps (loss curves, decision boundaries).

Figures are saved to ``outputs/`` and returned so notebooks can display them.
Matplotlib uses the non-interactive Agg backend when no display is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except ImportError:
        return False


if not _running_in_ipython():
    matplotlib.use("Agg")  # scripts / tests: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
Predictor = Callable[[np.ndarray], np.ndarray]


def _save(fig: plt.Figure, filename: str | None) -> plt.Figure:
    if filename:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=110, bbox_inches="tight")
    return fig


def plot_loss(histories: dict[str, list[float]], title: str, ylabel: str = "loss",
              filename: str | None = None, logy: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, h in histories.items():
        ax.plot(h, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_regression_fit(y_true: np.ndarray, y_pred: np.ndarray, title: str,
                        filename: str | None = None) -> plt.Figure:
    """Predicted vs actual score; perfect predictions lie on the diagonal."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect")
    ax.set_xlabel("true score")
    ax.set_ylabel("predicted score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_decision_boundaries(X: np.ndarray, y: np.ndarray, predictors: dict[str, Predictor],
                             title: str, filename: str | None = None,
                             feature_names: tuple[str, str] = ("hours_studied (std)", "hours_slept (std)")) -> plt.Figure:
    """One subplot per predictor; the background colour is the predicted class."""
    n = len(predictors)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    x0 = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 200)
    x1 = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 200)
    g0, g1 = np.meshgrid(x0, x1)
    grid = np.column_stack([g0.ravel(), g1.ravel()])
    for ax, (label, predict) in zip(axes[0], predictors.items()):
        zz = np.asarray(predict(grid)).reshape(g0.shape)
        ax.contourf(g0, g1, zz, levels=[-0.5, 0.5, 1.5], colors=["#f4a582", "#92c5de"], alpha=0.6)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#ca0020", s=10, label="failed")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#0571b0", s=10, label="passed")
        ax.set_title(label)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, filename)
