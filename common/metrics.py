"""Evaluation metrics written from scratch with NumPy.

Regression:      mse, rmse, r2
Classification:  accuracy, confusion_matrix, precision_recall
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: 1 = perfect, 0 = as good as predicting the mean."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows = true class, columns = predicted class, for classes {0, 1}."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(precision), float(recall)


def regression_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = {"model": name, "mse": mse(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
    print(f"[{name}] MSE={report['mse']:.2f}  RMSE={report['rmse']:.2f}  R2={report['r2']:.3f}")
    return report


def classification_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy(y_true, y_pred)
    prec, rec = precision_recall(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"[{name}] accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}")
    print(f"           confusion matrix (rows=true, cols=pred):\n{cm}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm}
