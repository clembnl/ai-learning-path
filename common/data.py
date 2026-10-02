"""Dataset generation, loading, splitting and preprocessing.

The whole learning path uses ONE problem:

    Given how many hours a student studied and slept, predict
      * the exam score        (regression target, 0-100)
      * whether they passed   (classification target, score >= 50)

The data is synthetic so the *true* generating process is known and every
model can be checked against it.  A deliberately non-linear "sleep sweet
spot" is included so that a straight decision boundary is good but not
perfect, which gives the neural network (step 6) a reason to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "students.csv"
FEATURES = ["hours_studied", "hours_slept"]
PASS_THRESHOLD = 50.0

# True coefficients of the generating process (kept explicit on purpose).
#   score = 35 + 5 * studied - 2 * (slept - 7.5)^2 + noise
# Studying always helps (linear); sleeping helps up to ~7.5h then hurts (quadratic).
TRUE_INTERCEPT = 35.0
TRUE_W_STUDIED = 5.0
TRUE_W_SLEPT = 0.0
TRUE_SLEEP_SWEET_SPOT = 7.5
TRUE_SLEEP_CURVATURE = 2.0
NOISE_STD = 4.0


def generate(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Return an array of shape (n, 4): studied, slept, score, passed."""
    rng = np.random.default_rng(seed)
    studied = rng.uniform(0.0, 10.0, n_samples)
    slept = rng.uniform(3.0, 10.0, n_samples)
    score = (
        TRUE_INTERCEPT
        + TRUE_W_STUDIED * studied
        + TRUE_W_SLEPT * slept
        - TRUE_SLEEP_CURVATURE * (slept - TRUE_SLEEP_SWEET_SPOT) ** 2
        + rng.normal(0.0, NOISE_STD, n_samples)
    )
    score = np.clip(score, 0.0, 100.0)
    passed = (score >= PASS_THRESHOLD).astype(float)
    return np.column_stack([studied, slept, score, passed])


def save_csv(path: Path = CSV_PATH, **kwargs) -> Path:
    data = generate(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(FEATURES + ["score", "passed"]),
        comments="",
        fmt="%.4f",
    )
    return path


def load(path: Path = CSV_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, 2), score (n,), passed (n,) as float arrays."""
    if not path.exists():
        save_csv(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, :2], data[:, 2], data[:, 3]


@dataclass
class Split:
    """Train / validation / test partitions of the dataset."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    score_train: np.ndarray
    score_val: np.ndarray
    score_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def split(
    X: np.ndarray,
    score: np.ndarray,
    y: np.ndarray,
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 0,
) -> Split:
    """Shuffle once, then cut into train / val / test (default 70/15/15)."""
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("train and val fractions must satisfy 0 < train, train + val < 1")
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(train * n)
    n_val = int(val * n)
    tr, va, te = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    return Split(
        X[tr], X[va], X[te],
        score[tr], score[va], score[te],
        y[tr], y[va], y[te],
    )


@dataclass
class Standardizer:
    """z = (x - mean) / std, with statistics computed on the TRAIN set only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        std = X.std(axis=0)
        std[std == 0] = 1.0  # guard against constant columns
        return cls(mean=X.mean(axis=0), std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def load_split_standardized(seed: int = 0) -> tuple[Split, Standardizer]:
    """One-liner used by every step: load, split, standardize with train stats."""
    X, score, y = load()
    s = split(X, score, y, seed=seed)
    scaler = Standardizer.fit(s.X_train)
    s.X_train = scaler.transform(s.X_train)
    s.X_val = scaler.transform(s.X_val)
    s.X_test = scaler.transform(s.X_test)
    return s, scaler


if __name__ == "__main__":
    p = save_csv()
    X, score, y = load(p)
    print(f"saved {p} with {len(X)} rows")
    print(f"pass rate: {y.mean():.2%}   score mean/std: {score.mean():.1f}/{score.std():.1f}")
