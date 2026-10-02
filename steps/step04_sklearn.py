"""Step 4 - The same three models with scikit-learn.

WHAT YOU LEARN
    * The scikit-learn API:  estimator.fit(X, y) -> estimator.predict(X) -> score
    * Pipelines chain preprocessing (StandardScaler) and the model, so the
      train-only statistics rule from step 1 is enforced automatically.
    * The library computes THE SAME weights as our NumPy code:
        - LinearRegression solves the normal equation (step 1 closed form)
        - LogisticRegression(C=np.inf) finds the same optimum as our GD
        - Perceptron uses the same error-driven rule as step 3
    * Libraries give you speed, numerical robustness, regularisation and many
      more algorithms - but no magic.  Now that you have written the maths
      yourself you know exactly what `.fit()` does.

TASK
    Reproduce steps 1-3 and compare weights and metrics side by side.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.data import load, split
from common.metrics import classification_report, regression_report
from steps.step01_linear_regression import LinearRegressionGD
from steps.step02_logistic_regression import LogisticRegressionGD
from steps.step03_perceptron import Perceptron as ManualPerceptron
from steps.step03_perceptron import to_pm1


def main(verbose: bool = True) -> dict:
    # Raw (non-standardized) data: the pipelines do the scaling themselves.
    X, score, y = load()
    d = split(X, score, y)

    # --- scikit-learn models -------------------------------------------------
    sk_linear = make_pipeline(StandardScaler(), LinearRegression()).fit(d.X_train, d.score_train)
    sk_sgd = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-6, random_state=0)).fit(d.X_train, d.score_train)
    sk_logistic = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=1000)).fit(d.X_train, d.y_train)
    sk_perceptron = make_pipeline(StandardScaler(), Perceptron(max_iter=50, eta0=0.01, random_state=0)).fit(d.X_train, d.y_train)

    # --- our NumPy models on identically standardized data -------------------
    scaler = StandardScaler().fit(d.X_train)
    Xtr, Xte = scaler.transform(d.X_train), scaler.transform(d.X_test)
    my_linear = LinearRegressionGD(lr=0.1, epochs=500).fit(Xtr, d.score_train)
    my_logistic = LogisticRegressionGD(lr=0.5, epochs=1000).fit(Xtr, d.y_train)
    my_perceptron = ManualPerceptron(lr=0.01, epochs=50).fit(Xtr, to_pm1(d.y_train))

    if verbose:
        print("=== Weights (standardized feature space) ===")
        print(f"linear   | numpy GD : w={np.round(my_linear.w, 4)} b={my_linear.b:.4f}")
        print(f"linear   | sklearn  : w={np.round(sk_linear[-1].coef_, 4)} b={sk_linear[-1].intercept_:.4f}")
        print(f"linear   | SGDRegr. : w={np.round(sk_sgd[-1].coef_, 4)} b={sk_sgd[-1].intercept_[0]:.4f}")
        print(f"logistic | numpy GD : w={np.round(my_logistic.w, 4)} b={my_logistic.b:.4f}")
        print(f"logistic | sklearn  : w={np.round(sk_logistic[-1].coef_[0], 4)} b={sk_logistic[-1].intercept_[0]:.4f}")
        print(f"perceptr.| numpy    : w={np.round(my_perceptron.w, 4)} b={my_perceptron.b:.4f}")
        print(f"perceptr.| sklearn  : w={np.round(sk_perceptron[-1].coef_[0], 4)} b={sk_perceptron[-1].intercept_[0]:.4f}")
        print("(perceptron weights differ: the rule depends on the visiting order and has no unique optimum)")
        print("\n=== Test metrics ===")

    reports = {
        "linear_numpy": regression_report("linear regression - numpy GD", d.score_test, my_linear.predict(Xte)),
        "linear_sklearn": regression_report("linear regression - sklearn", d.score_test, sk_linear.predict(d.X_test)),
        "linear_sgd": regression_report("SGDRegressor - sklearn", d.score_test, sk_sgd.predict(d.X_test)),
        "logistic_numpy": classification_report("logistic regression - numpy GD", d.y_test, my_logistic.predict(Xte)),
        "logistic_sklearn": classification_report("logistic regression - sklearn", d.y_test, sk_logistic.predict(d.X_test)),
        "perceptron_numpy": classification_report("perceptron - numpy", d.y_test, my_perceptron.predict01(Xte)),
        "perceptron_sklearn": classification_report("perceptron - sklearn", d.y_test, sk_perceptron.predict(d.X_test)),
    }
    reports["weight_gap_linear"] = float(np.abs(my_linear.w - sk_linear[-1].coef_).max())
    reports["weight_gap_logistic"] = float(np.abs(my_logistic.w - sk_logistic[-1].coef_[0]).max())
    if verbose:
        print(f"\nmax |w_numpy - w_sklearn|  linear={reports['weight_gap_linear']:.2e}  logistic={reports['weight_gap_logistic']:.2e}")
    return reports


if __name__ == "__main__":
    main()
