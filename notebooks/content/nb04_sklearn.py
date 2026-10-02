from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 04_sklearn.ipynb by notebooks/build_notebooks.py.

_INTRO = ("md", r"""# Step 4 — The same three models with scikit-learn

## Why this step exists

You have now written linear regression, logistic regression and a perceptron **by hand**.
In practice nobody does that — you call a library. This step has two goals:

1. **Learn the scikit-learn API**, which is the de-facto standard for classical ML in Python.
2. **Prove that the library does exactly what you wrote.** We will compare weights
   number by number. When they match, you know `.fit()` is not magic — it is your own
   maths, implemented more robustly.

That second point matters psychologically: from here on you can use libraries with
confidence rather than superstition.""")

_API = ("md", r"""## 1 — The scikit-learn contract

Every model ("estimator") in scikit-learn exposes the same three methods:

```python
estimator.fit(X, y)      # learn the parameters from data      -> returns self
estimator.predict(X)     # apply the learned parameters        -> returns predictions
estimator.score(X, y)    # default metric: R² for regressors, accuracy for classifiers
```

Learned parameters are stored as attributes with a **trailing underscore** —
that is the scikit-learn convention meaning "this was learned from data, not set by you":

| Our code | scikit-learn attribute |
|---|---|
| `model.w` | `estimator.coef_` |
| `model.b` | `estimator.intercept_` |

### The equivalence table

| Our from-scratch class | scikit-learn equivalent | How it solves the problem |
|---|---|---|
| `closed_form()` (step 1) | `LinearRegression` | Normal equation — the same algebra |
| `LinearRegressionGD` (step 1) | `SGDRegressor` | Iterative gradient descent, like ours |
| `LogisticRegressionGD` (step 2) | `LogisticRegression(C=np.inf)` | L-BFGS optimiser, same convex objective |
| `Perceptron` (step 3) | `Perceptron` | The same error-driven rule |

Note `C=np.inf` for logistic regression: by default scikit-learn adds **L2
regularisation**, which our version does not have. To compare apples with apples we
switch it off. (We explore what it does in section 5.)""")

_PIPELINE = ("md", r"""## 2 — Pipelines: preprocessing that cannot leak

In steps 1–3 we standardised manually and were careful to fit the scaler on the training
set only. A `Pipeline` **encodes that discipline in the object itself**:

```python
make_pipeline(StandardScaler(), LogisticRegression())
```

- On `.fit(X_train, y_train)`: the scaler computes $\mu,\sigma$ from `X_train`, transforms
  it, and passes the result to the model.
- On `.predict(X_test)`: the scaler applies the **stored training** $\mu,\sigma$ — it does
  *not* recompute them.

This makes leakage structurally impossible, which is why pipelines are the recommended
way to combine preprocessing with a model. Notice we now feed **raw** features
(hours, not z-scores) because the pipeline handles scaling internally.""")

_FIT_CODE = ("code", '''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, SGDRegressor
from common.data import split

X, score, y = load()
d = split(X, score, y)          # RAW features: the pipeline will scale them

lin = make_pipeline(StandardScaler(), LinearRegression()).fit(d.X_train, d.score_train)
sgd = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-6, random_state=0)).fit(d.X_train, d.score_train)
log = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=1000)).fit(d.X_train, d.y_train)
per = make_pipeline(StandardScaler(), Perceptron(max_iter=50, eta0=0.01, random_state=0)).fit(d.X_train, d.y_train)

print("Three lines of code replaced three notebooks of maths:")
print(f"  LinearRegression  R²       (test) = {lin.score(d.X_test, d.score_test):.3f}")
print(f"  SGDRegressor      R²       (test) = {sgd.score(d.X_test, d.score_test):.3f}")
print(f"  LogisticRegression accuracy(test) = {log.score(d.X_test, d.y_test):.3f}")
print(f"  Perceptron         accuracy(test) = {per.score(d.X_test, d.y_test):.3f}")''')

_VERIFY_INTRO = ("md", r"""## 3 — The verification: do the weights actually match?

This is the heart of the notebook. We refit **our** NumPy models on the same standardised
data and compare parameter by parameter.

What to expect:
- **Linear regression: agreement to ~1e-14.** Both solve the same convex problem with a
  unique optimum — scikit-learn via the normal equation, ours via gradient descent
  (which we already proved converges to the normal-equation answer in step 1).
- **Logistic regression: agreement to ~1e-3.** Also a unique optimum, but reached by two
  *different* iterative optimisers (L-BFGS vs plain gradient descent), each stopping at its
  own tolerance. Tiny numerical disagreement is expected and harmless.
- **Perceptron: weights will NOT match.** Expected! As step 3 showed, the perceptron has
  no unique solution — the answer depends on sample ordering. Different implementation
  details → different (equally valid) line.""")

_VERIFY_CODE = ("code", '''from steps.step01_linear_regression import LinearRegressionGD, closed_form
from steps.step02_logistic_regression import LogisticRegressionGD
from steps.step03_perceptron import Perceptron as MyPerceptron, to_pm1

# our models, on identically standardised data
scaler = StandardScaler().fit(d.X_train)
Xtr, Xte = scaler.transform(d.X_train), scaler.transform(d.X_test)
my_lin = LinearRegressionGD(lr=0.1, epochs=500).fit(Xtr, d.score_train)
my_log = LogisticRegressionGD(lr=0.5, epochs=1000).fit(Xtr, d.y_train)
my_per = MyPerceptron(lr=0.01, epochs=50).fit(Xtr, to_pm1(d.y_train))
w_cf, b_cf = closed_form(Xtr, d.score_train)

print("LINEAR REGRESSION")
print(f"  ours (gradient descent) : w={my_lin.w.round(6)}  b={my_lin.b:.6f}")
print(f"  ours (closed form)      : w={w_cf.round(6)}  b={b_cf:.6f}")
print(f"  sklearn LinearRegression: w={lin[-1].coef_.round(6)}  b={float(lin[-1].intercept_):.6f}")
print(f"  max difference (ours vs sklearn) = {np.abs(my_lin.w - lin[-1].coef_).max():.2e}   <- IDENTICAL")
print()
print("LOGISTIC REGRESSION")
print(f"  ours (gradient descent)  : w={my_log.w.round(6)}  b={my_log.b:.6f}")
print(f"  sklearn LogisticRegression: w={log[-1].coef_[0].round(6)}  b={float(log[-1].intercept_[0]):.6f}")
print(f"  max difference = {np.abs(my_log.w - log[-1].coef_[0]).max():.2e}   <- same optimum, different solver")
print()
print("PERCEPTRON")
print(f"  ours    : w={my_per.w.round(4)}  b={my_per.b:.4f}")
print(f"  sklearn : w={per[-1].coef_[0].round(4)}  b={float(per[-1].intercept_[0]):.4f}")
print("  -> DIFFERENT, and that is correct: no unique solution (see step 3).")''')

_VERIFY_COMMENT = ("md", """**This is the moment the library stops being a black box.**

Linear regression agrees to fourteen decimal places. Logistic regression agrees to three.
Those are not coincidences — they are the same equations you derived, solved by better
numerical code.

From now on, when you write `LogisticRegression().fit(X, y)`, you know precisely what is
happening inside: a sigmoid, a cross-entropy loss, and an optimiser walking downhill on the
gradient $X^\\top(p-y)/n$.""")

_FULL_REPORT = ("md", """## 4 — The full side-by-side report

`steps/step04_sklearn.py::main()` prints every model's metrics together, so you can confirm
that matching weights produce matching predictions and matching metrics.""")

_FULL_REPORT_CODE = ("code", '''from steps.step04_sklearn import main
_ = main()''')


_REG_THEORY = ("md", r"""## 5 — Experiment: regularisation, a knob we never had

Our hand-written logistic regression minimises cross-entropy alone.
scikit-learn by default minimises cross-entropy **plus a penalty on large weights**:

$$\mathcal L_{\text{sklearn}} = \underbrace{\text{BCE}}_{\text{fit the data}}
\;+\; \underbrace{\frac{1}{2C}\|w\|^2}_{\text{keep weights small}}$$

$C$ is the **inverse** regularisation strength (confusingly — it is the reciprocal of the
usual $\lambda$):

- **Small $C$ (e.g. 0.01)** → strong penalty → weights are squeezed toward zero →
  a *simpler*, flatter model. Under-fits if too extreme.
- **Large $C$ (e.g. 100)** → weak penalty → behaves like our unregularised version.

### Why would you ever want to shrink the weights?

With only 2 features and 700 samples we do not need it. But with 200 features and
50 samples, an unregularised model can drive weights to huge values to fit noise exactly —
memorising the training set and failing on new data (**over-fitting**). Regularisation is
one of the main defences, and the reason libraries enable it by default.

Watch what happens to the weight magnitudes as $C$ increases:""")

_REG_CODE = ("code", '''print(f"{'C':<10}{'w (studied, slept)':<28}{'||w||':<10}{'train acc':<12}{'test acc'}")
print("-" * 70)
for C in (0.001, 0.01, 0.1, 1.0, 100.0, 10000.0):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(d.X_train, d.y_train)
    w = m[-1].coef_[0]
    print(f"{C:<10}{str(w.round(3)):<28}{np.linalg.norm(w):<10.3f}"
          f"{m.score(d.X_train, d.y_train):<12.3f}{m.score(d.X_test, d.y_test):.3f}")
print()
print(f"ours (no penalty at all): w={my_log.w.round(3)}  ||w||={np.linalg.norm(my_log.w):.3f}")
print()
print("Small C  -> tiny weights, the model barely commits (under-fits).")
print("Large C  -> converges to our unregularised solution.")
print("On this easy 2-feature problem accuracy is stable, so regularisation is not needed here;")
print("its value appears with many features and few samples.")''')

_MORE_MODELS = ("md", """## 6 — The real payoff: dozens of models, one API

The genuine advantage of a library is not speed — it is that **every model shares the same
interface**, so trying a completely different algorithm costs one line.

Below, four models you have never implemented, applied to our problem with no new code:""")

_MORE_CODE = ("code", '''from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

candidates = {
    "LogisticRegression (linear)": LogisticRegression(C=np.inf, max_iter=1000),
    "DecisionTree (depth 4)":      DecisionTreeClassifier(max_depth=4, random_state=0),
    "RandomForest (100 trees)":    RandomForestClassifier(n_estimators=100, random_state=0),
    "SVC (RBF kernel)":            SVC(kernel="rbf", random_state=0),
    "KNeighbors (k=15)":           KNeighborsClassifier(n_neighbors=15),
}
print(f"{'model':<30}{'test accuracy':<16}{'linear?'}")
print("-" * 58)
fitted = {}
for name, est in candidates.items():
    pipe = make_pipeline(StandardScaler(), est).fit(d.X_train, d.y_train)
    fitted[name] = pipe
    is_linear = "yes" if "Logistic" in name else "no  <- can bend!"
    print(f"{name:<30}{pipe.score(d.X_test, d.y_test):<16.3f}{is_linear}")
print()
print("Every non-linear model beats logistic regression, because they can all")
print("curve around the sleep sweet spot. This is a preview of step 6:")
print("the neural network is simply the non-linear model we will build ourselves.")''')

_MORE_PLOT = ("code", '''# Draw the boundaries on standardised axes so they are comparable with the other steps.
# The pipelines expect RAW features, so we invert the standardisation before predicting.
show = ["LogisticRegression (linear)", "DecisionTree (depth 4)", "SVC (RBF kernel)"]
predictors = {
    name: (lambda P: (lambda Xs: P.predict(scaler.inverse_transform(Xs))))(fitted[name])
    for name in show
}
plot_decision_boundaries(Xte, d.y_test, predictors,
                         "Step 4 — a straight line vs boundaries that can bend")
plt.show()''')

_SUMMARY = ("md", r"""## Summary — what you learned in step 4

| Concept | Takeaway |
|---|---|
| **The API** | `fit` / `predict` / `score` — identical for every estimator in scikit-learn. |
| **Naming convention** | Learned attributes end in `_` (`coef_`, `intercept_`). |
| **Pipeline** | Chains scaler + model so preprocessing is fitted on train only — leakage becomes structurally impossible. |
| **Verification** | Linear weights match ours to ~1e-14; logistic to ~1e-3. `.fit()` is your own maths. |
| **Solver differences** | sklearn logistic uses L-BFGS, ours uses plain GD — same optimum, different path and tolerance. |
| **Perceptron mismatch** | Expected: no unique solution, so implementations legitimately differ. |
| **Regularisation (`C`)** | Adds $\|w\|^2$ to the loss to shrink weights; guards against over-fitting when features ≫ samples. |
| **The real payoff** | One line to swap in a decision tree, random forest, SVM or k-NN. |
| **Preview** | Every non-linear model here beats the linear one, foreshadowing step 6. |

### When to write it yourself vs use a library

| Situation | Choice |
|---|---|
| Learning how something works | Write it yourself (steps 1–3) |
| Production, research, anything real | Use the library |
| Debugging a suspicious library result | Write a tiny reference implementation and compare — exactly what we just did |

**Next:** step 5 rebuilds the same two models in **PyTorch**, where `loss.backward()`
computes the gradient we derived by hand — the last stop before real neural networks.""")

CELLS = [
    _INTRO,
    ("code", SETUP),
    _API,
    _PIPELINE,
    _FIT_CODE,
    _VERIFY_INTRO,
    _VERIFY_CODE,
    _VERIFY_COMMENT,
    _FULL_REPORT,
    _FULL_REPORT_CODE,
    _REG_THEORY,
    _REG_CODE,
    _MORE_MODELS,
    _MORE_CODE,
    _MORE_PLOT,
    _SUMMARY,
]
