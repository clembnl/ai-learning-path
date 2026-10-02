from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 01_linear_regression.ipynb by notebooks/build_notebooks.py.

# ─── theory cells ────────────────────────────────────────────────────────────

_INTRO = ("md", r"""# Step 1 — Linear regression from scratch

## The three ingredients of every ML algorithm

Every algorithm in this learning path is a variation on the **same triple**:

| Ingredient | Role | Here |
|---|---|---|
| **Model** | A parametric function that produces a prediction | $\hat y = Xw + b$ |
| **Loss** | A single number measuring how wrong the model is | MSE |
| **Optimiser** | A procedure that updates the parameters to reduce the loss | Gradient descent |

## Task
Predict the student's exam **score** (0–100) from `hours_studied` and `hours_slept`.
Both features are **standardised** before training (explained in section 8).""")

_MODEL = ("md", r"""## 1 — The model: $\hat y = Xw + b$

```python
y_hat = X @ self.w + self.b   # forward pass — line 41
```

- $X$ — feature matrix, shape `(n_students, 2)`.
- $w$ — weight vector, shape `(2,)`: one weight per feature.
- $b$ — bias (intercept): a single scalar.
- $\hat y$ — predicted scores, shape `(n_students,)`.

$w$ and $b$ are the **parameters** of the model.
They start at zero (a completely ignorant model that predicts 0 for everyone)
and are nudged toward better values by the optimiser on every epoch.

After training, a positive $w_1$ means "more study hours → higher predicted score".""")

_LOSS = ("md", r"""## 2 — The loss: Mean Squared Error (MSE)

$$\mathcal{L}(w, b) = \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

For each student: compute the **residual** $\hat{y}_i - y_i$ (prediction minus true score),
square it, average across all students.

The loss is a **single number** summarising "how wrong is this model right now?"
The optimiser's only job is to make this number smaller.

Why square and not absolute value?
- Squaring makes the loss **smooth and differentiable** everywhere — gradient descent requires this.
- It penalises large mistakes disproportionately: off by 20 pts is **4× worse** than off by 10 pts.

```python
err = y_hat - y                              # residuals, shape (n,)
self.history.append(float(np.mean(err**2)))  # MSE logged at this epoch — line 43
```""")

_GD = ("md", r"""## 3 — The optimiser: gradient descent

The **gradient** tells you in which direction each parameter would need to change to make the loss *worse*
(i.e. the direction of steepest ascent). Moving in the **opposite** direction decreases the loss.

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{n} X^\top (\hat{y} - y)
\qquad
\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_i (\hat{y}_i - y_i)$$

Both gradients are proportional to the residuals: if predictions are too high, both gradients push
$w$ and $b$ down; if too low, they push them up — the model self-corrects.

**Update rule** — applied once per **epoch** (one full pass over all training samples):

$$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
\qquad
b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}$$

```python
grad_w = (2.0 / n) * X.T @ err   # dL/dw — line 44
grad_b = (2.0 / n) * err.sum()   # dL/db — line 45
self.w -= self.lr * grad_w        # step against the gradient — line 46
self.b -= self.lr * grad_b        # line 47
```

$\eta$ (`self.lr` in code) is the **learning rate** — the stride length of each step.""")


# ─── closed form cells ───────────────────────────────────────────────────────

_CLOSED_FORM_THEORY = ("md", r"""## 4 — Closed-form solution and why we compare it to gradient descent

### What is the closed-form solution?

For linear regression **only**, setting the gradient of the loss to zero and solving algebraically
gives an **exact, one-shot answer** — no iterations, no learning rate needed:

$$\theta = (X_b^\top X_b)^{-1} X_b^\top y$$

where $X_b$ is $X$ with a column of 1s prepended (so the bias $b$ is treated as just another weight,
multiplying that constant 1 column). `np.linalg.solve` computes this directly.

```python
Xb = np.c_[np.ones(len(X)), X]                 # prepend 1s for the bias — line 56
theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)  # solve (XᵀX)θ = Xᵀy  — line 57
# theta[0] = b,  theta[1:] = w
```

### Why compare it to gradient descent?

Gradient descent is an **iterative, approximate** search.
Nothing in principle guarantees it reached the true minimum:
a bad learning rate, too few epochs, or a bug could all produce a wrong answer that *looks* plausible.

The closed form is an **independent, exact answer** computed in one matrix solve.
If both methods return the same `w, b`, gradient descent provably converged to the true optimum.
If they disagree, something went wrong.

**This cross-check only exists for linear regression** — for logistic regression, neural networks,
and everything else in this course there is no closed form. Building the habit of verifying
your optimiser here, on the one model where you can, is the whole point of including it.""")

_CLOSED_FORM_CODE = ("code", """from steps.step01_linear_regression import LinearRegressionGD, closed_form, learning_rate_experiment
import inspect
print(inspect.getsource(LinearRegressionGD.fit))""")

_CLOSED_FORM_RUN = ("code", """model = LinearRegressionGD(lr=0.1, epochs=500).fit(data.X_train, data.score_train)
w_cf, b_cf = closed_form(data.X_train, data.score_train)

print("Gradient descent  w =", model.w.round(4), " b =", round(model.b, 4))
print("Closed form       w =",   w_cf.round(4),  " b =", round(b_cf, 4))
print(f"Max |difference|    = {max(np.abs(model.w - w_cf).max(), abs(model.b - b_cf)):.2e}")
print()
print("=> ~5e-14 is floating-point noise, i.e. effectively zero.")
print("   Gradient descent converged to the true mathematical optimum.")""")

_TRAINING_LOSS_PLOT = ("code", """plot_loss({"train MSE": model.history}, "Step 1 — training loss over 500 epochs", ylabel="MSE")
plt.show()""")

_TRAINING_LOSS_COMMENT = ("md", """The curve drops steeply in the first ~20 epochs then flattens.
The model learns most of what it can learn quickly; after that, only tiny adjustments remain.
When the curve is flat, the loss has **converged** — more epochs would not help.""")



# ─── optimised params + evaluation cells ─────────────────────────────────────

_PARAMS = ("md", r"""## 5 — The optimised parameters: what `w` and `b` mean after training

After 500 epochs, `model.w` and `model.b` are the **final, optimised parameters** —
the specific values that minimise the training MSE.

Because the inputs were standardised, the weights live in **standardised units**.
Dividing by the feature standard deviations converts them back to "score points per original hour":""")

_PARAMS_CODE = ("code", """w_original = model.w / scaler.std
print("Weights in original feature units:")
print(f"  hours_studied : +{w_original[0]:.2f} score points per extra study hour")
print(f"  hours_slept   : +{w_original[1]:.2f} score points per extra sleep hour  (linear proxy)")
print(f"  intercept (b) :  {model.b:.2f}  (predicted score when both std-features = 0, i.e. at their means)")
print()
print("True formula: score = 35 + 5*studied - 2*(slept-7.5)^2 + noise")
print("=> w_studied is close to 5.0 (correct).")
print("   w_slept is positive but small: the model has no choice but to fit a straight line")
print("   through a curved cloud — it picks up a positive average slope on the lower half.")""")

_PARAMS_INSIGHT = ("md", r"""**Key insight:** the model assigns a *positive* linear effect to sleep, even though the true
effect is a **concave curve** centred at 7.5 h.
A linear model cannot represent a curve — it fits the best straight line through a curved cloud.
This is the limitation that the neural network in step 6 will overcome.""")

_EVAL_THEORY = ("md", r"""## 6 — Evaluation: what MSE, RMSE and R² actually tell you

After training we evaluate on the **test set** — data the model has never seen.

### MSE — the loss itself (not very human-readable)
MSE is in **squared score points**. It is the number that was minimised during training.
The squaring makes large errors dominate, which is useful for optimisation,
but `66.83 squared-score-points` is hard to interpret directly.

### RMSE — the human-readable error
$$\text{RMSE} = \sqrt{\text{MSE}} \approx 8.18 \text{ score points}$$

Back in the same units as the target: on average the model is off by about **8 points out of 100**.
This is the number you would quote to a teacher:
*"our model predicts exam scores with a typical error of ±8 points."*

Look at the scatter plot below — points cluster around the red diagonal (perfect prediction = $\hat y = y$)
with a typical spread of roughly ±8 points. That spread *is* the RMSE.

### R² — how much of the pattern is explained
$$R^2 = 1 - \frac{\sum_i(\hat y_i - y_i)^2}{\sum_i(\bar y - y_i)^2} \approx 0.786$$

- $R^2 = 1.0$ → perfect predictions (every dot on the diagonal).
- $R^2 = 0.0$ → no better than always predicting the mean score.
- $R^2 = 0.786$ → the model explains **79 % of the variance** in scores.

The remaining ~21 % cannot be captured because:
1. The sleep effect is **quadratic** (`-2·(slept−7.5)²`) and our model is linear.
2. There is genuine **random noise** ($\sigma=4$ points) baked into the data.

You will see R² stay near 0.79 through steps 2–5 (all linear). It only improves in step 6.""")

_EVAL_CODE = ("code", """y_pred = model.predict(data.X_test)
regression_report("linear regression - test", data.score_test, y_pred)
plot_regression_fit(data.score_test, y_pred, "predicted vs true score (test set)")
plt.show()""")

_EVAL_PLOT_COMMENT = ("md", r"""**Reading the scatter plot:**
- Each dot = one test student.  x = true score, y = model's predicted score.
- Red dashed diagonal = perfect prediction ($\hat y = y$).
- Dots **above** the diagonal: model over-predicted.
- Dots **below** the diagonal: model under-predicted.
- The cloud is tightest in the mid-score range and wider at the extremes —
  very high and very low scorers are harder to predict because they are on the
  tails of the sleep curve the model cannot see.""")



# ─── learning rate + standardisation + summary cells ─────────────────────────

_LR_THEORY = ("md", r"""## 7 — Experiment: the learning rate $\eta$

The learning rate is the single most important **hyper-parameter** of gradient descent.
It appears on every parameter update (lines 46–47):

```python
self.w -= self.lr * grad_w    # step size  =  lr  ×  gradient magnitude
self.b -= self.lr * grad_b
```

**Intuition — finding the bottom of a valley in fog:**
The gradient tells you *which direction is downhill* from where you currently stand.
The learning rate is your **stride length**.

- **Tiny stride (lr=0.001):** you move in the right direction but so slowly that
  you barely reach the bottom before you give up (run out of epochs).
- **Good stride (lr=0.1):** you walk confidently downhill and reach the bottom quickly.
- **Giant stride (lr=1.05):** you leap clean over the valley and land higher on the
  opposite slope — *worse* than where you started — and it compounds every step,
  sending the loss to infinity.

We test three values over 60 epochs to make all three regimes visible at once.""")

_LR_CODE = ("code", """plot_loss(
    learning_rate_experiment(data.X_train, data.score_train, lrs=(0.001, 0.1, 1.05), epochs=60),
    "Effect of the learning rate  (log-scale y-axis)",
    ylabel="MSE  (capped at 1e6)",
    logy=True,
)
plt.show()""")

_LR_READING = ("md", r"""**Reading the three curves (log-scale y-axis):**

| Curve | lr | What you see | Why |
|---|---|---|---|
| 🔵 Blue | 0.001 | Loss barely moves, stays near ~2 500 | Steps too small — 60 epochs moves almost nowhere |
| 🟠 Orange | 0.1 | Loss drops sharply in ~15 epochs, flattens ~65 | Right stride — converges quickly and accurately |
| 🟢 Green | 1.05 | Loss **explodes** to 1 000 000 by epoch 30 | Steps overshoot the minimum; each epoch is worse than the last |

**Why does overshoot compound?**
Imagine the minimum is at parameter value 5 and you are at 10.
With lr=1.05 the step lands you at −3 (you jumped past 5 and went negative).
Next step pushes you to +8 … oscillating further and further from the bottom.
The loss diverges instead of converging.

The "good" orange curve (`lr=0.1`) is what the main training above used for 500 epochs.""")

_STD_THEORY = ("md", r"""## 8 — Experiment: why standardise the features?

Standardisation ($z = (x - \mu) / \sigma$, fitted on the **training set only**) is applied before training.
It is not cosmetic — it directly affects whether gradient descent can even run.

Our two raw features live on very different scales:
- `hours_studied` ∈ [0, 10], mean ≈ 5, std ≈ 2.9
- `hours_slept`   ∈ [3, 10], mean ≈ 6.5, std ≈ 2.0

A single learning rate must serve **both** parameter directions simultaneously.
On raw features the gradient has very different magnitudes along each axis — a learning rate
that is safe for one direction can be catastrophically large for the other.
After standardisation both dimensions have the same scale (mean 0, std 1) and gradient descent
handles them with the same stride.

We use `lr=0.1` and only `60 epochs` to make the contrast sharp:""")

_STD_CODE = ("code", r"""import numpy as np
X_raw, score_raw, _ = load()
# cap the raw-feature loss to 1e8 for plotting (it diverges)
with np.errstate(over="ignore", invalid="ignore"):
    raw     = LinearRegressionGD(lr=0.1, epochs=60).fit(X_raw, score_raw)
    std_mdl = LinearRegressionGD(lr=0.1, epochs=60).fit(scaler.transform(X_raw), score_raw)

raw_hist = [min(v, 1e8) for v in raw.history]
plot_loss(
    {"raw features (capped at 1e8)": raw_hist, "standardised features": std_mdl.history},
    "Same lr=0.1, 60 epochs — raw vs standardised features",
    ylabel="MSE (log scale)", logy=True,
)
plt.show()
print(f"After 60 epochs with lr=0.1:")
print(f"  raw features  : final MSE = {raw.history[-1]:.2e}  (diverged!)")
print(f"  standardised  : final MSE = {std_mdl.history[-1]:.1f}  (converged)")""")

_STD_COMMENT = ("md", r"""With `lr=0.1` the raw-feature run **diverges** (same overshoot behaviour as `lr=1.05` in the
learning-rate experiment, because the raw gradients are much larger).
The standardised run converges cleanly with the exact same learning rate.

This is why we always standardise first:
- Train statistics (`scaler.mean_`, `scaler.std_`) are computed **only on the training set**.
- The same statistics are applied to the validation and test sets — they never "see" the test data.
- The `load_split_standardized()` helper in `common/data.py` enforces this automatically.""")

_SUMMARY = ("md", r"""## Summary — what you learned in step 1

| Concept | One-sentence takeaway |
|---|---|
| **Model** | $\hat y = Xw + b$ — a parametric function; `w` and `b` are the parameters to learn. |
| **Loss (MSE)** | One number measuring total wrongness; minimising it is the goal. |
| **RMSE** | $\sqrt{\text{MSE}}$ in the same units as the target (~8 score points here) — the human-readable error. |
| **R²** | Fraction of variance explained (0.79 here); the rest is noise + the curved sleep effect a linear model can't capture. |
| **Gradient** | Direction of steepest loss *increase*; we step in the **opposite** direction. |
| **Learning rate** | Stride length of each gradient step: too small → slow; too large → diverges. |
| **Optimised parameters** | `model.w` and `model.b` after training — the values that minimise training MSE. |
| **Closed form** | Exact algebraic solution (exists only for linear regression); used here to *verify* GD converged (difference 5e-14 ✓). |
| **Standardisation** | Makes the loss landscape symmetric so one learning rate works well for all features. |

**Next step:** keep the exact same model skeleton (`Xw + b`) but add a sigmoid activation and swap
MSE for cross-entropy loss → logistic regression classifies **pass / fail** instead of predicting a number.""")

# ─── assemble CELLS list (this is what build_notebooks.py imports) ────────────

CELLS = [
    _INTRO,
    ("code", SETUP),
    _MODEL,
    _LOSS,
    _GD,
    _CLOSED_FORM_THEORY,
    _CLOSED_FORM_CODE,
    _CLOSED_FORM_RUN,
    _TRAINING_LOSS_PLOT,
    _TRAINING_LOSS_COMMENT,
    _PARAMS,
    _PARAMS_CODE,
    _PARAMS_INSIGHT,
    _EVAL_THEORY,
    _EVAL_CODE,
    _EVAL_PLOT_COMMENT,
    _LR_THEORY,
    _LR_CODE,
    _LR_READING,
    _STD_THEORY,
    _STD_CODE,
    _STD_COMMENT,
    _SUMMARY,
]
