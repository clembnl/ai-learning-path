from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 02_logistic_regression.ipynb by notebooks/build_notebooks.py.

_INTRO = ("md", r"""# Step 2 — Logistic regression from scratch

## What changes from step 1?

Only **two things**. That is the whole lesson of this notebook.

| | Step 1 (regression) | Step 2 (classification) |
|---|---|---|
| Question | "What score?" | "Pass or fail?" |
| Output | Any real number | A probability in $(0,1)$ |
| Model | $\hat y = Xw + b$ | $p = \sigma(Xw + b)$ ← **sigmoid added** |
| Loss | MSE | Binary cross-entropy ← **loss swapped** |
| Optimiser | Gradient descent | Gradient descent (**unchanged**) |
| Gradient | $\frac2n X^\top(\hat y - y)$ | $\frac1n X^\top(p - y)$ ← **same shape!** |

The linear core $Xw + b$ is *identical*. We wrap it in a squashing function and
measure error differently. Everything else carries over.""")

_SIGMOID = ("md", r"""## 1 — The sigmoid: turning a score into a probability

The linear part $z = Xw + b$ can output anything: $-47.3$, $0$, $+128.9$.
But a probability must live in $(0, 1)$. The **sigmoid** (logistic function) does that:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Its properties, all of which matter:

| Property | Value | Consequence |
|---|---|---|
| $\sigma(0)$ | $0.5$ | $z=0$ is the **decision boundary** — maximum uncertainty |
| $\sigma(z) \to 1$ | as $z \to +\infty$ | large positive $z$ → confident "pass" |
| $\sigma(z) \to 0$ | as $z \to -\infty$ | large negative $z$ → confident "fail" |
| $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | smooth, never negative | **differentiable** → gradient descent works |

The prediction rule: predict class 1 if $p \ge 0.5$, which is exactly the same as
saying $z \ge 0$, i.e. $w \cdot x + b \ge 0$.

That last equation defines a **straight line** in the 2D feature space — which is why
logistic regression is called a *linear* classifier despite the curved sigmoid.""")

_SIGMOID_CODE = ("code", '''from steps.step02_logistic_regression import LogisticRegressionGD, sigmoid, binary_cross_entropy
z = np.linspace(-6, 6, 200)
plt.figure(figsize=(7, 4))
plt.plot(z, sigmoid(z), lw=2)
plt.axhline(0.5, ls="--", c="gray", label="p = 0.5 (decision threshold)")
plt.axvline(0, ls="--", c="red", label="z = 0 (the boundary)")
plt.fill_between(z, 0, 1, where=(z >= 0), alpha=0.1, color="blue", label="predict PASS")
plt.fill_between(z, 0, 1, where=(z < 0), alpha=0.1, color="red", label="predict FAIL")
plt.xlabel("z = w·x + b  (the linear part)"); plt.ylabel("p = sigmoid(z)")
plt.title("The sigmoid squashes any real number into (0, 1)")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()''')

_BCE = ("md", r"""## 2 — The loss: binary cross-entropy (and why not MSE)

$$\mathcal L = -\frac1n\sum_i \big[\,y_i\log p_i + (1-y_i)\log(1-p_i)\,\big]$$

Read it one sample at a time. Only one of the two terms survives, because $y_i$ is 0 or 1:

- True label $y_i = 1$ (passed) → the loss is $-\log p_i$.
  Predict $p=0.99$ → loss $0.01$ (tiny). Predict $p=0.01$ → loss $4.6$ (huge).
- True label $y_i = 0$ (failed) → the loss is $-\log(1-p_i)$ — mirror image.

**The key property: confident mistakes are punished brutally.**
As $p \to 0$ while the truth is 1, $-\log p \to \infty$.
The model is strongly discouraged from being confidently wrong.

### Why not just use MSE on the probabilities?

1. **Vanishing gradients.** With MSE the gradient contains a factor $\sigma'(z) = p(1-p)$,
   which is ≈ 0 when $p$ is near 0 or 1. A model that is *confidently wrong* would receive
   almost no gradient and could never correct itself. With cross-entropy that factor
   cancels algebraically, leaving a clean $(p - y)$ — no vanishing.
2. **Non-convexity.** MSE combined with a sigmoid produces a loss surface with local minima.
   Cross-entropy with a sigmoid is **convex** — one global minimum, guaranteed reachable.""")

_BCE_DEMO = ("code", '''# The true label is 1; sweep the predicted probability and compare the two losses
p_vals = np.linspace(0.001, 0.999, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(p_vals, -np.log(p_vals), lw=2, label="cross-entropy: -log(p)")
axes[0].plot(p_vals, (p_vals - 1.0) ** 2, lw=2, label="MSE: (p-1)²")
axes[0].set_xlabel("predicted p   (true label = 1)"); axes[0].set_ylabel("loss")
axes[0].set_title("Cross-entropy explodes for confident mistakes")
axes[0].legend(); axes[0].grid(alpha=0.3)

# gradient w.r.t. z (the linear output), which is what actually drives learning
axes[1].plot(p_vals, np.abs(p_vals - 1.0), lw=2, label="|dBCE/dz| = |p - y|")
axes[1].plot(p_vals, np.abs(2 * (p_vals - 1.0) * p_vals * (1 - p_vals)), lw=2,
             label="|dMSE/dz|  (contains p(1-p))")
axes[1].set_xlabel("predicted p   (true label = 1)"); axes[1].set_ylabel("|gradient|")
axes[1].set_title("MSE gradient VANISHES when confidently wrong (p → 0)")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("At p = 0.01 (confidently wrong, true label = 1):")
print(f"  cross-entropy |gradient| = {abs(0.01 - 1):.4f}   <- strong correction signal")
print(f"  MSE           |gradient| = {abs(2*(0.01-1)*0.01*0.99):.4f}   <- ~zero, the model is stuck")''')

_GD = ("md", r"""## 3 — The optimiser: same gradient descent, and a beautiful surprise

$$\frac{\partial\mathcal L}{\partial w} = \frac1n X^\top(p - y),
\qquad \frac{\partial\mathcal L}{\partial b} = \frac1n\sum_i (p_i - y_i)$$

Compare with step 1: $\frac{2}{n} X^\top(\hat y - y)$.
**Identical structure** — `Xᵀ · (prediction − truth) / n`. Only a factor of 2 differs.

This is not a coincidence: the sigmoid and cross-entropy were *designed* so their
derivatives cancel. The messy $\sigma'(z) = p(1-p)$ term that any other loss would
introduce disappears, leaving the simplest possible expression.

**Practical consequence:** the code from step 1 barely changes.

```python
p = sigmoid(X @ self.w + self.b)     # forward: probabilities instead of raw scores
self.history.append(binary_cross_entropy(y, p))
grad_w = X.T @ (p - y) / n           # same shape as step 1
grad_b = (p - y).sum() / n
self.w -= self.lr * grad_w           # identical update rule
self.b -= self.lr * grad_b
```""")

_SRC = ("code", '''import inspect
print(inspect.getsource(LogisticRegressionGD.fit))''')

_TRAIN = ("code", '''model = LogisticRegressionGD(lr=0.5, epochs=1000).fit(data.X_train, data.y_train)
print("Learned parameters (standardised feature space):")
print(f"  w = {model.w.round(4)}    b = {model.b:.4f}")
print(f"  final train BCE = {model.history[-1]:.4f}")
print()
print("Both weights are positive: more studying AND more sleeping both raise")
print("the predicted probability of passing (the best linear approximation of the truth).")
print(f"w_studied ({model.w[0]:.2f}) is ~2x w_slept ({model.w[1]:.2f}): studying matters more.")
plot_loss({"train BCE": model.history}, "Step 2 — cross-entropy during training",
          ylabel="binary cross-entropy")
plt.show()''')

_TRAIN_COMMENT = ("md", """The loss falls smoothly from 0.693 to about 0.27.

Why 0.693 at the start? Because `w` and `b` begin at zero, so $z = 0$ and $p = \\sigma(0) = 0.5$
for every student — the model predicts a coin flip. And $-\\log(0.5) = 0.693$.
**That number is the baseline of a useless classifier**; any BCE below it means the model
has learned something.""")


_METRICS = ("md", r"""## 4 — Evaluation: why accuracy alone is not enough

For regression we used MSE/RMSE/R². Classification needs a different toolbox,
built from the **confusion matrix**:

|  | predicted FAIL | predicted PASS |
|---|---|---|
| **actually FAIL** | True Negative (TN) | False Positive (FP) |
| **actually PASS** | False Negative (FN) | True Positive (TP) |

From those four counts:

$$\text{accuracy} = \frac{TP + TN}{\text{total}}
\qquad
\text{precision} = \frac{TP}{TP + FP}
\qquad
\text{recall} = \frac{TP}{TP + FN}$$

**In plain words, for our problem:**
- **Accuracy** — "what fraction of all students did we classify correctly?"
- **Precision** — "of the students we *predicted would pass*, how many actually did?"
  (Low precision = we gave false hope.)
- **Recall** — "of the students who *actually passed*, how many did we catch?"
  (Low recall = we discouraged students who would have succeeded.)

**Why accuracy alone can lie:** imagine 95 % of students failed. A model that always
predicts "fail" achieves 95 % accuracy while being completely useless (precision and
recall for the pass class would both be 0). Our dataset is deliberately balanced (~50 %),
so accuracy *is* meaningful here — but always check the other two.""")

_METRICS_CODE = ("code", '''y_pred = model.predict(data.X_test)
report = classification_report("logistic regression - test", data.y_test, y_pred)
cm = report["confusion_matrix"]
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
print()
print("Reading the confusion matrix:")
print(f"  {tn:3d} correctly predicted to FAIL   (true negatives)")
print(f"  {tp:3d} correctly predicted to PASS   (true positives)")
print(f"  {fp:3d} predicted PASS but FAILED     (false positives -> false hope)")
print(f"  {fn:3d} predicted FAIL but PASSED     (false negatives -> wrongly discouraged)")''')

_BOUNDARY = ("code", '''plot_decision_boundaries(data.X_test, data.y_test,
                         {"logistic regression": model.predict},
                         "Step 2 — the decision boundary is a STRAIGHT LINE")
plt.show()''')

_BOUNDARY_COMMENT = ("md", r"""## 5 — Reading the decision boundary — and spotting the limitation

The line where the background colour changes is exactly where $w\cdot x + b = 0$,
i.e. where the model is 50/50 undecided.

$$w_1 x_1 + w_2 x_2 + b = 0
\quad\Longleftrightarrow\quad
x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}$$

That is the equation of a **line** — slope $-w_1/w_2$, intercept $-b/w_2$.
A logistic regression can rotate and translate this line, but it can **never bend it**.

**Now look for the mistakes.** In the upper-left region (little studying, lots of sleep)
there are red points sitting inside the blue "predict pass" zone. Those are the
over-sleepers: the true rule penalises them via $-2(\text{slept}-7.5)^2$, but our model
believes more sleep is monotonically good, so it over-predicts them.

**No straight line can fix this** — you would need a boundary that curves back down at
high sleep values. That is precisely what step 6 delivers.""")


_THRESHOLD = ("md", r"""## 6 — Experiment: probabilities are more useful than classes

`predict()` throws information away by collapsing $p$ to 0/1 at the 0.5 threshold.
`predict_proba()` keeps the confidence, and **you can move the threshold** to trade
precision against recall depending on which mistake costs more.

- Lower threshold (0.3) → predict "pass" more freely → **higher recall**, lower precision.
- Higher threshold (0.7) → only predict "pass" when very confident → **higher precision**, lower recall.

There is no universally correct threshold; it is a business/ethical decision.
For scholarship screening you may want high precision; for an early-warning system
for at-risk students you want high recall.""")

_THRESHOLD_CODE = ("code", '''proba = model.predict_proba(data.X_test)
print(f"{'threshold':<12}{'accuracy':<11}{'precision':<12}{'recall':<10}{'# predicted PASS'}")
print("-" * 62)
for t in (0.2, 0.3, 0.5, 0.7, 0.8):
    pred = (proba >= t).astype(float)
    p, r = precision_recall(data.y_test, pred)
    print(f"{t:<12}{accuracy(data.y_test, pred):<11.3f}{p:<12.3f}{r:<10.3f}{int(pred.sum())}")
print()
print("As the threshold rises: fewer students are predicted to pass,")
print("precision goes UP (more often right when we say 'pass'),")
print("recall goes DOWN (we miss more of the students who would have passed).")''')

_CONFIDENCE = ("code", '''# Which students is the model least sure about? Those with p closest to 0.5.
idx = np.argsort(np.abs(proba - 0.5))[:8]
print("The 8 most uncertain test students:")
print(f"{'studied(std)':<15}{'slept(std)':<14}{'p(pass)':<11}{'predicted':<12}{'actual'}")
for i in idx:
    pred_lbl = "PASS" if proba[i] >= 0.5 else "FAIL"
    act_lbl  = "PASS" if data.y_test[i] == 1 else "FAIL"
    flag = "" if pred_lbl == act_lbl else "   <- wrong"
    print(f"{data.X_test[i,0]:<15.2f}{data.X_test[i,1]:<14.2f}{proba[i]:<11.3f}{pred_lbl:<12}{act_lbl}{flag}")
print()
print("Most errors sit near the boundary, where the model itself signals low confidence.")
print("Probabilities let you say 'I am not sure' instead of guessing blindly.")''')

_SUMMARY = ("md", r"""## Summary — what you learned in step 2

| Concept | Takeaway |
|---|---|
| **Sigmoid** | $\sigma(z)=1/(1+e^{-z})$ maps any real number into $(0,1)$ so it reads as a probability. |
| **Decision boundary** | $p=0.5 \iff z=0 \iff w\cdot x + b = 0$ — always a **straight line** here. |
| **Cross-entropy** | $-\log p$ for positives; punishes confident mistakes without limit. |
| **Why not MSE** | MSE's gradient contains $p(1-p)$, which vanishes exactly when the model is confidently wrong. |
| **Gradient** | $\frac1n X^\top(p-y)$ — *same form* as linear regression; sigmoid + BCE were designed for this. |
| **BCE = 0.693** | The loss of a coin-flip model ($-\log 0.5$); anything lower means real learning. |
| **Accuracy** | Overall correctness — trustworthy only when classes are balanced (ours are). |
| **Precision** | Of predicted passes, how many were right. Guards against false hope. |
| **Recall** | Of actual passes, how many we caught. Guards against missing people. |
| **Threshold** | 0.5 is a *default*, not a law. Move it to trade precision vs recall. |
| **The limitation** | A line cannot bend around the sleep sweet spot — fixed in step 6. |

**Next:** step 3 goes back in time to the **perceptron** — the same weighted sum but with a
*hard* activation — to show why smooth, differentiable activations were such a breakthrough.""")

CELLS = [
    _INTRO,
    ("code", SETUP),
    _SIGMOID,
    _SIGMOID_CODE,
    _BCE,
    _BCE_DEMO,
    _GD,
    _SRC,
    _TRAIN,
    _TRAIN_COMMENT,
    _METRICS,
    _METRICS_CODE,
    _BOUNDARY,
    _BOUNDARY_COMMENT,
    _THRESHOLD,
    _THRESHOLD_CODE,
    _CONFIDENCE,
    _SUMMARY,
]
