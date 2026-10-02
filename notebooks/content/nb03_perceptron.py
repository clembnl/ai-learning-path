from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 03_perceptron.ipynb by notebooks/build_notebooks.py.

_INTRO = ("md", r"""# Step 3 — The perceptron: one artificial neuron

## Why go "backwards" to 1958?

Logistic regression (step 2) already works well. So why study an older, weaker model?

Because the perceptron is where the phrase **"artificial neuron"** comes from, and because
its *failure* explains every design choice in modern deep learning. By the end of this
notebook you will understand why neural networks use smooth activations (sigmoid, ReLU)
rather than the biologically-inspired hard threshold — and that is the doorway to
backpropagation in steps 5 and 6.

## A neuron is just two things

$$\underbrace{z = w \cdot x + b}_{\text{1. weighted sum}}
\qquad
\underbrace{\hat y = \text{activation}(z)}_{\text{2. activation}}$$

That is the **entire** definition. Every model in this course fits it:

| Model | Weighted sum | Activation |
|---|---|---|
| Linear regression (step 1) | $w\cdot x + b$ | identity (none) |
| Logistic regression (step 2) | $w\cdot x + b$ | sigmoid → smooth, $(0,1)$ |
| **Perceptron (step 3)** | $w\cdot x + b$ | **step → hard, $\{-1,+1\}$** |
| Hidden units in the MLP (step 6) | $w_j\cdot x + b_j$ | ReLU → $\max(0,z)$ |

A neural network is nothing more than **many of these units stacked**.
Understanding one unit deeply is understanding the building block of all of deep learning.""")

_MODEL = ("md", r"""## 1 — The model: a hard threshold

$$z = w\cdot x + b,
\qquad
\hat y = \operatorname{step}(z) = \begin{cases}+1 & \text{if } z\ge 0\\[2pt] -1 & \text{if } z<0\end{cases}$$

Note the label convention: the perceptron traditionally uses $\{-1, +1\}$ instead of $\{0, 1\}$.
This is not cosmetic — it makes the learning rule below elegant, because the product
$y \cdot z$ is **positive when correct** and **negative when wrong**, for both classes at once.

Our code keeps small adapters so metrics stay comparable with the other steps:
- `to_pm1(y)` converts $\{0,1\} \to \{-1,+1\}$ before training.
- `predict01(X)` converts predictions back to $\{0,1\}$ for the metric functions.

**What the perceptron cannot do:** there is no probability. The output is a bare decision.
You get no confidence, no "I am not sure", no threshold to tune — all things step 2 gave you.""")

_ACTIVATION_PLOT = ("code", '''from steps.step03_perceptron import Perceptron, to_pm1, step_function
from steps.step02_logistic_regression import LogisticRegressionGD, sigmoid

z = np.linspace(-6, 6, 400)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(z, step_function(z), lw=2, label="step (perceptron)")
axes[0].plot(z, 2 * sigmoid(z) - 1, lw=2, label="scaled sigmoid (logistic)")
axes[0].set_xlabel("z = w·x + b"); axes[0].set_ylabel("activation output")
axes[0].set_title("Hard vs smooth activation"); axes[0].legend(); axes[0].grid(alpha=0.3)

# the derivative is what gradient descent needs
axes[1].plot(z, np.zeros_like(z), lw=3, label="d/dz step = 0  EVERYWHERE")
axes[1].plot(z, 2 * sigmoid(z) * (1 - sigmoid(z)), lw=2, label="d/dz sigmoid > 0")
axes[1].set_xlabel("z"); axes[1].set_ylabel("derivative")
axes[1].set_title("THE problem: the step function has no usable gradient")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()''')

_WHY_NO_GD = ("md", r"""## 2 — Why gradient descent is impossible here

Look at the right-hand plot. The derivative of the step function is:
- **exactly 0** everywhere except at $z=0$,
- **undefined** (infinite) at $z=0$ itself.

Gradient descent updates parameters with $w \leftarrow w - \eta \frac{\partial \mathcal L}{\partial w}$.
By the chain rule that gradient contains the activation's derivative as a factor:

$$\frac{\partial \mathcal L}{\partial w}
= \underbrace{\frac{\partial \mathcal L}{\partial \hat y}}_{\text{fine}}
\cdot \underbrace{\frac{\partial \hat y}{\partial z}}_{= \; 0 \; !!}
\cdot \underbrace{\frac{\partial z}{\partial w}}_{= \; x}
\;=\; 0$$

**Every gradient is zero, so no parameter would ever move.** Learning is impossible.

This is *the* historical obstacle. Rosenblatt therefore invented a different,
non-gradient rule (next section). It took until the 1980s and the popularisation of
**backpropagation with smooth activations** for multi-layer networks to become trainable —
exactly what you will use in steps 5 and 6.""")

_RULE = ("md", r"""## 3 — The perceptron learning rule

Instead of a loss gradient, Rosenblatt's rule is **error-driven** and works one sample at a time:

> For each sample $(x_i, y_i)$: if it is **misclassified** — that is, if $y_i\,(w\cdot x_i+b) \le 0$ —
> then nudge the parameters toward that sample:
> $$w \leftarrow w + \eta\,y_i\,x_i, \qquad b \leftarrow b + \eta\,y_i$$
> If the sample is already correct, **do nothing**.

### Why this works, intuitively

Suppose $y_i = +1$ (should pass) but the model said $-1$ (predicted fail), so $z < 0$.
The update adds $\eta x_i$ to $w$. The new prediction for that same sample becomes:

$$z_{\text{new}} = (w + \eta x_i)\cdot x_i + b + \eta = z_{\text{old}} + \eta(\|x_i\|^2 + 1)$$

Since $\|x_i\|^2 + 1 > 0$ always, $z$ moved **up** — toward the correct side.
The rule geometrically rotates the boundary toward the misclassified point.

### The three critical differences from step 2

| | Logistic regression (step 2) | Perceptron (step 3) |
|---|---|---|
| Updates on | **every** sample, always | only **misclassified** samples |
| Driven by | the gradient of a smooth loss | a heuristic correction rule |
| Guarantee | converges to the unique global optimum | converges **only if data is linearly separable** |

That last row is what we will now observe failing.""")

_SRC = ("code", '''import inspect
print(inspect.getsource(Perceptron.fit))''')

_TRAIN = ("code", '''perceptron = Perceptron(lr=0.01, epochs=50).fit(data.X_train, to_pm1(data.y_train))
print(f"w = {perceptron.w.round(4)}   b = {perceptron.b:.4f}")
print(f"\\nmistakes per epoch (out of {len(data.X_train)} training samples):")
print(perceptron.history)
print(f"\\nfirst epoch: {perceptron.history[0]} errors")
print(f"last  epoch: {perceptron.history[-1]} errors")
print(f"best  epoch: {min(perceptron.history)} errors")
print("=> it NEVER reaches 0 and does not steadily improve: it oscillates.")
plot_loss({"misclassified (train)": perceptron.history},
          "Step 3 — perceptron mistakes per epoch (never converges)", ylabel="# errors")
plt.show()''')

_TRAIN_COMMENT = ("md", r"""## 4 — Reading the error curve: a non-convergent algorithm

Compare this jagged, flat-trending curve with the smooth monotonic decrease of steps 1 and 2.
**The perceptron does not converge.** Why?

The **perceptron convergence theorem** guarantees termination in a finite number of steps
*only if the classes are linearly separable* — i.e. if some straight line classifies every
single training point correctly.

Our data is **not** separable, for two independent reasons:
1. **Noise** ($\sigma = 4$ points): some students near the 50-point cut-off passed or failed
   by luck. No boundary of any shape can classify those correctly.
2. **The sleep sweet spot**: the true boundary is curved, so even without noise a *line*
   would misclassify the over-sleepers.

With non-separable data the rule keeps finding misclassified points forever, and each
correction *un-corrects* other points. The boundary wanders instead of settling.

There is no notion of "good enough" in the rule — unlike a loss function, which quantifies
*how* wrong we are and can plateau at its minimum. The perceptron only asks
"is this point wrong? then move" — a binary signal with no sense of magnitude.""")

_COMPARE = ("code", '''logistic = LogisticRegressionGD(lr=0.5, epochs=1000).fit(data.X_train, data.y_train)
classification_report("perceptron - test", data.y_test, perceptron.predict01(data.X_test))
print()
classification_report("logistic regression (step 2) - test", data.y_test, logistic.predict(data.X_test))
print()
print("Both are LINEAR classifiers, so both draw a straight line.")
print("Logistic regression wins by ~2 points because it optimises a principled loss")
print("over ALL samples, while the perceptron reacts only to individual mistakes.")
plot_decision_boundaries(data.X_test, data.y_test,
                         {"perceptron (error-driven rule)": perceptron.predict01,
                          "logistic regression (gradient descent)": logistic.predict},
                         "Two linear neurons — same model family, different training rules")
plt.show()''')

_COMPARE_COMMENT = ("md", """**Both boundaries are straight lines** — as they must be, since both models compute
`w·x + b` and threshold it. The difference is only in *where* the line ended up:

- The **logistic** line sits at the position that minimises average cross-entropy over all
  700 training students. It is the statistically optimal line.
- The **perceptron** line is wherever the last few corrections happened to leave it.
  It is a valid separator attempt, but nothing guarantees it is the *best* line.

This is the practical value of having a **loss function**: it defines what "best" means,
so the optimiser has a target rather than just a reflex.""")


_SEED_EXP = ("md", """## 5 — Experiment: the result depends on the sample order

Because updates happen per-sample and only on mistakes, the final boundary depends on
**the order in which samples are visited**. Change the shuffle seed and you get a
different model — sometimes noticeably better or worse.

Contrast with logistic regression: it is fully deterministic and always lands on the same
unique optimum, regardless of ordering. The perceptron has **no unique solution**.""")

_SEED_CODE = ("code", '''print(f"{'seed':<8}{'w':<26}{'b':<10}{'test accuracy'}")
print("-" * 58)
accs = []
for seed in range(6):
    p = Perceptron(lr=0.01, epochs=50, seed=seed).fit(data.X_train, to_pm1(data.y_train))
    acc = accuracy(data.y_test, p.predict01(data.X_test))
    accs.append(acc)
    print(f"{seed:<8}{str(p.w.round(3)):<26}{p.b:<10.3f}{acc:.3f}")
print()
print(f"accuracy spread across seeds: {min(accs):.3f} to {max(accs):.3f}  (range {max(accs)-min(accs):.3f})")
print(f"logistic regression, any seed: {accuracy(data.y_test, logistic.predict(data.X_test)):.3f}  (deterministic)")''')

_SEP_EXP = ("md", """## 6 — Experiment: prove the theorem by making the data separable

The claim was: *the perceptron reaches zero errors if and only if the data is linearly
separable*. Let's test it by building an easy, clearly separable toy set and re-running
the **identical** algorithm.""")

_SEP_CODE = ("code", '''# Two well-spaced blobs -> linearly separable by construction
rng = np.random.default_rng(0)
n = 200
blob_fail = rng.normal([-2.0, -2.0], 0.6, size=(n // 2, 2))
blob_pass = rng.normal([+2.0, +2.0], 0.6, size=(n // 2, 2))
X_sep = np.vstack([blob_fail, blob_pass])
y_sep = np.r_[-np.ones(n // 2), np.ones(n // 2)]

p_sep = Perceptron(lr=0.01, epochs=50).fit(X_sep, y_sep)
print("SEPARABLE blobs:")
print(f"  errors per epoch: {p_sep.history}")
print(f"  -> reached 0 errors and STOPPED EARLY after {len(p_sep.history)} epochs.")
print()
print("OUR student data (not separable):")
print(f"  ran all {len(perceptron.history)} epochs, best was {min(perceptron.history)} errors, never 0.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(X_sep[y_sep == -1, 0], X_sep[y_sep == -1, 1], s=12, c="#ca0020", label="class -1")
axes[0].scatter(X_sep[y_sep == 1, 0], X_sep[y_sep == 1, 1], s=12, c="#0571b0", label="class +1")
xx = np.linspace(-4, 4, 10)
axes[0].plot(xx, -(p_sep.w[0] * xx + p_sep.b) / p_sep.w[1], "k-", lw=2, label="perceptron boundary")
axes[0].set_title("Separable -> converges to 0 errors")
axes[0].legend(fontsize=8); axes[0].set_ylim(-4.5, 4.5)
axes[1].plot(p_sep.history, "o-", label="separable blobs")
axes[1].plot(perceptron.history, "s-", alpha=0.7, ms=3, label="our student data")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("# errors")
axes[1].set_title("Convergence vs perpetual oscillation")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()''')

_SEP_COMMENT = ("md", """The theorem holds exactly as advertised: on separable blobs the identical algorithm drives
the error count to **zero** within a handful of epochs and terminates early (thanks to the
`if errors == 0: break` line in `fit`). On our realistic data it oscillates forever.

This is the honest picture of the perceptron: a historically pivotal idea with a hard
mathematical limit.""")

_SUMMARY = ("md", r"""## Summary — what you learned in step 3

| Concept | Takeaway |
|---|---|
| **A neuron** | Weighted sum + activation. That is *all*. Every model here fits this template. |
| **Perceptron** | The neuron with a **hard step** activation and $\{-1,+1\}$ labels. |
| **Learning rule** | $w \leftarrow w + \eta y x$ on **mistakes only** — error-driven, not gradient-driven. |
| **Why not gradient descent** | $\frac{d}{dz}\text{step}(z) = 0$ everywhere → every gradient is 0 → nothing learns. |
| **Convergence theorem** | Zero errors guaranteed **iff** the data is linearly separable (proven on blobs above). |
| **Our data is not separable** | Noise + the curved sleep effect → the error count oscillates forever. |
| **No unique solution** | The final boundary depends on the sample visiting order (seed experiment). |
| **No probabilities** | A bare decision — no confidence, no tunable threshold. |
| **Why it matters** | Its failure motivated smooth activations + backpropagation, the foundation of steps 5–6. |

### The historical arc, in one line

> Perceptron (hard activation, no gradients) → logistic regression (smooth activation,
> gradient descent, still one linear unit) → **multi-layer network** (many smooth units
> stacked, trained by backpropagation) = step 6.

**Next:** step 4 puts the hand-written maths down and picks up **scikit-learn**, verifying
that the library reproduces our NumPy results exactly.""")

CELLS = [
    _INTRO,
    ("code", SETUP),
    _MODEL,
    _ACTIVATION_PLOT,
    _WHY_NO_GD,
    _RULE,
    _SRC,
    _TRAIN,
    _TRAIN_COMMENT,
    _COMPARE,
    _COMPARE_COMMENT,
    _SEED_EXP,
    _SEED_CODE,
    _SEP_EXP,
    _SEP_CODE,
    _SEP_COMMENT,
    _SUMMARY,
]
