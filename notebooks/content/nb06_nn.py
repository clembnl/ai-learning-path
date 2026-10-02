from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 06_neural_network.ipynb by notebooks/build_notebooks.py.

_INTRO = ("md", r"""# Step 6 — A small neural network

## The payoff

Five steps, five models, and **every single one drew a straight line**:

| Step | Model | Boundary | Test accuracy |
|---|---|---|---|
| 2 | Logistic regression (NumPy) | straight | 0.893 |
| 3 | Perceptron (NumPy) | straight | 0.873 |
| 4 | Logistic regression (sklearn) | straight | 0.893 |
| 5 | `nn.Linear(2,1)` (PyTorch) | straight | 0.893 |

But we know from notebook 00 that the true rule contains
$-2\,(\text{slept}-7.5)^2$ — a **curve**. Every linear model has been fighting a
battle it cannot win.

In this notebook we change **one line of code** and the boundary finally bends.

## What changes from step 5?

```python
# Step 5:
model = nn.Linear(2, 1)

# Step 6:
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
```

The binary cross-entropy loss and core training loop are reused. This step
also switches to Adam and introduces validation-based early stopping.
This is the reward for building the PyTorch foundation in step 5.""")

_ARCH = ("md", r"""## 1 — The architecture: 2 → 16 → 1

$$\underbrace{x \in \mathbb{R}^2}_{\text{input}}
\;\xrightarrow{\;W_1, b_1\;}\;
\underbrace{h = \text{ReLU}(W_1 x + b_1) \in \mathbb{R}^{16}}_{\text{hidden layer: 16 learned features}}
\;\xrightarrow{\;W_2, b_2\;}\;
\underbrace{z = W_2 h + b_2 \in \mathbb{R}}_{\text{output logit}}$$

### Read it as "logistic regression on learned features"

The **output layer** is *exactly* the logistic regression of step 2 — a weighted sum
followed (inside the loss) by a sigmoid. The only difference is what it receives as input:

- Step 2: the raw features `(studied, slept)` — 2 numbers you provided.
- Step 6: `h`, **16 numbers the network invented for itself** during training.

Each hidden unit $j$ computes $\text{ReLU}(w_j \cdot x + b_j)$ — a step-3-style neuron with
a continuous, piecewise-linear activation. Each one becomes a detector for some region of the input space;
together they give the output layer a much richer description to work with.

### Parameter count

| Layer | Weights | Biases | Total |
|---|---|---|---|
| `Linear(2, 16)` | $2\times16=32$ | 16 | 48 |
| `Linear(16, 1)` | $16\times1=16$ | 1 | 17 |
| **Network** | | | **65** |
| *Logistic regression, for comparison* | 2 | 1 | *3* |

65 parameters for 700 training samples — roughly 1 parameter per 11 samples. Enough
flexibility to capture the curve, few enough to avoid memorising the noise (we will verify
this with the validation curve).""")

_RELU = ("md", r"""## 2 — Why the non-linearity is essential

$$\text{ReLU}(z) = \max(0, z)$$

**Without an activation between the layers, depth would be worthless.** Two stacked
linear layers collapse algebraically into a single linear layer:

$$W_2(W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{\text{just another matrix}} x
+ \underbrace{(W_2 b_1 + b_2)}_{\text{just another vector}}
= W' x + b'$$

That is *still a straight line*. Stacking 100 linear layers would still give a straight line.
**The activation is the only reason depth buys expressiveness.**

### Why ReLU rather than the sigmoid from step 2?

| | Sigmoid | ReLU |
|---|---|---|
| Formula | $1/(1+e^{-z})$ | $\max(0, z)$ |
| Derivative | $\le 0.25$, →0 at both ends | exactly 1 when $z>0$ |
| Deep networks | gradients **vanish** through layers | gradient passes through undamped |
| Cost | exponential | a single comparison |

With sigmoids, each layer multiplies the gradient by at most 0.25, so after 10 layers the
activation-derivative contribution is $\le 0.25^{10} \approx 10^{-6}$. Weight matrices also affect the full gradient; this illustrates why gradients can shrink (the **vanishing gradient
problem**). ReLU's derivative is exactly 1 on the positive side, so gradients flow.
This single change is a large part of why deep learning became practical after 2010.

Note the contrast with step 3: the perceptron's step function had derivative **0** — useless.
ReLU is also "kinked", but its derivative is 0 only for negative inputs, and 1 elsewhere.
That is enough for gradient descent to work.""")

_RELU_CODE = ("code", '''import torch
from torch import nn
from steps.step06_neural_network import build_mlp, train_with_early_stopping, add_sleep_squared
from steps.step05_pytorch import predict_class, predict_proba, train, make_loader

z = np.linspace(-3, 3, 300)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(z, np.maximum(0, z), lw=2, label="ReLU(z) = max(0, z)")
axes[0].plot(z, 1 / (1 + np.exp(-z)), lw=2, label="sigmoid(z)")
axes[0].set_title("Activations"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel("z")
axes[1].plot(z, (z > 0).astype(float), lw=2, label="d/dz ReLU = 1 if z>0")
s = 1 / (1 + np.exp(-z))
axes[1].plot(z, s * (1 - s), lw=2, label="d/dz sigmoid <= 0.25")
axes[1].set_title("Derivatives — why ReLU keeps gradients alive")
axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xlabel("z")
plt.tight_layout(); plt.show()
print("Max sigmoid derivative:", round(float((s*(1-s)).max()), 4), "-> gradients shrink through layers")
print("ReLU derivative for z>0: 1.0                  -> gradients pass through unchanged")''')

_BACKPROP = ("md", r"""## 3 — Backpropagation: the chain rule, layer by layer

In step 1 you derived $\partial\mathcal L/\partial w$ by hand for one layer.
With two layers the output weights are easy, but the **hidden** weights $W_1$ affect the
loss only *indirectly* — through $h$, then through the output layer. The chain rule handles it:

$$\frac{\partial \mathcal L}{\partial W_1}
= \frac{\partial \mathcal L}{\partial z}
\cdot \frac{\partial z}{\partial h}
\cdot \underbrace{\frac{\partial h}{\partial (W_1x+b_1)}}_{\text{ReLU}' = 0 \text{ or } 1}
\cdot \frac{\partial (W_1x+b_1)}{\partial W_1}$$

**Backpropagation** = evaluating this product from the loss *backwards* to the first layer,
reusing intermediate results instead of recomputing them. It is what makes training deep
networks tractable rather than exponentially expensive.

**You do not implement it.** `loss.backward()` does exactly this — the same call you already
used in step 5 on a one-layer model. That is the point of autograd: the training loop does
not change when the model gets deeper.""")

_TOOLS = ("md", r"""## 4 — Two practical tools: Adam and early stopping

### Adam
Plain SGD applies the same learning rate to all 65 parameters. **Adam** keeps a per-parameter
running estimate of gradient magnitude and scales each step accordingly, so rarely-updated
parameters get larger steps. It is the sensible default for neural networks and needs far
less learning-rate tuning than SGD.

### Early stopping — the first real use of the validation set
With 65 parameters the model *can* start memorising noise. The symptom:

- training loss keeps falling (it is fitting the noise),
- **validation loss stops falling and starts rising** — that is over-fitting.

Early stopping monitors validation loss each epoch, remembers the weights that achieved the
best value, and stops after `patience=30` epochs without improvement. The returned model is
the *best* one seen, not the last one.

This is exactly why we keep three splits: the validation set decides *when to stop*, so the
test set stays untouched and its score remains honest.""")

_TRAIN = ("code", '''mlp = build_mlp(hidden=16)
print(mlp)
n_params = sum(p.numel() for p in mlp.parameters())
print(f"\\nparameters: {n_params}   (logistic regression has 3)")
print()
hist = train_with_early_stopping(mlp, data)      # Adam lr=1e-2, patience=30
best_epoch = int(np.argmin(hist["val"])) + 1
print(f"ran {len(hist['train'])} epochs; best validation BCE = {min(hist['val']):.4f} at epoch {best_epoch}")
print(f"{len(hist['train']) - best_epoch} further epochs without improvement triggered the stop,")
print(f"and the weights from epoch {best_epoch} were restored.")
plot_loss({"train BCE": hist["train"], "val BCE": hist["val"]},
          "Step 6 — MLP training with early stopping", ylabel="binary cross-entropy")
plt.show()''')

_TRAIN_COMMENT = ("md", """**Reading this curve — compare it with step 5's.**

In step 5 (3 parameters) the train and validation curves were glued together.
Here they **separate**: the training loss keeps going down while the validation loss flattens
and begins to creep up. That gap *is* over-fitting, made visible.

Early stopping catches the model at the bottom of the validation curve — the best compromise
between under-fitting (too few epochs) and over-fitting (too many).""")

_EVAL = ("code", '''classification_report("MLP 2-16-1 - test", data.y_test, predict_class(mlp, data.X_test))''')


_COMPARE = ("md", """## 5 — The comparison that justifies the whole course""")

_COMPARE_CODE = ("code", '''from steps.step02_logistic_regression import LogisticRegressionGD
from steps.step03_perceptron import Perceptron, to_pm1

perceptron = Perceptron(lr=0.01, epochs=50).fit(data.X_train, to_pm1(data.y_train))
logistic = LogisticRegressionGD(lr=0.5, epochs=1000).fit(data.X_train, data.y_train)
fe = LogisticRegressionGD(lr=0.5, epochs=2000).fit(add_sleep_squared(data.X_train), data.y_train)

acc_per = accuracy(data.y_test, perceptron.predict01(data.X_test))
acc_log = accuracy(data.y_test, logistic.predict(data.X_test))
acc_fe  = accuracy(data.y_test, fe.predict(add_sleep_squared(data.X_test)))
acc_mlp = accuracy(data.y_test, predict_class(mlp, data.X_test))

rows = [
    ("step 3   perceptron (NumPy)",          acc_per, 3,        "straight"),
    ("step 2   logistic regression (NumPy)", acc_log, 3,        "straight"),
    ("bonus    logistic + slept^2 feature",  acc_fe,  4,        "curved (by hand)"),
    ("step 6   MLP 2-16-1 (PyTorch)",        acc_mlp, n_params, "curved (learned)"),
]
print(f"{'model':<38}{'params':<9}{'boundary':<20}{'test accuracy'}")
print("-" * 82)
for name, acc, params, shape in rows:
    print(f"{name:<38}{params:<9}{shape:<20}{acc:.3f}")
print()
print(f"The MLP beats logistic regression by {acc_mlp - acc_log:+.3f} "
      f"({(acc_mlp - acc_log) * 100:.1f} percentage points),")
print("with a more flexible boundary, Adam, and validation-based early stopping.")''')

_BOUNDARY_CODE = ("code", '''plot_decision_boundaries(
    data.X_test, data.y_test,
    {"perceptron (step 3)": perceptron.predict01,
     "logistic regression (step 2)": logistic.predict,
     "MLP 2-16-1 (step 6)": lambda X: predict_class(mlp, X)},
    "Step 6 — two straight lines, and one learned curve")
plt.show()''')

_BOUNDARY_COMMENT = ("md", r"""## 6 — Reading the boundary: the moment it all pays off

Look at the third panel. The blue "predict pass" region **curves back** in the upper area
of the plot, carving out the students who slept a lot but studied little.

**Nobody told the network about $(\text{slept}-7.5)^2$.** It discovered from 700 examples
that the relationship with sleep is non-monotonic, and encoded that in its 65 weights.
This is *feature learning*, and it is the central promise of deep learning.

Compare with the two left panels: however you rotate or shift a straight line, you cannot
produce that shape. Not with more epochs, not with a better optimiser, not with more data —
the model class is simply incapable of it.""")

_FE_EXP = ("md", r"""## 7 — Experiment: feature engineering vs feature learning

Notice the third row of the table: logistic regression with a hand-added $\text{slept}^2$
feature scores just as well as the network.

**This is an important, humbling result.** If *you* know the right transformation, a simple
linear model is enough — and it is cheaper, more interpretable, and easier to debug.

| Approach | Needs | Gives |
|---|---|---|
| **Feature engineering** (add $\text{slept}^2$) | *You* must know the right form | Best accuracy, full interpretability, 4 parameters |
| **Feature learning** (the MLP) | Only data | Same accuracy without domain knowledge, 65 parameters, harder to interpret |

The network's real advantage is not raw accuracy on this toy problem — it is that it works
**when you have no idea what the right features are** (pixels, audio waveforms, text tokens).
That is precisely where deep learning dominates and hand-engineering fails.""")

_FE_CODE = ("code", '''print("Same model class (logistic regression), different input features:")
print(f"  raw features (studied, slept)          -> {acc_log:.3f}")
print(f"  plus the engineered feature slept^2    -> {acc_fe:.3f}")
print(f"  MLP, no engineered features at all     -> {acc_mlp:.3f}")
print()
print(f"weights of the engineered model: {fe.w.round(3)}")
print("The third weight is the coefficient on slept^2 - strongly NEGATIVE,")
print("which is exactly the -2*(slept-7.5)^2 penalty of the true formula.")''')

_HIDDEN_EXP = ("md", """## 8 — Experiment: how many hidden units are needed?

`hidden=1` gives a single ReLU unit feeding the output — still effectively a straight
boundary. How many units before the curve appears?""")

_HIDDEN_CODE = ("code", '''print(f"{'hidden units':<15}{'parameters':<14}{'epochs run':<14}{'test accuracy'}")
print("-" * 58)
for h in (1, 2, 4, 8, 16, 64):
    m = build_mlp(hidden=h)
    hh = train_with_early_stopping(m, data)
    p = sum(q.numel() for q in m.parameters())
    print(f"{h:<15}{p:<14}{len(hh['train']):<14}{accuracy(data.y_test, predict_class(m, data.X_test)):.3f}")
print()
print("1 unit  -> barely better than logistic regression (still essentially a line).")
print("4 units -> already enough to bend around the sweet spot.")
print("64 units-> no real gain: the problem only needs a little curvature.")
print("Bigger is not automatically better; capacity should match the problem.")''')


_NORELU_EXP = ("md", r"""## 9 — Experiment: proof that ReLU is what matters

The claim from section 2 was that **without an activation, depth collapses to a single
linear layer**. Let's verify empirically: train `Linear(2,16) → Linear(16,1)` with *no* ReLU
between them, and compare against logistic regression.""")

_NORELU_CODE = ("code", '''torch.manual_seed(0)
no_relu = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 1))   # 65 params, NO activation
h_nr = train(no_relu, make_loader(data.X_train, data.y_train), nn.BCEWithLogitsLoss(),
             torch.optim.Adam(no_relu.parameters(), lr=1e-2), epochs=200,
             X_val=data.X_val, y_val=data.y_val)
acc_nr = accuracy(data.y_test, predict_class(no_relu, data.X_test))

print(f"{'model':<42}{'params':<10}{'test accuracy'}")
print("-" * 66)
print(f"{'logistic regression (1 linear layer)':<42}{3:<10}{acc_log:.3f}")
print(f"{'2 linear layers, NO ReLU':<42}{sum(p.numel() for p in no_relu.parameters()):<10}{acc_nr:.3f}")
print(f"{'2 linear layers WITH ReLU (our MLP)':<42}{n_params:<10}{acc_mlp:.3f}")
print()
print("65 parameters without an activation buys NOTHING over 3 parameters:")
print("the two layers collapse into one linear map, so the boundary is still a line.")
plot_decision_boundaries(
    data.X_test, data.y_test,
    {"2 layers, no ReLU (still a line!)": lambda X: predict_class(no_relu, X),
     "2 layers with ReLU (curved)": lambda X: predict_class(mlp, X)},
    "The activation, not the depth, is what creates the curve")
plt.show()''')

_SURFACE_EXP = ("md", """## 10 — Experiment: the probability surface

`predict_class` only shows the 0.5 cut. Plotting `predict_proba` reveals the full confidence
landscape the network learned — and shows that it is smooth, not a patchwork.""")

_SURFACE_CODE = ("code", '''g0, g1 = np.meshgrid(np.linspace(-2.2, 2.2, 200), np.linspace(-2.2, 2.2, 200))
grid = np.column_stack([g0.ravel(), g1.ravel()])
p_mlp = predict_proba(mlp, grid).reshape(g0.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
c0 = axes[0].contourf(g0, g1, p_mlp, levels=20, cmap="RdBu")
axes[0].contour(g0, g1, p_mlp, levels=[0.5], colors="k", linewidths=2)
plt.colorbar(c0, ax=axes[0], label="P(passed)")
axes[0].scatter(data.X_test[:, 0], data.X_test[:, 1], c=data.y_test, cmap="RdBu",
                edgecolor="k", s=18, linewidth=0.5)
axes[0].set_title("MLP probability surface (black line = 0.5 boundary)")
axes[0].set_xlabel("hours studied (std)"); axes[0].set_ylabel("hours slept (std)")

p_log = logistic.predict_proba(grid).reshape(g0.shape)
c1 = axes[1].contourf(g0, g1, p_log, levels=20, cmap="RdBu")
axes[1].contour(g0, g1, p_log, levels=[0.5], colors="k", linewidths=2)
plt.colorbar(c1, ax=axes[1], label="P(passed)")
axes[1].scatter(data.X_test[:, 0], data.X_test[:, 1], c=data.y_test, cmap="RdBu",
                edgecolor="k", s=18, linewidth=0.5)
axes[1].set_title("Logistic regression: parallel straight contours")
axes[1].set_xlabel("hours studied (std)"); axes[1].set_ylabel("hours slept (std)")
plt.tight_layout(); plt.show()
print("Left : contours bend -> the network learned that too much sleep is harmful.")
print("Right: contours are parallel straight lines -> more sleep is always 'better'.")''')

_JOURNEY = ("md", r"""## Summary — the complete journey

| Step | Model | Key new idea | Boundary | Test |
|---|---|---|---|---|
| 1 | Linear regression (NumPy) | model / loss / optimiser; gradient descent | — | R² 0.79 |
| 2 | Logistic regression (NumPy) | sigmoid + cross-entropy | straight | 0.893 |
| 3 | Perceptron (NumPy) | a neuron; why hard activations block gradients | straight | 0.873 |
| 4 | scikit-learn | `fit`/`predict`, pipelines, verified weights | straight | 0.893 |
| 5 | PyTorch `nn.Linear` | tensors, autograd, the 4-line training loop | straight | 0.893 |
| 6 | **MLP 2-16-1** | **hidden layer + ReLU → learned features** | **curved** | **0.953** |

### The ten ideas that carried through

1. A **model** is a parametric function; learning means choosing its parameters.
2. A **loss** turns "how wrong am I" into one differentiable number.
3. **Gradient descent** follows the negative gradient; the learning rate is the stride.
4. Standardising features makes optimisation well-conditioned.
5. Classification = the same linear core + a squashing function + the right loss.
6. A **neuron** is a weighted sum plus an activation — nothing more.
7. The hard step blocks ordinary gradient training; sigmoid and piecewise-linear ReLU support backpropagation.
8. Libraries implement exactly the maths you derived — verified numerically in step 4.
9. **Autograd** removes the need to derive gradients, which is what makes depth practical.
10. **Depth + non-linearity** lets a model *learn features* instead of you engineering them.

### Where to go next

| Direction | What to try |
|---|---|
| Regularisation | `weight_decay` in Adam, or `nn.Dropout(0.2)` after the ReLU; watch the val curve |
| Depth | `Linear(2,16) → ReLU → Linear(16,16) → ReLU → Linear(16,1)` — does it help here? |
| Multi-class | Predict grades A/B/C with `nn.CrossEntropyLoss` and 3 output units |
| Real data | Swap in the UCI *Student Performance* dataset — the code does not change |
| Scaling up | The same 4-line loop trains convolutional and transformer models; only the model line changes |

**You have now built, from scratch and then with libraries, the complete path from linear
regression to a neural network — and you know exactly what every line does.**""")

CELLS = [
    _INTRO,
    ("code", SETUP),
    _ARCH,
    _RELU,
    _RELU_CODE,
    _BACKPROP,
    _TOOLS,
    _TRAIN,
    _TRAIN_COMMENT,
    _EVAL,
    _COMPARE,
    _COMPARE_CODE,
    _BOUNDARY_CODE,
    _BOUNDARY_COMMENT,
    _FE_EXP,
    _FE_CODE,
    _HIDDEN_EXP,
    _HIDDEN_CODE,
    _NORELU_EXP,
    _NORELU_CODE,
    _SURFACE_EXP,
    _SURFACE_CODE,
    _JOURNEY,
]
