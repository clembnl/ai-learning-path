from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 05_pytorch.ipynb by notebooks/build_notebooks.py.

_INTRO = ("md", r"""# Step 5 — The same models in PyTorch

## The strategy: learn the tool, not a new model

This notebook introduces **no new machine learning**. The models are exactly the linear
and logistic regressions of steps 1 and 2. What changes is the *framework*.

That is deliberate. PyTorch has many moving parts (tensors, autograd, modules, optimisers,
data loaders). Learning them on a model you already understand completely means that when
something looks odd, you know it is the *framework* and not the maths.

By the end you will have written the canonical training loop that is **identical** for a
3-parameter linear regression and a 175-billion-parameter language model.

## Why PyTorch at all, if scikit-learn already works?

| | scikit-learn | PyTorch |
|---|---|---|
| Model catalogue | Huge, but fixed | You *compose* models from layers |
| Gradients | Hidden inside each estimator | **Computed automatically for anything you write** |
| Suited to | Classical ML on tabular data | Deep learning, custom architectures, GPUs |

The decisive feature is **autograd**. In steps 1–2 you derived
$\partial\mathcal L/\partial w$ with pen and paper. That is feasible for one linear layer.
For the network in step 6 — and anything deeper — it is not. Autograd removes that ceiling.""")

_TENSORS = ("md", r"""## 1 — Tensors: arrays that remember their history

A `torch.Tensor` is a NumPy array plus one crucial extra: with `requires_grad=True`,
PyTorch records **every operation** applied to it, building a *computation graph*.
Calling `.backward()` walks that graph backwards and computes the derivative of the final
scalar with respect to every input.

Two conventions in our helper `to_tensor()` (see `steps/step05_pytorch.py`):
- Everything becomes `float32` — the standard precision for deep learning.
- Targets are reshaped from `(n,)` to `(n, 1)`, because PyTorch losses require predictions
  and targets to have *identical* shapes and `nn.Linear(2, 1)` outputs `(n, 1)`.

Let's verify autograd against derivatives we can check by hand.""")

_AUTOGRAD_CODE = ("code", '''import torch
from torch import nn
from steps.step05_pytorch import (to_tensor, make_loader, train, predict_regression,
                                 predict_class, predict_proba,
                                 build_linear_regression, build_logistic_regression)

# Example 1: L = sum(w^2)  =>  dL/dw = 2w
w = torch.tensor([1.0, -2.0], requires_grad=True)
(w ** 2).sum().backward()
print("Example 1:  L = sum(w^2)")
print(f"  w         = {w.detach().numpy()}")
print(f"  w.grad    = {w.grad.numpy()}      <- computed by autograd")
print(f"  expected  = {(2 * w).detach().numpy()}      <- 2w, by hand")
print()

# Example 2: reproduce the step-1 MSE gradient exactly
X = torch.tensor(data.X_train, dtype=torch.float32)
y = torch.tensor(data.score_train, dtype=torch.float32)
w2 = torch.zeros(2, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
((X @ w2 + b2 - y) ** 2).mean().backward()
n = len(X)
manual = (2.0 / n) * X.T @ (X @ w2.detach() + b2.detach() - y)
print("Example 2:  the step-1 MSE gradient")
print(f"  autograd  dL/dw          = {w2.grad.numpy().round(6)}")
print(f"  our formula 2/n*X^T(y_hat-y) = {manual.numpy().round(6)}")
print(f"  max difference = {(w2.grad - manual).abs().max().item():.2e}")
print()
print("=> autograd computes EXACTLY the formula you derived by hand in step 1.")''')

_LOOP = ("md", r"""## 2 — The training loop: four lines to memorise

```python
for xb, yb in loader:              # one mini-batch of samples
    optimizer.zero_grad()          # 1. clear gradients from the previous step
    loss = loss_fn(model(xb), yb)  # 2. forward pass + measure the loss
    loss.backward()                # 3. backward pass: autograd fills every .grad
    optimizer.step()               # 4. update parameters using those .grad values
```

**Why `zero_grad()` is not optional.** PyTorch *accumulates* gradients into `.grad` by
default (a deliberate feature used by some advanced techniques). If you forget to clear
them, batch 2 adds its gradients on top of batch 1's and your effective learning rate
silently explodes. Forgetting `zero_grad()` is the single most common PyTorch bug.

**The division of labour:**

| Line | Who does the work |
|---|---|
| `model(xb)` | your model definition |
| `loss_fn(...)` | the loss function you chose |
| `loss.backward()` | **autograd** — replaces your hand-derived gradient |
| `optimizer.step()` | the optimiser (`w -= lr * w.grad` for plain SGD) |

The `train()` helper below wraps this loop and also records the validation loss each epoch.
**Step 6 reuses this exact function unchanged** — proof that the loop does not care how
complex the model is.""")

_LOOP_SRC = ("code", '''import inspect
print(inspect.getsource(train))''')

_MODULES = ("md", r"""## 3 — `nn.Linear`, and where the sigmoid went

```python
model = nn.Linear(2, 1)     # 2 inputs -> 1 output
```

This single object holds `weight` (shape `(1,2)`) and `bias` (shape `(1,)`) and computes
$Xw^\top + b$. **It is literally the model from step 1.**

For classification we use the *same* `nn.Linear(2, 1)`. So where is the sigmoid?

**Inside the loss.** `nn.BCEWithLogitsLoss()` applies the sigmoid internally, then computes
binary cross-entropy. This is not laziness — it is numerical stability:

- Computing `sigmoid(z)` then `log(p)` separately can produce `log(0) = -inf` when
  $z$ is very negative.
- The fused version rearranges the algebra to the *log-sum-exp* form, which never overflows.

**The practical rule:**

| Situation | What to do |
|---|---|
| Training a binary classifier | model outputs **logits** (raw $z$), use `BCEWithLogitsLoss` |
| Getting probabilities for inspection | apply `torch.sigmoid(model(x))` yourself |
| Getting hard class predictions | `z > 0` is equivalent to `sigmoid(z) > 0.5` |

Our helpers do exactly that: `predict_proba` applies the sigmoid, `predict_class`
thresholds at 0.5.""")

_BATCH_THEORY = ("md", r"""## 4 — Mini-batches: a third flavour of gradient descent

Steps 1–2 used **full-batch** gradient descent: one gradient over all 700 samples, one step.
PyTorch's `DataLoader` serves shuffled **mini-batches** (64 here), so one epoch performs
$\lceil 700/64 \rceil = 11$ updates instead of 1.

| Variant | Samples per update | Updates per epoch | Character |
|---|---|---|---|
| Full-batch (steps 1–2) | 700 | 1 | smooth but slow per epoch |
| **Mini-batch (standard)** | 64 | 11 | the practical compromise |
| Stochastic (strict SGD) | 1 | 700 | very noisy |

Consequences:
- **Faster convergence per epoch** — 11× more updates. This is why the torch versions need
  only 100–200 epochs where our NumPy loops used 500–1000.
- **Noisier loss curve** — each batch is a slightly biased sample. The noise is usually
  *beneficial*: it helps escape flat or poor regions of the loss surface.
- **Necessary for large data** — you cannot fit a million images into memory at once.""")

_LIN_RUN = ("code", '''lin, hist = build_linear_regression(data)      # nn.Linear(2,1) + MSELoss + SGD
w_torch = lin.weight.detach().numpy()[0]
print("LINEAR REGRESSION in PyTorch")
print(f"  torch  : w={w_torch.round(4)}  b={lin.bias.item():.4f}")
print(f"  step 1 : w=[15.3466  7.8198]  b=49.8436   (our NumPy gradient descent)")
print(f"  step 4 : identical to sklearn LinearRegression")
print()
regression_report("linear regression - torch", data.score_test, predict_regression(lin, data.X_test))
plot_loss({"train MSE": hist["train"], "val MSE": hist["val"]},
          "Step 5 — torch linear regression (100 epochs, mini-batch)", ylabel="MSE")
plt.show()''')

_LIN_COMMENT = ("md", """Two things to notice on this curve:

1. **It converges in ~20 epochs**, versus ~100 for our full-batch NumPy version, thanks to
   11 updates per epoch.
2. **Train and validation curves sit on top of each other.** With 3 parameters and 700
   samples there is no capacity to over-fit. Keep this shape in mind — in step 6 the model
   has 65 parameters and the two curves start to separate.""")

_LOG_RUN = ("code", '''log, hist = build_logistic_regression(data)    # SAME nn.Linear(2,1), different loss
w_torch = log.weight.detach().numpy()[0]
print("LOGISTIC REGRESSION in PyTorch  (identical model object, BCEWithLogitsLoss instead of MSELoss)")
print(f"  torch  : w={w_torch.round(4)}  b={log.bias.item():.4f}")
print(f"  step 2 : w=[3.4615  1.7336]  b=-0.0755   (our NumPy gradient descent)")
print()
classification_report("logistic regression - torch", data.y_test, predict_class(log, data.X_test))
plot_loss({"train BCE": hist["train"], "val BCE": hist["val"]},
          "Step 5 — torch logistic regression", ylabel="binary cross-entropy")
plt.show()
plot_decision_boundaries(data.X_test, data.y_test,
                         {"torch nn.Linear(2,1)": lambda X: predict_class(log, X)},
                         "Step 5 — still the same straight line as step 2")
plt.show()''')

_LOG_COMMENT = ("md", """Same accuracy, same weights (to within mini-batch noise), **same straight boundary**.

The lesson: switching framework changed nothing about the model's capability. PyTorch did
not make the model smarter — it removed the need to derive gradients, which is what lets us
make the model *structurally* richer in step 6.""")


_BATCH_EXP = ("md", """## 5 — Experiment: the effect of batch size

Setting `batch_size = len(X_train)` recovers exactly the full-batch gradient descent of
step 2. Smaller batches mean noisier but far more frequent updates.

All runs below use the same learning rate and the same 50 epochs — only batch size changes.""")

_BATCH_CODE = ("code", '''results = {}
print(f"{'batch size':<14}{'updates/epoch':<16}{'final train BCE':<18}{'test accuracy'}")
print("-" * 62)
for bs in (8, 32, 64, len(data.X_train)):
    torch.manual_seed(0)
    m = nn.Linear(2, 1)
    h = train(m, make_loader(data.X_train, data.y_train, batch_size=bs),
              nn.BCEWithLogitsLoss(), torch.optim.SGD(m.parameters(), lr=0.1), epochs=50)
    results[f"batch={bs}"] = h["train"]
    n_upd = int(np.ceil(len(data.X_train) / bs))
    print(f"{bs:<14}{n_upd:<16}{h['train'][-1]:<18.4f}"
          f"{accuracy(data.y_test, predict_class(m, data.X_test)):.3f}")

plot_loss(results, "Effect of batch size (same lr=0.1, same 50 epochs)", ylabel="train BCE")
plt.show()
print()
print("Small batches: many updates -> lowest loss after 50 epochs, but a wobblier curve.")
print("Full batch (700): only 50 updates in total -> clearly under-trained at 50 epochs.")''')

_OPTIM_EXP = ("md", r"""## 6 — Experiment: optimisers beyond plain SGD

`optimizer.step()` hides a design choice. Plain SGD performs the update from step 1:

$$w \leftarrow w - \eta \nabla w$$

Better optimisers adapt the step size automatically:

| Optimiser | Idea | When to use |
|---|---|---|
| `SGD` | fixed step — exactly our formula | simple, needs careful lr tuning |
| `SGD(momentum=0.9)` | accumulates a velocity, damping oscillations | classic for vision models |
| `Adam` | per-parameter adaptive step sizes | **the sensible default** for neural nets |

Adam is what step 6 uses. Compare how fast each reaches a low loss:""")

_OPTIM_CODE = ("code", '''curves = {}
configs = {
    "SGD (lr=0.1)":       lambda p: torch.optim.SGD(p, lr=0.1),
    "SGD + momentum 0.9": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
    "Adam (lr=0.1)":      lambda p: torch.optim.Adam(p, lr=0.1),
}
print(f"{'optimiser':<26}{'final train BCE':<18}{'test accuracy'}")
print("-" * 58)
for name, make_opt in configs.items():
    torch.manual_seed(0)
    m = nn.Linear(2, 1)
    h = train(m, make_loader(data.X_train, data.y_train), nn.BCEWithLogitsLoss(),
              make_opt(m.parameters()), epochs=60)
    curves[name] = h["train"]
    print(f"{name:<26}{h['train'][-1]:<18.4f}"
          f"{accuracy(data.y_test, predict_class(m, data.X_test)):.3f}")
plot_loss(curves, "Optimisers: same model, same lr, 60 epochs", ylabel="train BCE")
plt.show()
print()
print("Momentum and Adam reach a low loss faster than plain SGD.")
print("On this convex 3-parameter problem all three end up in the same place;")
print("on the non-convex network of step 6 the difference is much larger.")''')


_EVAL_MODE = ("md", r"""## 7 — Two habits that prevent silent bugs

### `model.train()` vs `model.eval()`
Some layers behave differently during training and inference (dropout active/inactive,
batch-norm using batch vs running statistics). Our linear model has none of those, so it
changes nothing *here* — but form the habit now, because forgetting `model.eval()` with
dropout produces randomly wrong predictions that are very hard to debug.

### `torch.no_grad()`
During inference you do not need gradients. Wrapping predictions in `with torch.no_grad():`
tells PyTorch not to build the computation graph, saving memory and time.
Our `predict_*` helpers are decorated with `@torch.no_grad()`, which is why they can return
plain NumPy arrays.""")

_EVAL_CODE = ("code", '''X_t = to_tensor(data.X_test[:3])
with_graph = log(X_t)
print("without no_grad -> requires_grad =", with_graph.requires_grad)
try:
    with_graph.numpy()
except RuntimeError as e:
    print("  .numpy() fails:", str(e)[:75], "...")
with torch.no_grad():
    no_graph = log(X_t)
print("with    no_grad -> requires_grad =", no_graph.requires_grad,
      "   .numpy() works:", no_graph.numpy().ravel().round(3))
print()
print("Our predict_* helpers use @torch.no_grad(), so they return clean NumPy arrays.")''')

_SUMMARY = ("md", r"""## Summary — what you learned in step 5

| Concept | Takeaway |
|---|---|
| **Tensor** | NumPy array + gradient tracking (`requires_grad`). |
| **Autograd** | `loss.backward()` fills `.grad` for every parameter — verified identical to our hand-derived $\frac2n X^\top(\hat y-y)$. |
| **`nn.Linear(2,1)`** | *Is* linear regression (with `MSELoss`) and *is* logistic regression (with `BCEWithLogitsLoss`). |
| **`BCEWithLogitsLoss`** | Fuses sigmoid + cross-entropy for numerical stability; the model outputs raw **logits**. |
| **The 4-line loop** | `zero_grad` → forward+loss → `backward` → `step`. Identical for every model, forever. |
| **`zero_grad()`** | Mandatory: gradients accumulate by default. The #1 PyTorch bug. |
| **Mini-batches** | 11 updates per epoch instead of 1 → faster convergence, noisier curve. |
| **Optimisers** | SGD / momentum / Adam differ in how they turn gradients into steps. |
| **`eval()` + `no_grad()`** | Inference hygiene: correct layer behaviour, no wasted graph building. |
| **Result** | Same weights, same accuracy, same straight boundary as steps 1–2 and 4. |

### Why this matters for step 6

You now have a training loop that works for *any* model expressible as an `nn.Module`.
Step 6 changes **one line** —
`model = nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,1))` —
and everything else stays identical. That is the moment the ceiling lifts.

**Next:** add a hidden layer and finally bend that boundary.""")

CELLS = [
    _INTRO,
    ("code", SETUP),
    _TENSORS,
    _AUTOGRAD_CODE,
    _LOOP,
    _LOOP_SRC,
    _MODULES,
    _BATCH_THEORY,
    _LIN_RUN,
    _LIN_COMMENT,
    _LOG_RUN,
    _LOG_COMMENT,
    _BATCH_EXP,
    _BATCH_CODE,
    _OPTIM_EXP,
    _OPTIM_CODE,
    _EVAL_MODE,
    _EVAL_CODE,
    _SUMMARY,
]
