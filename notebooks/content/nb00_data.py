from . import SETUP

# Each tuple is ("md", markdown_source) or ("code", python_source).
# Assembled into 00_data.ipynb by notebooks/build_notebooks.py.

_INTRO = ("md", r"""# 00 — The problem and the data

## One problem for the whole course

Every one of the six steps that follow solves the **same problem**:

> Given how many hours a student **studied** and **slept**, predict
> their exam **score** (a number → *regression*)
> and whether they **passed** (yes/no → *classification*).

Keeping the problem fixed is deliberate. When the model changes but the data stays
the same, you remove changes of dataset as a source of differences. Optimisers,
hyperparameters and random seeds can still affect the result. That is what makes the six steps comparable.

## Why synthetic data?

The dataset is **generated** by `common/data.py` rather than downloaded:

$$\text{score} = 35 + 5\cdot\text{studied} - 2\,(\text{slept}-7.5)^2 + \varepsilon,
\qquad \varepsilon\sim\mathcal N(0, 4^2)$$

$$\text{passed} = \begin{cases}1 & \text{if score} \ge 50\\ 0 & \text{otherwise}\end{cases}$$

Three advantages:
1. **We know the true answer.** After training we can check whether the learned weights
   match the real coefficients (5.0 for studied). Compare in original units: standardised weights use a different scale, and
a linear model that omits the quadratic term need not recover the generating coefficients.
2. **We control the difficulty.** The noise level and the shape of the sleep effect were
   chosen so linear models reach ~89 % accuracy and the neural network ~96 % — a gap
   large enough to see, small enough to stay honest.
3. **Only 2 features**, so everything can be drawn in 2D.""")

_STRUCTURE = ("md", r"""## The structure of the generating formula

| Term | Meaning |
|---|---|
| $35$ | Base score — a student who studies 0 h and sleeps exactly 7.5 h scores 35. |
| $+5 \cdot \text{studied}$ | **Linear** effect: every extra study hour is worth +5 points, always. |
| $-2\,(\text{slept}-7.5)^2$ | **Non-linear** effect: a downward parabola peaking at 7.5 h. Sleeping 5 h *or* 10 h both cost $-2 \times 2.5^2 = -12.5$ points. |
| $\varepsilon \sim \mathcal N(0, 4^2)$ | **Irreducible noise**: exam-day luck. No model can ever predict this. |

That quadratic sleep term is the heart of the course.
**A linear model on these two raw features cannot represent a parabola.
Adding a squared-sleep feature changes that limitation.** It will fit the best straight line it can
and be systematically wrong at the extremes.  In step 6, the neural network will
discover the curve on its own — that is the payoff of the whole journey.""")

_VOCAB = ("md", r"""## Vocabulary you need for every step

| Term | Definition | Here |
|---|---|---|
| **Feature / input** ($X$) | What the model is allowed to look at | `hours_studied`, `hours_slept` |
| **Target / label** ($y$) | What the model must predict | `score` (regression) or `passed` (classification) |
| **Sample** | One row = one student | 1 000 of them |
| **Training set** | Data used to fit the parameters | 700 samples (70 %) |
| **Validation set** | Data used to *tune* choices (when to stop, how many neurons) | 150 samples (15 %) |
| **Test set** | Touched **once**, at the very end, to report honest performance | 150 samples (15 %) |
| **Standardisation** | Rescale each feature to mean 0, std 1 | $z = (x-\mu)/\sigma$ |

### Why three splits and not two?

If you use the test set to make decisions (e.g. "the MLP with 16 neurons scores better than
with 8, let's keep 16"), you have *leaked* test information into your model choice, and your
final number is optimistically biased. The validation set exists to absorb those decisions
so the test set stays pristine.

### The golden rule of standardisation

$\mu$ and $\sigma$ are computed on the **training set only**, then applied unchanged to
validation and test. Computing them on the full dataset would let the model peek at test
statistics — a subtle but real form of leakage. `common/data.py::load_split_standardized()`
enforces this automatically.""")

_EXPLORE_INTRO = ("md", """## Looking at the data before modelling anything

Always plot your data first. Three views:
1. score vs `hours_studied` — should look like a straight upward trend.
2. score vs `hours_slept` — should reveal the parabola.
3. the two features against each other, coloured by pass/fail — this is the space in which
   every classifier of steps 2–6 draws its decision boundary.""")

_EXPLORE = ("code", '''X, score, passed = load()
print(f"dataset shape: X={X.shape}, score={score.shape}, passed={passed.shape}")
print(f"pass rate: {passed.mean():.1%}   (deliberately balanced, so accuracy is meaningful)")
print(f"score: min={score.min():.1f} mean={score.mean():.1f} max={score.max():.1f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].scatter(X[:, 0], score, s=6, alpha=0.5)
axes[0].set_xlabel("hours studied"); axes[0].set_ylabel("score")
axes[0].set_title("LINEAR: more study, higher score")
axes[1].scatter(X[:, 1], score, s=6, alpha=0.5, c="darkorange")
axes[1].set_xlabel("hours slept"); axes[1].set_ylabel("score")
axes[1].set_title("NON-LINEAR: sweet spot near 7.5h")
axes[1].axvline(7.5, ls="--", c="k", lw=1, label="true optimum 7.5h"); axes[1].legend()
axes[2].scatter(X[passed == 0, 0], X[passed == 0, 1], s=6, c="#ca0020", label="failed")
axes[2].scatter(X[passed == 1, 0], X[passed == 1, 1], s=6, c="#0571b0", label="passed")
axes[2].set_xlabel("hours studied"); axes[2].set_ylabel("hours slept")
axes[2].set_title("CLASSIFICATION: can one line separate these?"); axes[2].legend()
plt.tight_layout(); plt.show()''')

_EXPLORE_COMMENT = ("md", r"""**What to notice in each panel:**

**Left** — a clear upward linear trend, but with a lot of vertical spread.
That spread is *not* random noise alone: it is mostly the hidden sleep effect.
A model given only `hours_studied` could never explain it.

**Middle** — the parabola is visible. Scores peak around 7.5 h of sleep and fall off on
both sides. **This is the shape no straight line can capture.**

**Right** — the classification view. Notice the top-left corner: students who slept a lot
(9–10 h) but studied little **failed**. A single straight line must sacrifice either those
points or others. Keep this picture in mind — you will see exactly this line drawn in
steps 2, 3, 4, 5, and then *bent* in step 6.""")


_SPLIT_INTRO = ("md", """## The split and the standardisation, step by step

Let us perform manually what `load_split_standardized()` does for every step,
so you can see each stage.""")

_SPLIT_CODE = ("code", '''from common.data import split, Standardizer
s = split(X, score, passed)
print(f"train: {len(s.X_train)}   val: {len(s.X_val)}   test: {len(s.X_test)}")
print(f"pass rate  train={s.y_train.mean():.1%}  val={s.y_val.mean():.1%}  test={s.y_test.mean():.1%}")
print()
scaler = Standardizer.fit(s.X_train)      # statistics from TRAIN only
print("train mean:", scaler.mean.round(3), "  train std:", scaler.std.round(3))
print()
X_tr_std = scaler.transform(s.X_train)
X_te_std = scaler.transform(s.X_test)
print("after standardising:")
print(f"  train mean={X_tr_std.mean(axis=0).round(6)} std={X_tr_std.std(axis=0).round(6)}  <- exactly 0 and 1")
print(f"  test  mean={X_te_std.mean(axis=0).round(3)} std={X_te_std.std(axis=0).round(3)}  <- close but NOT exactly 0/1")''')

_SPLIT_COMMENT = ("md", """The test set's mean is **not** exactly 0 after transformation — and that is correct.
We applied the *training* mean and std to it. If the test mean were exactly 0, it would mean
we had computed statistics on the test data, which is the leakage we are avoiding.

All six steps call `load_split_standardized()`, which performs exactly this sequence.""")

_SUMMARY = ("md", r"""## Summary — the setup for all six steps

| Item | Value |
|---|---|
| Features | `hours_studied` ∈ [0,10], `hours_slept` ∈ [3,10] |
| Regression target | `score` ∈ [0,100] |
| Classification target | `passed` = score ≥ 50 (≈ 50 % positive) |
| Samples | 1 000 (700 train / 150 val / 150 test) |
| Preprocessing | Standardisation with train-set statistics only |
| Hidden difficulty | The quadratic sleep term — invisible to linear models |
| Irreducible noise | $\sigma = 4$ score points |

**What to expect as you progress:**

| Step | Model | Expected test result |
|---|---|---|
| 1 | Linear regression (NumPy) | R² ≈ 0.79 |
| 2 | Logistic regression (NumPy) | accuracy ≈ 0.89 |
| 3 | Perceptron (NumPy) | accuracy ≈ 0.87 |
| 4 | scikit-learn versions | identical to steps 1–3 |
| 5 | PyTorch versions | identical to steps 1–2 |
| 6 | Small neural network | **accuracy ≈ 0.95** |

**Next:** step 1 builds linear regression from scratch and introduces the
model / loss / optimiser triple that every later step reuses.""")

CELLS = [
    _INTRO,
    _STRUCTURE,
    _VOCAB,
    ("code", SETUP),
    _EXPLORE_INTRO,
    _EXPLORE,
    _EXPLORE_COMMENT,
    _SPLIT_INTRO,
    _SPLIT_CODE,
    _SPLIT_COMMENT,
    _SUMMARY,
]
