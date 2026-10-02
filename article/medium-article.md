# From Linear Regression to a Neural Network: One Dataset, Six Steps

*Build the maths in NumPy, verify it with scikit-learn, and discover what a hidden layer actually changes.*

A neural network is easier to understand when you can see the problem a simpler model cannot solve.

That is the idea behind this project: keep the data fixed, change the model, and plot what happens. Start with a weighted sum. Give it a loss function. Teach it to improve. Then carry those same ideas into a library, an automatic differentiation system, and finally a small neural network.

The course uses seven Jupyter notebooks: one for the data and six for the models. You can run them locally, inspect the implementations, and change the experiments yourself.

The question throughout is deliberately ordinary:

**Given how many hours a student studied and slept, can we predict their exam score—and whether they passed?**

## A small problem with a useful hidden curve

The dataset contains 1,000 synthetic students and just two input features: study hours and sleep hours.

The scores come from a rule we know:

```python
score = 35 + 5 * studied - 2 * (slept - 7.5)**2 + noise
score = np.clip(score, 0, 100)
passed = score >= 50
```

The noise is Gaussian, with a standard deviation of four score points.

Study has a linear effect in this invented world. Sleep has a quadratic effect: scores peak around 7.5 hours and fall on either side. This is a teaching device, not a claim about actual students or a prescription for sleep.

That curve gives us something specific to look for. A linear classifier using the two original features draws a straight decision boundary. It cannot bend around the quadratic sleep effect. A model with nonlinear features can.

Two features also mean we can draw the entire input space. The models' strengths and mistakes become visible rather than disappearing into a single accuracy number.

![The synthetic dataset: study, sleep and pass/fail](assets/data-overview.png)

*Study has an upward trend; sleep has a sweet spot. The pass/fail view reveals the shape each classifier must approximate.*

Before fitting anything, the data is split into 700 training rows, 150 validation rows and 150 test rows. Feature means and standard deviations are computed on the training set only, then applied to the other two sets.

Training fits the parameters. Validation helps make training decisions. Test data is reserved for evaluation. Keeping those jobs separate is part of learning machine learning, not an administrative detail.

## Step 1: learning means changing parameters

Linear regression starts with a prediction:

```python
prediction = X @ w + b
```

`w` contains one weight per feature. `b` is the bias. Initially, these parameters do not describe the data well.

Mean squared error measures how far the predicted scores are from the observed scores. Gradient descent then updates the parameters in the direction that reduces that error:

```python
error = X @ w + b - y
w -= learning_rate * (2 / len(X)) * (X.T @ error)
b -= learning_rate * 2 * error.mean()
```

This introduces the three pieces that stay with us through the course:

- **Model:** how a prediction is computed.
- **Loss:** how a prediction is judged.
- **Optimiser:** how parameters are changed.

The notebook compares the gradient descent solution with the closed-form least-squares solution. That gives us a concrete check: our iterative implementation should converge to the same fitted parameters.

It also varies the learning rate. Small steps converge slowly; steps that are too large can diverge. Standardising the features helps make one learning rate work sensibly across inputs with different scales.

The model explains much of the score variation, with a reference test R² near 0.79. But it cannot express the quadratic sleep effect using only a weighted sum of the two raw features. Successful optimisation does not remove a limitation in the model itself.

## Step 2: keep the linear core, change the task

Now the target becomes pass or fail rather than a numerical score.

Logistic regression keeps the same weighted sum, but applies a sigmoid:

```python
logit = X @ w + b
probability = 1 / (1 + np.exp(-logit))
```

The result is between zero and one. We train with binary cross-entropy and classify a student as passing when the predicted probability is at least 0.5.

The gradient has a familiar form:

```python
error = probability - y
w -= learning_rate * (X.T @ error) / len(X)
b -= learning_rate * error.mean()
```

A new task, a new output transformation, a new loss—and much of the original structure survives.

The notebook also changes the classification threshold and examines the confusion matrix. A probability contains more information than a yes/no prediction; choosing a threshold changes which kinds of mistakes we make.

The reference test accuracy is roughly 89%. The boundary is still straight: applying the sigmoid does not give a linear score the ability to model a curved boundary.

## Step 3: what is an artificial neuron?

The perceptron strips the output down to a hard threshold. Compute a weighted sum, then choose a class based on its sign.

This makes the idea of a neuron tangible: weights combine inputs, a bias shifts the result, and an activation produces the output.

But a hard step has zero derivative away from its discontinuity. Ordinary gradient descent through that activation provides no useful learning signal. The perceptron instead uses an error-driven update rule when it misclassifies an example.

On linearly separable data, the classical perceptron algorithm can find a separating boundary. Our dataset includes both noise and a curved rule, so that guarantee does not apply. The notebook shows the error count oscillating and explores the effect of sample order.

The lesson is more valuable than the accuracy: a model's representation and its learning rule both matter. Training longer cannot make a straight boundary represent a curve.

## Step 4: make libraries less mysterious

After implementing the models ourselves, we use scikit-learn.

Its estimator interface is compact:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The notebook compares library fits with the NumPy implementations and checks their fitted weights. For meaningful comparisons, the preprocessing and optimisation objective must match; scikit-learn's default logistic regression regularisation is a detail worth inspecting.

Pipelines combine scaling and estimation so that preprocessing is fitted together with the model on training data.

This step changes how much code we write. It does not automatically change what a linear model can represent. The notebook also explores other estimators, including nonlinear ones, to show that the library is broader than its linear models.

## Step 5: let PyTorch compute the gradients

PyTorch introduces tensors, mini-batches and automatic differentiation.

A single layer still expresses our familiar linear mapping:

```python
model = nn.Linear(2, 1)
```

Use mean squared error for score prediction, or `BCEWithLogitsLoss` for binary classification. The latter combines the sigmoid and binary cross-entropy in a numerically stable calculation, so the model returns logits during training.

The core loop is short:

```python
optimizer.zero_grad()
loss = loss_fn(model(x_batch), y_batch)
loss.backward()
optimizer.step()
```

`backward()` computes parameter gradients through the operations that produced the loss. We no longer have to derive and implement each gradient ourselves.

The notebook varies batch sizes and optimisers. These choices affect training dynamics, but a single linear layer still draws a straight classification boundary. Automatic differentiation makes training more convenient; changing the representation requires changing the model.

## Step 6: give the model nonlinear features

Finally, replace the single layer with:

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
```

The network has 65 trainable parameters. Its hidden layer computes 16 features from the two inputs, and its output layer combines those learned features into a logit.

ReLU is the crucial ingredient. Without a nonlinear activation, two stacked affine layers collapse into one:

```text
W2 @ (W1 @ x + b1) + b2
= (W2 @ W1) @ x + (W2 @ b1 + b2)
```

More layers alone would leave us with the same representational limitation. ReLU introduces piecewise-linear features, allowing the final boundary to bend.

The core PyTorch training loop is reused, while this step also introduces Adam and early stopping. Validation loss determines when to stop, and the best validation checkpoint is restored before evaluation.

![Straight boundaries compared with the learned nonlinear boundary](assets/decision-boundaries.png)

*The perceptron and logistic regression draw straight boundaries. The small MLP can bend its boundary around the synthetic sleep effect.*

In the verified local run, the MLP correctly classifies 143 of 150 test examples (95.3%), compared with 134 of 150 (89.3%) for logistic regression. That is a useful illustration, not a universal ranking: the test set has only 150 examples, and the models differ in optimiser and stopping strategy as well as architecture. Exact results can change with dependency versions and training conditions.

## The experiment that keeps the conclusion honest

The final notebook also gives logistic regression an extra feature:

```python
X_with_curve = np.column_stack([X, X[:, 1] ** 2])
```

Now a model linear in its parameters can express a nonlinear boundary in the original input space. A squared standardised sleep feature still supplies the required quadratic term when combined with the original sleep feature and an intercept.

In the same run, the engineered-feature model correctly classifies 144 of 150 examples (96.0%)—one more than the MLP:

| Model | Correct predictions | Test accuracy |
|---|---|---|
| Perceptron | 131 / 150 | 87.3% |
| Logistic regression | 134 / 150 | 89.3% |
| Logistic regression + squared sleep | 144 / 150 | 96.0% |
| MLP, 2 → 16 → 1 | 143 / 150 | 95.3% |

These values come from one fixed split on Python 3.12; the repository records the tested dependency versions. A one-example difference should not be treated as evidence of a general advantage.

This baseline matters. Neural networks are one way to learn useful nonlinear features; they are not the only way to model nonlinear relationships. When we already know which feature transformation a problem needs, a simpler model can be competitive.

The MLP's contribution here is that it learns a flexible representation from the two original inputs. The feature-engineered model gets a representation we designed ourselves.

## Run the course yourself

The companion repository is intended to be available at [clembnl/ai-learning-path](https://github.com/clembnl/ai-learning-path). **This draft's repository link must be verified after publication before posting the article.**

Install Python 3.11 or 3.12, then download the whole repository as a ZIP, or clone it:

```bash
git clone https://github.com/clembnl/ai-learning-path.git
cd ai-learning-path
python3 scripts/start.py
```

On Windows, use `py -3.12 scripts/start.py` for the final command.

The launcher creates an isolated environment, installs the libraries, registers the notebook kernel and opens JupyterLab. Open `00_data.ipynb`, select **Python (ai-learning-path)**, and run the cells in order. Continue through notebooks 01–06.

No GPU is required. All examples use a local synthetic CSV, and the repository includes the source scripts, tests and notebook generator. The setup follows the official [JupyterLab installation guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) and [PyTorch installer](https://pytorch.org/get-started/locally/).

For each notebook, change one thing before moving on: the learning rate, classification threshold, batch size or hidden-layer width. Predict what will happen, run the experiment, and compare the plot with your prediction.

The goal is to recognise the same structure as the tools become more capable: inputs flow through a model, a loss judges its predictions, and an optimiser changes its parameters. Once those pieces are familiar, a small neural network becomes something you can inspect, question and build yourself.
