# AI Learning Path: From Linear Regression to a Neural Network

Learn the foundations of machine learning through **one problem, seven guided notebooks, and six modelling steps**. Build models in NumPy, verify them with scikit-learn, then train a small neural network with PyTorch.

**Start here:** [00 — The data](notebooks/00_data.ipynb). Read the companion [article](article/medium-article.md).

## Download and run

Install **Python 3.11 or 3.12** first. A CPU is enough; no GPU, account or external dataset is needed to run the course.

Download this repository using GitHub's **Code → Download ZIP**, extract it, and open a terminal in the extracted folder. Or clone it:

```bash
git clone https://github.com/clembnl/ai-learning-path.git
cd ai-learning-path
```

On macOS/Linux:

```bash
python3 scripts/start.py
```

On Windows:

```powershell
py -3.12 scripts/start.py
```

The launcher creates `.venv`, installs `requirements.txt`, registers **Python (ai-learning-path)** inside that environment, and opens JupyterLab. Open `00_data.ipynb`, select that kernel, and choose **Run → Run All Cells**. Continue in numerical order. Use `--no-browser` to print the local URL without opening a browser. First installation needs Internet access and downloads PyTorch; subsequent sessions can use `python3 scripts/start.py --skip-install`.

## What you will learn

| Notebook | Focus | Main idea |
|---|---|---|
| [00 — Data](notebooks/00_data.ipynb) | Synthetic data, splits, scaling | Train, validation and test have different jobs |
| [01 — Linear regression](notebooks/01_linear_regression.ipynb) | NumPy, MSE, gradient descent | A model + a loss + an optimiser |
| [02 — Logistic regression](notebooks/02_logistic_regression.ipynb) | Sigmoid, binary cross-entropy | Turn a linear score into a probability |
| [03 — Perceptron](notebooks/03_perceptron.ipynb) | Hard threshold, error-driven updates | One artificial neuron and its limitations |
| [04 — scikit-learn](notebooks/04_sklearn.ipynb) | Estimator API, pipelines | Check library models against your own maths |
| [05 — PyTorch](notebooks/05_pytorch.ipynb) | Tensors, autograd, mini-batches | Automate differentiation |
| [06 — Neural network](notebooks/06_neural_network.ipynb) | Hidden layer, ReLU, Adam, early stopping | Learn a nonlinear boundary |

Basic Python and some familiarity with arrays are useful. The notebooks explain the equations and include experiments with learning rates, thresholds, regularisation, batch sizes and hidden layers.

## One deliberately simple dataset

1,000 synthetic students have two features, `hours_studied` and `hours_slept`. Predict an exam score or a pass/fail label:

```text
score = clip(35 + 5 * studied - 2 * (slept - 7.5)**2 + noise, 0, 100)
noise ~ Normal(0, 4²)
passed = score >= 50
```

This is an invented teaching example, not evidence about students or sleep. Data generation uses seed 42; the split uses seed 0: 700 training, 150 validation, 150 test rows. Standardisation uses training statistics only.

Typical reference results are R² around 0.79 for linear regression, accuracy around 89% for logistic regression, and around 95% for the 2→16→1 MLP. These are small-sample teaching results, not guarantees or a benchmark. The MLP has 65 parameters; logistic regression has 3. A quadratic engineered feature also gives logistic regression a curved boundary (96.0% in the verified local run, versus 95.3% for the MLP). Optimisation and early stopping differ, so the comparison does not isolate architecture alone.

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name ai-learning-path --display-name "Python (ai-learning-path)"
python -m jupyterlab notebooks/
```

Windows: create the environment with `py -3.12 -m venv .venv` and activate using `.venv\Scripts\Activate.ps1`. If PowerShell activation is blocked, use `.venv\Scripts\python.exe` directly or the launcher; activation is optional.

Installation references: [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html), [PyTorch](https://pytorch.org/get-started/locally/). On Linux, install CPU PyTorch from its official CPU index before the other requirements to avoid unnecessary accelerator packages.

## Check the project

After setup, use the environment's Python (`.venv/bin/python` on macOS/Linux, `.venv\Scripts\python.exe` on Windows):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_notebooks.py
.venv/bin/python -m steps.step06_neural_network
```

The notebook verifier runs all seven notebooks with fresh kernels and saves executed copies under `outputs/executed/`. Scripts also save figures under `outputs/`. GitHub Actions runs the tests and notebooks on Linux/Python 3.12. The [first verification run](https://github.com/clembnl/ai-learning-path/actions/runs/36980655644) passed.

## Troubleshooting

- **Missing modules / wrong kernel:** select Python (ai-learning-path); rerun the launcher if needed.
- **Python too old:** use Python 3.11 or 3.12 explicitly, for example `python3.12 scripts/start.py`.
- **No compatible torch wheel:** check the official PyTorch installer for your OS/architecture. Modern Apple Silicon, Windows and Linux are the primary targets; older Intel Macs can require older compatible versions.
- **Imports fail:** keep `common/`, `steps/`, `data/` and `notebooks/` together. Download the whole repository, not individual notebooks.
- **Port already used:** JupyterLab normally selects another port; use the URL printed in the terminal.

## Layout and maintenance

```text
common/             shared data, metrics and plots
steps/              six runnable model implementations
notebooks/          seven guided notebooks
notebooks/content/  notebook source cells
scripts/            setup/launch and notebook verification
tests/              model and data smoke tests
data/students.csv   committed synthetic dataset
article/           English Medium draft, figures and publishing notes
```

Edit notebook content in `notebooks/content/`, then regenerate with `python notebooks/build_notebooks.py`. Regeneration clears outputs. The committed CSV is used by all steps; `python -m common.data` recreates it with the default seed.

Dependency ranges are in `requirements.txt`; `requirements-tested.txt` records the exact versions used for the local verification, if available. Ranges allow compatible updates; the snapshot is a reference for reproducing that run.

## License

Code and notebooks are available under the [MIT License](LICENSE).
