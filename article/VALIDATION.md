# Local verification — 2026-10-02

Environment: macOS Apple Silicon, Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, PyTorch 2.14.1, Matplotlib 3.11.2.

- 11 existing pytest tests passed.
- All seven notebooks executed end to end in fresh kernels.
- After replacing deprecated scikit-learn `penalty=None` with `C=np.inf`, the corresponding model test and notebook were verified again.
- The launcher started JupyterLab successfully; its HTTP API served the seven notebooks, then the test server was stopped.
- The launcher started JupyterLab successfully; its HTTP API served the seven notebooks, then the test server was stopped.
- The launcher installed the `ai-learning-path` kernel in `.venv/share/jupyter/kernels/`.
- Linear regression test R²: 0.786.
- Perceptron: 131/150 correct (87.3%).
- Logistic regression: 134/150 correct (89.3%).
- Logistic regression with squared sleep: 144/150 correct (96.0%).
- MLP: 143/150 correct (95.3%).

Figures in `article/assets/` were extracted from the executed notebooks, not from pre-existing exported plots.

The full dependency snapshot is `requirements-tested.txt`. The local install initially used the supplied requirements; all installed direct dependencies also satisfy the updated bounded ranges. Automatic GUI browser opening and Windows execution were not verified locally.

## GitHub verification

The public repository is published at https://github.com/clembnl/ai-learning-path under the MIT License. The [first GitHub Actions run](https://github.com/clembnl/ai-learning-path/actions/runs/36980655644) completed successfully on Linux/Python 3.12: dependency installation, the model tests and execution of all seven notebooks passed. Code commit: `3ef664f`.
