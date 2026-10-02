"""Generate the guided notebooks (one per step) with nbformat.

Run from the repo root:   python notebooks/build_notebooks.py
The notebooks import the shared code in common/ and steps/ so that the
narrative lives here and the tested implementation lives in the scripts.
Cell contents are defined in notebooks/content/*.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from content import nb00_data, nb01_linear, nb02_logistic, nb03_perceptron, nb04_sklearn, nb05_pytorch, nb06_nn  # noqa: E402

NOTEBOOKS = {
    "00_data.ipynb": nb00_data.CELLS,
    "01_linear_regression.ipynb": nb01_linear.CELLS,
    "02_logistic_regression.ipynb": nb02_logistic.CELLS,
    "03_perceptron.ipynb": nb03_perceptron.CELLS,
    "04_sklearn.ipynb": nb04_sklearn.CELLS,
    "05_pytorch.ipynb": nb05_pytorch.CELLS,
    "06_neural_network.ipynb": nb06_nn.CELLS,
}


def build(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "ai-learning-path",
        "display_name": "Python (ai-learning-path)",
        "language": "python",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells
    ]
    return notebook


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        path = HERE / name
        nbf.write(build(cells), path)
        print("wrote", path)


if __name__ == "__main__":
    main()
