"""Execute every notebook with a fresh kernel; save results under outputs/executed."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs" / "executed"
DEST.mkdir(parents=True, exist_ok=True)
paths = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook", choices=[p.name for p in paths], help="Verify one notebook")
args = parser.parse_args()
for path in paths:
    if args.notebook and path.name != args.notebook:
        continue
    print(f"Executing {path.name} ...", flush=True)
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(notebook, DEST / path.name)
    print(f"OK {path.name}", flush=True)
