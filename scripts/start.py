"""Create a local environment, install dependencies, and launch JupyterLab."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true", help="Install without starting JupyterLab")
    parser.add_argument("--skip-install", action="store_true", help="Start an already configured environment")
    parser.add_argument("--no-browser", action="store_true", help="Print a URL without opening a browser")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10+ required; Python 3.11 or 3.12 recommended.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating .venv ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    if not args.skip_install:
        run(python, "-m", "pip", "install", "-r", ROOT / "requirements.txt")
    # Kernel belongs to this environment; no global Jupyter configuration needed.
    run(python, "-m", "ipykernel", "install", "--prefix", env,
        "--name", "ai-learning-path", "--display-name", "Python (ai-learning-path)")
    if not args.setup_only:
        run(python, "-m", "jupyterlab", "--notebook-dir", ROOT / "notebooks",
            *(["--no-browser"] if args.no_browser else []))

if __name__ == "__main__":
    main()
