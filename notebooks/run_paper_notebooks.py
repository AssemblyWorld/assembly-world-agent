"""Execute the four paper notebooks using this project's Python kernel."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient


def execute(path):
    notebook = nbformat.read(path, as_version=4)
    for i, cell in enumerate(notebook.cells):
        cell["id"] = hashlib.sha256(f"{path.name}:{i}".encode()).hexdigest()[:12]
    manager = AsyncKernelManager(kernel_name="python3", ip="127.0.0.1")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=7200,
        shutdown_kernel="immediate",
        resources={"metadata": {"path": str(path.parent.parent)}},
    )
    try:
        client.execute()
    finally:
        if client.km is not None:
            client._cleanup_kernel()
    nbformat.write(notebook, path)
    print(f"Executed {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.environ.setdefault("IPYTHONDIR", str(root / ".cache/paper-analysis/ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache/paper-analysis/jupyter"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache/paper-analysis/xdg"))
    names = args.names or [
        "tool_call_timeline.ipynb",
        "paper_results.ipynb",
        "paper_qualitative.ipynb",
        "paper_refinement.ipynb",
    ]
    for name in names:
        execute(root / name)
