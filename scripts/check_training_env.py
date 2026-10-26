"""Fail early when a Slurm job uses a Python without training dependencies."""

from __future__ import annotations

import importlib
import sys


def main() -> None:
    print(f"Training Python: {sys.executable}", flush=True)
    for module in ("numpy", "scipy", "PIL", "plyfile", "yaml", "warp"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            raise SystemExit(
                f"Missing Python module {error.name!r} in {sys.executable}. "
                f"Install with: {sys.executable} -m pip install -r requirements.txt"
            ) from error
    import warp as wp
    print(f"Warp version: {wp.__version__}", flush=True)
    print("Training dependencies import successfully.", flush=True)


if __name__ == "__main__":
    main()
