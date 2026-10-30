"""Fail early when a Slurm job uses a Python without training dependencies."""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also verify CUDA access")
    args = parser.parse_args()
    print(f"Training Python: {sys.executable}", flush=True)
    for module in ("numpy", "scipy.ndimage", "scipy.spatial", "PIL", "plyfile", "yaml", "warp"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            raise SystemExit(
                f"Missing Python module {error.name!r} in {sys.executable}. "
                f"Install with: {sys.executable} -m pip install -r requirements.txt"
            ) from error
        except (ImportError, AttributeError) as error:
            raise SystemExit(
                f"Cannot import {module!r} in {sys.executable}: {error}. "
                "The virtual environment may contain inconsistent NumPy/SciPy files. "
                "Check PYTHONPATH or reinstall the affected package in this environment."
            ) from error
    import warp as wp
    print(f"Warp version: {wp.__version__}", flush=True)
    if args.gpu:
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                if gpu_count:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            finally:
                pynvml.nvmlShutdown()
            if gpu_count < 1:
                raise RuntimeError("NVML found no NVIDIA GPUs")
        except Exception as error:
            raise SystemExit(f"GPU memory monitoring requires working nvidia-ml-py/NVML: {error}") from error
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
        wp.init()
        if not wp.is_cuda_available():
            wp.print_diagnostics()
            raise SystemExit("Warp cannot access a CUDA GPU in this job. Check the Slurm GPU allocation and NVIDIA driver above.")
        print("Warp CUDA device is available.", flush=True)
    print("Training dependencies import successfully.", flush=True)


if __name__ == "__main__":
    main()
