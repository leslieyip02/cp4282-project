"""Sample this process's NVIDIA GPU memory during a training or test phase."""

from __future__ import annotations

import os
from threading import Event, Lock, Thread


MIB = 1024 * 1024


class GPUMemoryMonitor:
    """Track sampled peak per-process GPU memory through NVML.

    NVML reports all framebuffer memory attributed to this PID, including CUDA
    context, Warp arrays, and temporary allocations. A short transient between
    samples can be missed, so this is a sampled peak rather than an exact peak.
    """

    def __init__(self, enabled: bool, interval_seconds: float = 0.25, nvml=None):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.nvml = nvml
        self.pid = os.getpid()
        self.handles = []
        self.device_names = []
        self._active_names = set()
        self._stop = Event()
        self._lock = Lock()
        self._thread = None
        self._peak_bytes = 0
        self._samples = 0
        self._error = None

    def __enter__(self):
        if not self.enabled:
            return self
        if self.nvml is None:
            try:
                import pynvml
            except ImportError as error:
                raise RuntimeError("GPU memory measurement needs nvidia-ml-py from requirements.txt") from error
            self.nvml = pynvml
        self.nvml.nvmlInit()
        try:
            self.handles = [self.nvml.nvmlDeviceGetHandleByIndex(index)
                            for index in range(self.nvml.nvmlDeviceGetCount())]
            if not self.handles:
                raise RuntimeError("NVML found no NVIDIA GPUs")
            self.device_names = [self.nvml.nvmlDeviceGetName(handle)
                                 for handle in self.handles]
            self.sample()
            self._thread = Thread(target=self._poll, name="gpu-memory-monitor", daemon=True)
            self._thread.start()
        except BaseException:
            self.nvml.nvmlShutdown()
            raise
        return self

    def _poll(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()
                break

    def sample(self):
        if not self.enabled:
            return
        if self._error is not None:
            raise RuntimeError("NVML GPU memory sampling failed") from self._error
        total = 0
        found = False
        unavailable = getattr(self.nvml, "NVML_VALUE_NOT_AVAILABLE", None)
        active_names = set()
        for handle, name in zip(self.handles, self.device_names):
            for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid != self.pid:
                    continue
                used = process.usedGpuMemory
                if used is None or used == unavailable:
                    continue
                total += int(used)
                found = True
                active_names.add(name.decode() if isinstance(name, bytes) else str(name))
        if found:
            with self._lock:
                self._peak_bytes = max(self._peak_bytes, total)
                self._samples += 1
                self._active_names.update(active_names)

    @property
    def peak_mib(self):
        with self._lock:
            return self._peak_bytes / MIB if self._samples else None

    @property
    def samples(self):
        with self._lock:
            return self._samples

    @property
    def gpu_name(self):
        with self._lock:
            return ", ".join(sorted(self._active_names)) if self._active_names else None

    def reset_peak(self):
        with self._lock:
            self._peak_bytes = 0
            self._samples = 0
        self.sample()

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            if exc_type is None:
                self.sample()
        finally:
            self.nvml.nvmlShutdown()
        return False
