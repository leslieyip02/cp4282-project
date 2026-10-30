"""GPU memory sampling keeps phase peaks separate and counts this PID only."""

import os
from types import SimpleNamespace
import unittest

from gpu_memory import GPUMemoryMonitor, MIB


class FakeNVML:
    NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1

    def __init__(self):
        self.used = 100 * MIB
        self.initialized = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.initialized = False

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return b"Fake GPU"

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=os.getpid(), usedGpuMemory=self.used),
                SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=900 * MIB)]


class MemoryMonitorTests(unittest.TestCase):
    def test_samples_only_own_pid_and_resets_between_phases(self):
        fake = FakeNVML()
        with GPUMemoryMonitor(enabled=True, interval_seconds=60, nvml=fake) as monitor:
            self.assertEqual(monitor.peak_mib, 100)
            self.assertEqual(monitor.gpu_name, "Fake GPU")
            fake.used = 240 * MIB
            monitor.sample()
            self.assertEqual(monitor.peak_mib, 240)
            fake.used = 150 * MIB
            monitor.reset_peak()
            self.assertEqual(monitor.peak_mib, 150)
        self.assertFalse(fake.initialized)

    def test_cpu_run_does_not_use_nvml(self):
        with GPUMemoryMonitor(enabled=False) as monitor:
            monitor.sample()
            self.assertIsNone(monitor.peak_mib)


if __name__ == "__main__":
    unittest.main()
