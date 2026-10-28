"""Kernel matching and culling support invariants."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from kernels import GAUSSIAN, LINEAR, LINEAR_RADIUS_SQUARED, support_radius_squared
from training_config import TrainingConfig


class KernelTests(unittest.TestCase):
    def test_half_maximum_radius(self):
        q = 2.0 * math.log(2.0)
        self.assertAlmostEqual(math.exp(-q / 2.0), 0.5)
        self.assertAlmostEqual(1.0 - math.sqrt(q / LINEAR_RADIUS_SQUARED), 0.5)

    def test_linear_support_is_finite_and_alpha_aware(self):
        compact = {"enabled": True, "beta": 0.5, "alpha_min": 1.0 / 255.0}
        opacity = np.array([0.001, 0.01, 0.5], np.float32)
        support = support_radius_squared(opacity, LINEAR, compact, 9.0, 1.0 / 255.0)
        self.assertEqual(support[0], 0.0)
        self.assertGreater(support[2], support[1])
        self.assertLessEqual(support[2], LINEAR_RADIUS_SQUARED)
        compact["enabled"] = False
        np.testing.assert_allclose(
            support_radius_squared(opacity, LINEAR, compact, 9.0, 1.0 / 255.0),
            LINEAR_RADIUS_SQUARED,
        )
        np.testing.assert_allclose(
            support_radius_squared(opacity, GAUSSIAN, compact, 9.0, 1.0 / 255.0),
            9.0,
        )

    def test_config_rejects_unknown_kernel(self):
        values = TrainingConfig._merge(TrainingConfig.DEFAULTS,
                                       {"experiment": {"kernel": "triangle"}}, "")
        with self.assertRaisesRegex(ValueError, "experiment.kernel"):
            TrainingConfig(values).validate()


if __name__ == "__main__":
    unittest.main()
