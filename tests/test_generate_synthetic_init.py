"""Checks for reproducible and safe synthetic initialization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from generate_synthetic_init import generate
from shared.gaussian_set import GaussianSet


class GenerateSyntheticInitTests(unittest.TestCase):
    def test_seed_reproduces_valid_gaussians_and_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for name, seed in (("first", 11), ("second", 11), ("third", 12)):
                data = root / name
                data.mkdir()
                (data / "transforms_train.json").write_text("{}")
                path = generate(data, seed=seed, points=8)
                outputs.append(path.read_bytes())
                scene = GaussianSet.from_ply(path)
                self.assertEqual(scene.means.shape, (8, 3))
                self.assertTrue(np.all(np.abs(scene.means) <= 1.3))
                self.assertEqual(json.loads((data / "init.json").read_text())["seed"], seed)
            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(outputs[0], outputs[2])
            with self.assertRaises(FileExistsError):
                generate(root / "first", seed=12, points=8)
            self.assertEqual((root / "first/init.ply").read_bytes(), outputs[0])


if __name__ == "__main__":
    unittest.main()
