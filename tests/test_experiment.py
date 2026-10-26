"""Checks for filter semantics and the state-preserving phase transition."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from image_filters import METHODS, filter_image
from prepared_targets import prepared_training_views
from synthetic_experiment import run

gs = importlib.import_module("3dgs_trainer")


class FilterTests(unittest.TestCase):
    def test_constant_image_and_dimensions(self):
        image = np.full((8, 10, 3), 0.4, np.float32)
        for method in METHODS:
            with self.subTest(method=method):
                result = filter_image(image, method)
                self.assertEqual(result.shape, (4, 5, 3) if method == "downsample" else image.shape)
                np.testing.assert_allclose(result, 0.4, atol=1e-5)

    def test_half_resolution_uses_area_average(self):
        image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3) / 48.0
        expected = image.reshape(2, 2, 2, 2, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(filter_image(image, "downsample"), expected)

    def test_bilateral_preserves_a_high_contrast_edge(self):
        image = np.zeros((16, 16, 3), np.float32)
        image[:, 8:] = 1.0
        bilateral = filter_image(image, "bilateral")
        gaussian = filter_image(image, "gaussian")
        self.assertLess(bilateral[8, 7, 0], gaussian[8, 7, 0])

    def test_prepared_targets_change_when_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = np.zeros((1, 8, 8, 3), np.float32)
            first = prepared_training_views(image, "gaussian", "lego", Path(tmp))
            image[0, 3, 3] = 1.0
            second = prepared_training_views(image, "gaussian", "lego", Path(tmp))
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(len(list((Path(tmp) / "lego").glob("*.npy"))), 2)


class RunnerTests(unittest.TestCase):
    def test_downsample_warmup_switch_and_full_resolution_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "custom_scene"
            matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]
            for split in ("train", "val", "test"):
                (data / split).mkdir(parents=True)
                image = np.full((16, 16, 4), 255, np.uint8)
                image[4:12, 4:12, :3] = (100, 80, 60)
                Image.fromarray(image, "RGBA").save(data / split / "r_0.png")
                manifest = {"camera_angle_x": 0.7, "frames": [
                    {"file_path": f"./{split}/r_0", "transform_matrix": matrix},
                ]}
                (data / f"transforms_{split}.json").write_text(json.dumps(manifest))
            scene = gs.GaussianSet(
                means=np.zeros((1, 3), np.float32),
                scales=np.full((1, 3), 0.3, np.float32),
                rotations=np.array([[1, 0, 0, 0]], np.float32),
                opacities=np.array([0.5], np.float32),
                colors=np.array([[0.4, 0.3, 0.2]], np.float32),
            )
            scene.to_ply(data / "init.ply")
            args = argparse.Namespace(
                config=Path(__file__).resolve().parents[1] / "config/synthetic.yaml",
                scene=None, data=data, init_ply=None, method="downsample",
                phase_switch_step=1, iterations=2, seed=3, arch="cpu",
                width=16, height=16, eval_every=1, output_root=root / "outputs",
                cache_dir=None,
            )
            run(args)
            result = root / "outputs/custom_scene/downsample/seed_3"
            with (result / "validation_metrics.csv").open(newline="") as stream:
                import csv
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["iteration"]) for row in rows], [0, 1, 2])
            self.assertEqual([row["phase"] for row in rows], ["warmup", "warmup", "original"])
            self.assertTrue((result / "final.ply").exists())
            self.assertEqual(json.loads((result / "test_summary.json").read_text())["views"], 1)
            run_yaml = yaml.safe_load(args.config.read_text())
            run_yaml["paths"].update(data=str(data), init_ply=str(data / "init.ply"))
            run_yaml["model"].update(width=16, height=16)
            run_yaml["training"]["iterations"] = 2
            run_yaml["reporting"]["eval_every"] = 1
            run_yaml["runtime"]["seed"] = 3
            run_yaml["experiment"].update(method="none", phase_switch_step=0)
            config_path = root / "one_run.yaml"
            config_path.write_text(yaml.safe_dump(run_yaml))
            args.config = config_path
            args.data = None
            args.method = None
            args.phase_switch_step = None
            args.iterations = None
            args.seed = None
            args.width = None
            args.height = None
            args.eval_every = None
            run(args)
            with (root / "outputs/custom_scene/none/seed_3/validation_metrics.csv").open(newline="") as stream:
                baseline_rows = list(csv.DictReader(stream))
            self.assertEqual([row["phase"] for row in baseline_rows], ["original"] * 3)

    def test_analysis_pairs_same_scene_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method, psnr in (("none", 20.0), ("gaussian", 21.5)):
                directory = root / "runs/lego" / method / "seed_7"
                directory.mkdir(parents=True)
                (directory / "run_config.json").write_text(json.dumps({"training": {"iterations": 2}}))
                (directory / "test_summary.json").write_text(json.dumps({
                    "completed_iterations": 2, "mean_psnr_db": psnr,
                    "mean_ssim": 0.7, "training_elapsed_seconds": 1.0,
                    "final_gaussians": 10,
                }))
                with (directory / "validation_metrics.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("iteration", "psnr_db", "ssim", "elapsed_seconds"))
                    writer.writerow((2, psnr, 0.7, 1.0))
            subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "analyze_synthetic.py"),
                "--input-root", str(root / "runs"), "--output-dir", str(root / "analysis"),
            ], check=True, capture_output=True, text=True)
            with (root / "analysis/paired_runs.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            gaussian = next(row for row in rows if row["method"] == "gaussian")
            self.assertEqual(float(gaussian["delta_psnr_db"]), 1.5)
            with (root / "analysis/missing_jobs.csv").open(newline="") as stream:
                missing = list(csv.DictReader(stream))
            self.assertEqual(len(missing), 19)


if __name__ == "__main__":
    unittest.main()
