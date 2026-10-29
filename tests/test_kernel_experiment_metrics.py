"""Final test reporting keeps LPIPS aligned with the other view metrics."""

import csv
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


experiment = importlib.import_module("kernel_experiment")
backfill = importlib.import_module("scripts.backfill_lpips")
analysis = importlib.import_module("analyze_kernels")


class FakeTrainer:
    kernel = "polynomial"
    last_pair_count = 12
    active = np.array([True, True, False])

    def reconfigure_views(self, targets, cameras, focal):
        self.targets = targets

    def render_single_view_image(self, view_id):
        value = (0.25, 0.5)[view_id]
        return np.full((16, 16, 3), value, np.float32), value * value


class TestFinalMetrics(unittest.TestCase):
    def test_lpips_is_written_per_view_and_in_summary(self):
        targets = np.zeros((2, 16, 16, 3), np.float32)
        views = (targets, None, None, None, ["a.png", "b.png"])
        with TemporaryDirectory() as directory, patch.object(
            experiment.gs, "load_views", return_value=views
        ):
            run_dir = Path(directory)
            experiment._test_evaluation(
                FakeTrainer(), run_dir, 16, 16, np.zeros(3), run_dir,
                {"completed_iterations": 2},
                lpips_metric=lambda prediction, target: float(np.mean(prediction - target)),
                lpips_net="vgg",
            )
            with (run_dir / "test_metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            summary = json.loads((run_dir / "test_summary.json").read_text())
            self.assertEqual([float(row["lpips"]) for row in rows], [0.25, 0.5])
            self.assertAlmostEqual(summary["mean_lpips"], 0.375)
            self.assertEqual(summary["lpips_net"], "vgg")
            self.assertEqual(summary["final_gaussians"], 2)

    def test_backfill_updates_existing_metrics_without_training(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_config.json").write_text(json.dumps({
                "model": {"width": 16, "height": 16, "background": "black"},
            }))
            (run_dir / "test_summary.json").write_text(json.dumps({"mean_psnr_db": 20.0}))
            (run_dir / "test_metrics.csv").write_text(
                "frame,psnr_db,ssim,render_seconds,tile_pairs\n"
                "a.png,20,0.5,0.01,12\nb.png,21,0.6,0.02,13\n"
            )

            def fake_evaluator(command, check):
                self.assertTrue(check)
                csv_path = Path(command[command.index("--csv") + 1])
                csv_path.write_text("view,file,mse,lpips\n0,a.png,0.1,0.2\n1,b.png,0.1,0.4\n")

            with patch.object(backfill.subprocess, "run", side_effect=fake_evaluator):
                backfill.backfill(run_dir, run_dir, "cpu")
            with (run_dir / "test_metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            summary = json.loads((run_dir / "test_summary.json").read_text())
            self.assertEqual([float(row["lpips"]) for row in rows], [0.2, 0.4])
            self.assertAlmostEqual(summary["mean_lpips"], 0.3)
            self.assertEqual(summary["lpips_net"], "vgg")
            with patch.object(backfill.subprocess, "run") as evaluator:
                backfill.backfill(run_dir, run_dir, "cpu")
                evaluator.assert_not_called()

    def test_analysis_pairs_lpips_with_matching_backbone(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for kernel, lpips_value in (("gaussian", 0.4), ("polynomial", 0.3)):
                run = root / "kernels" / "demo" / kernel / "seed_7"
                run.mkdir(parents=True)
                (run / "run_config.json").write_text(json.dumps({
                    "training": {"iterations": 2},
                    "experiment": {"kernel": kernel},
                    "projection": {"max_radius_fraction": 0.0},
                }))
                (run / "test_summary.json").write_text(json.dumps({
                    "completed_iterations": 2, "mean_psnr_db": 20.0,
                    "mean_ssim": 0.5, "mean_lpips": lpips_value,
                    "lpips_net": "vgg", "mean_render_seconds": 0.01,
                    "mean_tile_pairs": 10, "training_elapsed_seconds": 1,
                    "final_gaussians": 2,
                }))
                (run / "validation_metrics.csv").write_text(
                    "iteration,psnr_db,ssim\n2,20,0.5\n"
                )
            output = root / "analysis"
            argv = ["analyze_kernels.py", "--input-root", str(root / "kernels"),
                    "--output-dir", str(output), "--scenes", "demo", "--seeds", "7"]
            with patch.object(sys, "argv", argv):
                analysis.main()
            with (output / "paired_runs.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            polynomial = next(row for row in rows if row["kernel"] == "polynomial")
            self.assertAlmostEqual(float(polynomial["delta_lpips"]), -0.1)
            self.assertEqual(polynomial["lpips_net"], "vgg")


if __name__ == "__main__":
    unittest.main()
