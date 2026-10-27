"""Run the controlled NeRF-synthetic filtering experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from image_filters import METHODS
from image_metrics import ssim_rgb
from prepared_targets import dataset_name, prepared_training_views
from real_scene import RealScene

gs = importlib.import_module("3dgs_trainer")


def validate_synthetic_scene(data: Path):
    """Check that train, validation, and test are usable held-out splits."""
    seen = set()
    train_angle = None
    for split in ("train", "val", "test"):
        manifest_path = data / f"transforms_{split}.json"
        manifest = json.loads(manifest_path.read_text())
        frames = manifest.get("frames", [])
        if not frames:
            raise ValueError(f"{manifest_path} contains no views")
        angle = float(manifest["camera_angle_x"])
        if train_angle is None:
            train_angle = angle
        elif not np.isclose(angle, train_angle, rtol=0, atol=1e-7):
            raise ValueError("Synthetic splits must use the same camera_angle_x")
        current = {gs.resolve_frame_image(data, frame["file_path"]).resolve() for frame in frames}
        if len(current) != len(frames) or current & seen:
            raise ValueError("Synthetic train/val/test manifests reuse an image path")
        seen |= current


class PhaseSchedule:
    """Own only the training target/resolution switch, never model state."""

    def __init__(self, trainer, method, switch_step, warm, full):
        self.trainer = trainer
        self.method = method
        self.switch_step = switch_step
        self.warm = warm
        self.full = full
        self.in_warmup = method != "none" and switch_step > 0

    def before_step(self, iteration):
        if self.in_warmup and iteration == self.switch_step + 1:
            optimizers = self.trainer.optimizers
            trainable = self.trainer.trainable
            adam_step = optimizers.quaternion_adam_t
            self.trainer.reconfigure_views(*self.full)
            if (self.trainer.optimizers is not optimizers
                    or self.trainer.trainable is not trainable
                    or optimizers.quaternion_adam_t != adam_step):
                raise RuntimeError("Phase switch changed splat or optimizer state")
            self.in_warmup = False
            print(f"Step {iteration}: original-target refinement begins; Adam state preserved.", flush=True)

    @contextmanager
    def evaluation_context(self):
        # Spatial filters do not alter held-out views or image dimensions.
        if not self.in_warmup or self.method != "downsample":
            yield
            return
        self.trainer.reconfigure_views(*self.full)
        try:
            yield
        finally:
            self.trainer.reconfigure_views(*self.warm)

    def phase_at(self, iteration):
        return "warmup" if self.method != "none" and iteration <= self.switch_step else "original"


class ValidationReporter:
    def __init__(self, path: Path, schedule: PhaseSchedule):
        self.path = path
        self.schedule = schedule
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("iteration", "phase", "view_id", "psnr_db", "ssim", "elapsed_seconds", "gaussians"))

    def __call__(self, iteration, trainer, view_ids, elapsed):
        values = []
        losses = []
        with self.path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            for view_id in view_ids:
                view_id = int(view_id)
                prediction, mse = trainer.render_single_view_image(view_id)
                prediction = np.clip(prediction, 0.0, 1.0)
                psnr = gs.psnr_from_mse(mse) if hasattr(gs, "psnr_from_mse") else -10.0 * np.log10(max(mse, 1e-12))
                ssim = ssim_rgb(prediction, trainer.targets_host[view_id])
                values.append((float(psnr), float(ssim)))
                losses.append(float(mse))
                writer.writerow((iteration, self.schedule.phase_at(iteration), view_id,
                                 psnr, ssim, elapsed, int(trainer.active.sum())))
        # Preserve the first fixed view in the image buffer for the trainer's snapshot.
        trainer.render_single_view_image(int(view_ids[0]))
        mean = np.mean(values, axis=0)
        print(f"{iteration:6d}: validation PSNR={mean[0]:.2f} dB, SSIM={mean[1]:.4f} "
              f"({self.schedule.phase_at(iteration)}, {len(values)} views)", flush=True)
        return float(np.mean(losses))


def _test_evaluation(trainer, data: Path, width: int, height: int, background,
                     run_dir: Path, training_summary, real_scene=None):
    if real_scene is None:
        targets, cameras, focal, _, frame_paths = gs.load_views(
            data, width, height, background, manifest_name="transforms_test.json",
        )
    else:
        targets, cameras, focal, _, frame_paths = real_scene.load_views("test", width, height)
    trainer.reconfigure_views(targets, cameras, focal)
    rows = []
    for view_id, frame_path in enumerate(frame_paths):
        prediction, mse = trainer.render_single_view_image(view_id)
        prediction = np.clip(prediction, 0.0, 1.0)
        rows.append((frame_path, -10.0 * np.log10(max(float(mse), 1e-12)),
                     ssim_rgb(prediction, targets[view_id])))
    with (run_dir / "test_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "psnr_db", "ssim"))
        writer.writerows(rows)
    summary = {
        "views": len(rows),
        "mean_psnr_db": float(np.mean([row[1] for row in rows])),
        "mean_ssim": float(np.mean([row[2] for row in rows])),
        "final_gaussians": int(trainer.active.sum()),
        **training_summary,
    }
    (run_dir / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Final test: PSNR={summary['mean_psnr_db']:.2f} dB, "
          f"SSIM={summary['mean_ssim']:.4f}, Gaussians={summary['final_gaussians']:,}",
          flush=True)


def run(args):
    config = gs.TrainingConfig.load(args.config)
    values = config.values
    values["experiment"]["method"] = args.method or values["experiment"]["method"]
    if args.phase_switch_step is not None:
        values["experiment"]["phase_switch_step"] = args.phase_switch_step
    if args.iterations is not None:
        values["training"]["iterations"] = args.iterations
    if args.seed is not None:
        values["runtime"]["seed"] = args.seed
    if args.arch is not None:
        values["runtime"]["arch"] = args.arch
    if args.eval_every is not None:
        values["reporting"]["eval_every"] = args.eval_every
    if args.width is not None:
        values["model"]["width"] = args.width
    if args.height is not None:
        values["model"]["height"] = args.height
    config.validate()

    method = values["experiment"]["method"]
    switch = values["experiment"]["phase_switch_step"]
    data = (args.data or values["paths"]["data"]).resolve()
    scene = dataset_name(data, args.scene)
    real_scene = RealScene(data, values["reporting"]["eval_views"]) if (data / "nb-info.json").exists() else None
    init_ply = (
        args.init_ply
        or (values["paths"]["init_ply"] if args.data is None else None)
        or data / "init.ply"
    ).resolve()
    if not init_ply.exists() and real_scene is None:
        raise FileNotFoundError(f"Initial point cloud missing: {init_ply}")
    if real_scene is None:
        validate_synthetic_scene(data)
    run_dir = (args.output_root / scene / method / f"seed_{values['runtime']['seed']}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = "cpu" if values["runtime"]["arch"] == "cpu" else "cuda:0"
    if device != "cpu" and not wp.is_cuda_available():
        raise RuntimeError("runtime.arch=gpu requires CUDA")
    model = values["model"]
    width, height = model["width"], model["height"]
    background = np.ones(3, np.float32) if model["background"] == "white" else np.zeros(3, np.float32)
    if real_scene is None:
        training_targets, training_cameras, focal, distance, _ = gs.load_views(
            data, width, height, background,
        )
        full_targets, full_cameras, eval_ids, heldout_ids = gs.resolve_evaluation_views(
            training_targets, training_cameras, data, background, values["reporting"],
        )
    else:
        training_targets, training_cameras, train_intrinsics, distance, _ = real_scene.load_views(
            "train", width, height,
        )
        val_targets, val_cameras, val_intrinsics, _, _ = real_scene.load_views(
            "val", width, height,
        )
        full_targets = np.concatenate((training_targets, val_targets))
        full_cameras = np.concatenate((training_cameras, val_cameras))
        focal = np.concatenate((train_intrinsics, val_intrinsics))
        eval_ids = np.arange(len(training_targets), len(full_targets), dtype=np.int32)
        heldout_ids = list(range(len(val_targets)))
        values["reporting"]["eval_manifest"] = "reserved training views"
    full = (full_targets, full_cameras, focal)
    warm = full
    if method != "none" and switch > 0:
        filtered = prepared_training_views(
            training_targets, method, scene,
            args.cache_dir or args.output_root / "preprocessed",
        )
        if method == "downsample":
            if real_scene is None:
                low_targets, low_cameras, low_ids, _ = gs.resolve_evaluation_views(
                    filtered, training_cameras, data, background, values["reporting"],
                )
                if not np.array_equal(low_ids - len(training_targets), eval_ids - len(training_targets)):
                    raise RuntimeError("Warm-up and full-resolution validation views differ")
                warm = (low_targets, low_cameras, focal / 2.0)
            else:
                low_val, low_val_cameras, low_val_intrinsics, _, _ = real_scene.load_views(
                    "val", width // 2, height // 2,
                )
                warm = (np.concatenate((filtered, low_val)),
                        np.concatenate((training_cameras, low_val_cameras)),
                        np.concatenate((train_intrinsics / 2.0, low_val_intrinsics)))
        else:
            warm = (np.concatenate((filtered, full_targets[len(training_targets):])),
                    full_cameras, focal)
    if init_ply.exists():
        initial = gs.GaussianSet.from_ply(init_ply)
    else:
        initial = real_scene.initial_gaussians(model["initial_splats"])
        init_ply = run_dir / "colmap_init.ply"
        initial.to_ply(init_ply)
    values["paths"].update(data=data, init_ply=init_ply, output=run_dir / "final.png")
    (run_dir / "run_config.json").write_text(json.dumps(config.printable(), indent=2) + "\n")
    if real_scene is not None:
        (run_dir / "split.json").write_text(json.dumps({
            "train": real_scene.train_names, "validation": real_scene.validation_names,
            "test": real_scene.test_names,
        }, indent=2) + "\n")
    if len(initial.means) > model["capacity"]:
        raise ValueError("Initial point cloud exceeds model.capacity")
    trainer = gs.WarpImageTrainer(
        *warm[:2], warm[2], model["scene_radius"] or (
            real_scene.scene_radius if real_scene is not None else distance * 0.35
        ),
        model["capacity"], model["initial_splats"], device, values["runtime"]["seed"],
        init_scene=initial, background=background,
        learning_rates=values["learning_rates"],
        tile_pair_capacity=model["tile_pair_capacity"],
        training_options={key: values[key] for key in ("compact_box", "multiview_adc", "sparse")},
    )
    schedule = PhaseSchedule(trainer, method, switch, warm, full)
    reporter = ValidationReporter(run_dir / "validation_metrics.csv", schedule)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    training_summary = gs.run_training(
        trainer, config, training_targets, eval_ids,
        run_dir / "snapshots", run_dir / "final.png",
        description=f"{scene}/{method}/seed_{values['runtime']['seed']}",
        held_out_frame_ids=heldout_ids, phase_schedule=schedule,
        metric_reporter=reporter,
    )
    _test_evaluation(trainer, data, width, height, background, run_dir, training_summary,
                     real_scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/synthetic.yaml")
    parser.add_argument("--scene", help="Output label; defaults to the dataset directory name")
    parser.add_argument("--data", type=Path, help="NeRF-synthetic or Mip-NeRF 360 COLMAP scene directory")
    parser.add_argument("--init-ply", type=Path, help="Initial 3DGS PLY; defaults to scene/init.ply")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--phase-switch-step", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arch", choices=("cpu", "gpu"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--cache-dir", type=Path, help="Prepared target copies; defaults to output-root/preprocessed")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
