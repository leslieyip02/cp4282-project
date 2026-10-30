"""Train and evaluate one screen-space splat kernel on a held-out scene."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
import warp as wp

from gpu_memory import GPUMemoryMonitor
from image_metrics import LPIPSMetric, ssim_rgb
from shared.kernels import NAMES
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


class ValidationReporter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("iteration", "view_id", "psnr_db", "ssim", "elapsed_seconds", "gaussians"))

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
                writer.writerow((iteration, view_id,
                                 psnr, ssim, elapsed, int(trainer.active.sum())))
        # Preserve the first fixed view in the image buffer for the trainer's snapshot.
        trainer.render_single_view_image(int(view_ids[0]))
        mean = np.mean(values, axis=0)
        print(f"{iteration:6d}: validation PSNR={mean[0]:.2f} dB, SSIM={mean[1]:.4f} "
              f"({len(values)} views)", flush=True)
        return float(np.mean(losses))


def _test_evaluation(trainer, data: Path, width: int, height: int, background,
                     run_dir: Path, training_summary, real_scene=None, lpips_metric=None,
                     lpips_net="none", gpu_memory_monitor=None):
    if real_scene is None:
        targets, cameras, focal, _, frame_paths = gs.load_views(
            data, width, height, background, manifest_name="transforms_test.json",
        )
    else:
        targets, cameras, focal, _, frame_paths = real_scene.load_views("test", width, height)
    trainer.reconfigure_views(targets, cameras, focal)
    rows = []
    if len(frame_paths):
        trainer.render_single_view_image(0)  # compile/warm up outside timing
    for view_id, frame_path in enumerate(frame_paths):
        timings = []
        for _ in range(5):
            started = perf_counter()
            prediction, mse = trainer.render_single_view_image(view_id)
            timings.append(perf_counter() - started)
        render_seconds = statistics.median(timings)
        prediction = np.clip(prediction, 0.0, 1.0)
        rows.append((frame_path, -10.0 * np.log10(max(float(mse), 1e-12)),
                     ssim_rgb(prediction, targets[view_id]),
                     lpips_metric(prediction, targets[view_id]) if lpips_metric else None,
                     render_seconds, trainer.last_pair_count))
    with (run_dir / "test_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "psnr_db", "ssim", "lpips", "render_seconds", "tile_pairs"))
        writer.writerows(rows)
    summary = {
        "views": len(rows),
        "mean_psnr_db": float(np.mean([row[1] for row in rows])),
        "mean_ssim": float(np.mean([row[2] for row in rows])),
        "mean_render_seconds": float(np.mean([row[4] for row in rows])),
        "mean_tile_pairs": float(np.mean([row[5] for row in rows])),
        "lpips_net": lpips_net,
        "kernel": trainer.kernel,
        "final_gaussians": int(trainer.active.sum()),
        **training_summary,
    }
    if lpips_metric is not None:
        summary["mean_lpips"] = float(np.mean([row[3] for row in rows]))
    if gpu_memory_monitor is not None:
        gpu_memory_monitor.sample()
        summary["peak_test_gpu_memory_mib"] = gpu_memory_monitor.peak_mib
        if gpu_memory_monitor.enabled and summary["peak_test_gpu_memory_mib"] is None:
            raise RuntimeError("NVML did not report GPU memory for the test phase")
    (run_dir / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lpips_text = (f", LPIPS ({lpips_net})={summary['mean_lpips']:.4f}"
                  if lpips_metric is not None else "")
    print(f"Final test: PSNR={summary['mean_psnr_db']:.2f} dB, "
          f"SSIM={summary['mean_ssim']:.4f}{lpips_text}, "
          f"Gaussians={summary['final_gaussians']:,}",
          flush=True)


def run(args):
    config = gs.TrainingConfig.load(args.config)
    values = config.values
    values["experiment"]["kernel"] = args.kernel or values["experiment"]["kernel"]
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
    if values["model"]["long_edge"] is not None and (args.width is not None or args.height is not None):
        if args.width is None or args.height is None:
            raise ValueError("Override both --width and --height when using model.long_edge.")
        values["model"]["long_edge"] = None
    if args.lpips_net is not None:
        values["reporting"]["lpips_net"] = args.lpips_net
    config.validate()
    lpips_net = values["reporting"]["lpips_net"]
    lpips_metric = LPIPSMetric(lpips_net) if lpips_net != "none" else None

    kernel = values["experiment"]["kernel"]
    data = (args.data or values["paths"]["data"]).resolve()
    scene = args.scene or data.name
    real_scene = RealScene(data, values["reporting"]["eval_views"]) if (data / "nb-info.json").exists() else None
    if values["model"]["long_edge"] is not None:
        if real_scene is None:
            raise ValueError("model.long_edge is supported only for Mip-NeRF 360 scenes.")
        values["model"]["width"], values["model"]["height"] = real_scene.render_size(
            values["model"]["long_edge"])
    init_ply = (
        args.init_ply
        or (values["paths"]["init_ply"] if args.data is None else None)
        or data / "init.ply"
    ).resolve()
    if not init_ply.exists() and real_scene is None:
        raise FileNotFoundError(f"Initial point cloud missing: {init_ply}")
    if real_scene is None:
        validate_synthetic_scene(data)
    run_dir = (args.output_root / "kernels" / scene / kernel /
               f"seed_{values['runtime']['seed']}").resolve()
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
    with GPUMemoryMonitor(enabled=device != "cpu") as memory:
        trainer = gs.WarpImageTrainer(
            full_targets, full_cameras, focal, model["scene_radius"] or (
                real_scene.scene_radius if real_scene is not None else distance * 0.35
            ),
            model["capacity"], model["initial_splats"], device, values["runtime"]["seed"],
            init_scene=initial, background=background,
            learning_rates=values["learning_rates"],
            tile_pair_capacity=model["tile_pair_capacity"],
            training_options={key: values[key] for key in
                              ("compact_box", "projection", "multiview_adc", "sparse")},
            kernel=kernel,
        )
        memory.sample()
        if memory.enabled and memory.peak_mib is None:
            raise RuntimeError("NVML did not report this process's GPU memory after setup")
        reporter = ValidationReporter(run_dir / "validation_metrics.csv")
        (run_dir / "snapshots").mkdir(exist_ok=True)
        training_summary = gs.run_training(
            trainer, config, training_targets, eval_ids,
            run_dir / "snapshots", run_dir / "final.png",
            description=f"{scene}/{kernel}/seed_{values['runtime']['seed']}",
            held_out_frame_ids=heldout_ids,
            metric_reporter=reporter,
        )
        memory.sample()
        training_summary["peak_train_gpu_memory_mib"] = memory.peak_mib
        training_summary["gpu_name"] = memory.gpu_name
        training_summary["gpu_memory_sample_interval_ms"] = int(memory.interval_seconds * 1000)
        training_summary["gpu_memory_measurement"] = "nvml_process_sampled_peak"
        memory.reset_peak()
        _test_evaluation(trainer, data, width, height, background, run_dir, training_summary,
                         real_scene, lpips_metric, lpips_net, memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/synthetic.yaml")
    parser.add_argument("--scene", help="Output label; defaults to the dataset directory name")
    parser.add_argument("--data", type=Path, help="NeRF-synthetic or Mip-NeRF 360 COLMAP scene directory")
    parser.add_argument("--init-ply", type=Path, help="Initial 3DGS PLY; defaults to scene/init.ply")
    parser.add_argument("--kernel", choices=NAMES)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arch", choices=("cpu", "gpu"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--lpips-net", choices=("vgg", "alex", "squeeze", "none"),
                        help="LPIPS backbone for final test; default from config (vgg).")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
