# Experiment and outputs

The question is whether filtering posed training images early in 3D Gaussian
splatting improves reconstruction quality on unfiltered novel views. The seven
methods are `none`, `downsample`, `mean`, `gaussian`, `bilateral`, `kuwahara`, and
`unsharp`.

Each dataset runs all seven methods with seeds 7, 17, and 27. Methods use the
same training views and a separate seeded training-camera sequence. Validation
and test images are never filtered. The default run trains for 8,000 steps;
filtered targets are used through step 2,000, followed by original targets.
`none` uses original targets throughout. Downsampling halves the warm-up image
size and intrinsics, then returns to full resolution. The phase switch keeps
the current Gaussians and optimizer state.

Use `config/real.yaml` or `config/synthetic.yaml` to change shared settings.
For a one-off run, pass `--method`, `--seed`, `--iterations`, or
`--phase-switch-step` to `synthetic_experiment.py`. The runner handles both
dataset formats despite its name. `scripts/run_synthetic_matrix.py --list`
shows the Slurm array index for each method and seed.

Preparation caches filtered training arrays in `outputs/preprocessed/` for
reuse across seeds. Each training run writes to
`outputs/<scene>/<method>/seed_<seed>/`, including `run_config.json`,
`validation_metrics.csv`, `test_metrics.csv`, `test_summary.json`, the final
PLY, and snapshots. Real-scene runs also write `split.json`.

`analyze_synthetic.py` reads completed runs and writes four CSVs in
`outputs/analysis/`:

- `paired_runs.csv`: each run's test PSNR and SSIM, change from its matched
  `none` baseline, training time, and final Gaussian count.
- `method_summary.csv`: per-scene, per-method means across paired seeds.
- `validation_curves.csv`: sampled validation metrics and elapsed time.
- `missing_jobs.csv`: runs that are absent or incomplete.

Analysis is complete when `missing_jobs.csv` contains only its header and the
expected 21 runs per scene have finished. To evaluate a bonsai PLY separately:

```sh
.venv/bin/python 3dgs_evaluator.py outputs/bonsai/none/seed_7/final.ply \
  data/bonsai --width 390 --height 260 --arch gpu --metrics psnr ssim
```
