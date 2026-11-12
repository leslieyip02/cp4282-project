# Kernel-shape methodology

This file records the implemented comparison. The [run catalogue](experiments.md)
lists the completed and historical matrices; the [README](../README.md) gives
the commands to reproduce them.

## Question and scope

Under a fixed splat-capacity and training-step budget, how does the projected
screen-space kernel affect held-out reconstruction quality and rendering cost?
This changes the alpha footprint of projected 3D ellipsoids. Their 3D geometry,
projection, color model, and front-to-back compositing remain the same. It is a
screen-space kernel study, not a claim to replace the 3D primitive itself.

The implemented comparison is `gaussian`, `linear`, and `polynomial`.
The Gaussian weight is `exp(-q/2)`. The linear radial weight is
`max(0, 1 - sqrt(q / (8 ln 2)))`, with `q` the squared Mahalanobis distance in
the projected ellipse. The polynomial weight is
`max(0, 1 - q / R²)^2`, with `R² = 2 ln 2 / (1 - 1/sqrt(2))`.
Its degree is fixed at 2 for the main experiment. The trainer optimizes splat
parameters, not kernel degree. All three weights equal 1/2 at `q = 2 ln 2`.
This matches the
half-maximum radius, while leaving the total mass and support different; those
differences are part of what the experiment measures. The linear kernel has
finite support at `q = 8 ln 2`; polynomial has finite support at `q = R²`.
The existing alpha threshold and compact-box
setting can trim a faint splat before its half-maximum radius; training, tile
culling, ADC, and evaluation
use the same kernel-specific support rule. No image filtering or phase switch
is used.

Real scenes also set `projection.max_radius_fraction: 0.25`. A projected splat
is omitted from a view when either axis-aligned support radius exceeds 25% of
the larger image dimension. This guards against near-camera splats that form
large translucent color wedges. It is applied to all kernels in training,
validation, and test rendering. Synthetic scenes leave the guard disabled.
The guard is part of the real-scene configuration and is held fixed across
kernels within each real-scene matrix.
Real scenes use `model.long_edge: 390`; the short edge is derived from the
COLMAP camera aspect ratio and saved in `run_config.json`. All kernels and seeds
for a scene therefore use identical dimensions, while scenes with different
native aspect ratios retain their geometry.

## Implementation and verification

- `experiment.kernel` in the YAML selects `gaussian`, `linear`, or `polynomial` and is saved
  in every run's `run_config.json`.
- `3dgs_trainer.py` evaluates the kernel in dense and sparse forward/backward
  paths. `gaussian_first_tile_workspace_gpu.py` uses its support for tiling;
  the multi-view ADC scorer uses the matching CPU support estimate.
- `3dgs_evaluator.py` reads the saved run config when evaluating a final or
  snapshot PLY; `--kernel` and `--max-projected-radius-fraction` can override
  it for an external PLY or a diagnostic render of an old checkpoint.
- Run `WARP_CACHE_PATH=/tmp/warp-cache .venv/bin/python
  3dgs_gradient_check_gpu.py --device cpu` before cluster runs. It checks
  finite-difference gradients for all three kernels and the existing tiling checks.
  Run the same check on CUDA in the cluster environment.

A PLY does not encode the kernel. Keep it with its `run_config.json`, or pass
`--kernel` explicitly to the evaluator. The measured render time is the median
of five end-to-end single-view renders plus readback, after one warm-up render; tile
pairs count projected splat/tile records, not exact splat/pixel evaluations.
LPIPS is computed once per final test view, outside the render timing loop,
using the VGG backbone and version 0.1. It is saved beside PSNR and SSIM in
`test_metrics.csv` and averaged in `test_summary.json`; the analysis pairs only
LPIPS values produced with the same backbone. Existing final PLYs can be
backfilled with `scripts/backfill_lpips.py` without retraining.
New GPU runs also record `peak_train_gpu_memory_mib` and
`peak_test_gpu_memory_mib` in `test_summary.json`. These are per-process NVML
framebuffer-memory peaks sampled every 250 ms. Training sampling includes
trainer setup, validation, and snapshots; test sampling includes final test
rendering. The figures include the CUDA context and all live GPU allocations,
and can miss shorter transients. Compare them only on the same GPU model and
with the same image and splat budgets. Old runs have no memory measurement.

## Experimental procedure

1. **Synthetic matrix:** Eight NeRF-synthetic scenes, three kernels, and paired
   seeds 7, 17, and 27 produce 72 runs. Images are 256 × 256 on white; training
   lasts 8,000 steps with a 100,000-splat cap. Generate one `init.ply` per scene
   and reuse it for all nine runs. This matrix has no projected-radius guard.
2. **Real matrix:** Nine Mip-NeRF 360 scenes, the same kernels and seeds, produce
   81 runs. `config/real_20k_200k.yaml` uses a 390-pixel long edge, black
   background, 20,000 training steps, a 20,000-step position learning-rate
   schedule, a 200,000-splat cap, and the 0.25 projected-radius guard. Matched
   runs use the same scene COLMAP initialization and camera split. The cap is a
   budget; the achieved splat count is also reported.
3. **Evaluation:** Use the fixed final checkpoint of each matrix as its primary
   endpoint. Evaluate on held-out test views only. Save mean PSNR, SSIM, LPIPS
   (VGG v0.1), splat count, training time, test render time, and tile pairs.
   Reserved validation views and snapshots diagnose optimization, but do not
   select the reported checkpoint.
4. **Analysis:** Pair each alternative with Gaussian by scene and seed. Report
   PSNR/SSIM/LPIPS differences and seed variation by scene. Lower LPIPS is
   better. Test views within a scene are correlated; use scenes, not individual
   views, for cross-scene claims. Compare elapsed cost only on matched GPU
   models, and report tile pairs separately. A confidence interval crossing
   zero does not prove equivalence.

The synthetic and real matrices differ in several settings together, so their
different outcomes cannot identify the effect of scene type alone. Earlier
8k/100k, 16k/100k, and 20k/100k real runs remain separate; see the catalogue.
Save any new capacity, resolution, or schedule study to a distinct output root.

## Final presentation

- A scene-by-kernel table of test PSNR, SSIM, splat count, render time,
  LPIPS, training time, tile pairs, and sampled peak GPU memory.
- A paired-delta plot with one point per scene/seed and per-scene uncertainty.
- A quality-versus-render-time plot and validation curves.
- Matched image crops of edges and thin structures, with references and all
  kernel outputs.
- A concise conclusion scoped to these scenes, resolution, kernel definitions,
  cap, and training budget. Published linear-kernel work is prior art, so this
  is a controlled comparison in this renderer.
