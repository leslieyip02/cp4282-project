# Kernel-shape experiment

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
This is an exploratory correction motivated by the copied bonsai snapshots;
the earlier unguarded real-scene runs should be kept separate and all kernels
must be rerun under the new setting before drawing a comparison.
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

1. **Pilot:** Run all three kernels on one synthetic scene and one real scene with
   seed 7. Inspect renders, gradients, splat count, tile pairs, and timing.
   Resolve numerical or culling errors before the full matrix.
2. **Main matrix:** Run all three kernels on the eight NeRF-synthetic scenes,
   Bonsai, and Counter with seeds 7, 17, 27: 90 runs. If resources allow,
   add Bicycle and Garden to cover outdoor Mip-NeRF 360 scenes (108 runs total).
   Generate one `init.ply` per
   synthetic scene and reuse it across kernels and seeds. Real scenes use the
   same COLMAP initialization for matched runs. Hold image resolution, camera
   splits, optimizer, densification, 8,000 training steps, and the 100,000
   splat cap fixed within each scene. The cap is a budget, not a guarantee of
   equal actual splat counts.
3. **Metrics:** Use the final 8,000-step checkpoint for the primary comparison.
   Evaluate only on held-out test views. Record mean PSNR, SSIM, and LPIPS (VGG v0.1), splat count,
   training time, mean test-view render time, and tile pairs. Validation curves
   and saved snapshots diagnose convergence. If a validation-selected
   checkpoint becomes the primary endpoint, implement and predeclare that
   selection rule before examining test results.
4. **Analysis:** Pair each alternative with Gaussian by scene and seed. Report
   paired PSNR/SSIM/LPIPS deltas and per-scene 95% confidence intervals. Lower
   LPIPS is better. Treat the
   scene, rather than the individual view, as the unit for claims across
   scenes. Avoid interpreting a confidence interval crossing zero as proof of
   equivalence. Report both quality at equal steps and quality versus render
   time, since the kernels may perform different amounts of work. Freeze the
   degree at 2 for this matrix; a degree sweep would be a separate study with
   validation-based selection and fresh test evaluation.
5. **Longer-budget real-scene follow-up:** Use `config/real_20k.yaml` for all nine
   Mip-NeRF 360 scenes, all three kernels, and paired seeds 7, 17, and 27.
   Both training iterations and the position learning-rate schedule end at
   20,000 steps. Save to a new output root and analyze it separately from the
   8,000- and 16,000-step runs. This changes the schedule as well as the step
   count; it is a budget comparison, not an isolated extra-steps effect.

Run `scripts/run_kernel_matrix.py --list` for array indices. Outputs are in
`outputs/kernels/<scene>/<kernel>/seed_<seed>/`. `analyze_kernels.py` writes
`outputs/kernel_analysis/{paired_runs,kernel_summary,validation_curves,missing_jobs}.csv`.
Historical filter results in `outputs/analysis/` remain separate.
When changing the real-scene guard, save new runs to a fresh output root so
they cannot silently replace the unguarded runs.

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
