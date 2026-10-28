# Kernel-shape experiment

## Question and scope

Under a fixed splat-capacity and training-step budget, how does the projected
screen-space kernel affect held-out reconstruction quality and rendering cost?
This changes the alpha footprint of projected 3D ellipsoids. Their 3D geometry,
projection, color model, and front-to-back compositing remain the same. It is a
screen-space kernel study, not a claim to replace the 3D primitive itself.

The implemented comparison is `gaussian` versus `linear`. A smooth polynomial
kernel is a possible later extension and is not included in this run matrix.
The Gaussian weight is `exp(-q/2)`. The linear radial weight is
`max(0, 1 - sqrt(q / (8 ln 2)))`, with `q` the squared Mahalanobis distance in
the projected ellipse. Both weights equal 1/2 at `q = 2 ln 2`. This matches the
half-maximum radius, while leaving the total mass and support different; those
differences are part of what the experiment measures. The linear kernel has
finite support at `q = 8 ln 2`. The existing alpha threshold and compact-box
setting can trim a faint splat before its half-maximum radius; training, tile
culling, ADC, and evaluation
use the same kernel-specific support rule. No image filtering or phase switch
is used.

Real scenes also set `projection.max_radius_fraction: 0.25`. A projected splat
is omitted from a view when either axis-aligned support radius exceeds 25% of
the larger image dimension. This guards against near-camera splats that form
large translucent color wedges. It is applied to both kernels in training,
validation, and test rendering. Synthetic scenes leave the guard disabled.
This is an exploratory correction motivated by the copied bonsai snapshots;
the earlier unguarded real-scene runs should be kept separate and both kernels
must be rerun under the new setting before drawing a comparison.

## Implementation and verification

- `experiment.kernel` in the YAML selects `gaussian` or `linear` and is saved
  in every run's `run_config.json`.
- `3dgs_trainer.py` evaluates the kernel in dense and sparse forward/backward
  paths. `gaussian_first_tile_workspace_gpu.py` uses its support for tiling;
  the multi-view ADC scorer uses the matching CPU support estimate.
- `3dgs_evaluator.py` reads the saved run config when evaluating a final or
  snapshot PLY; `--kernel` and `--max-projected-radius-fraction` can override
  it for an external PLY or a diagnostic render of an old checkpoint.
- Run `WARP_CACHE_PATH=/tmp/warp-cache .venv/bin/python
  3dgs_gradient_check_gpu.py --device cpu` before cluster runs. It checks
  finite-difference gradients for both kernels and the existing tiling checks.
  Run the same check on CUDA in the cluster environment.

A PLY does not encode the kernel. Keep it with its `run_config.json`, or pass
`--kernel` explicitly to the evaluator. The measured render time is the median
of five end-to-end single-view renders plus readback, after one warm-up render; tile
pairs count projected splat/tile records, not exact splat/pixel evaluations.

## Experimental procedure

1. **Pilot:** Run both kernels on one synthetic scene and one real scene with
   seed 7. Inspect renders, gradients, splat count, tile pairs, and timing.
   Resolve numerical or culling errors before the full matrix.
2. **Main matrix:** Run both kernels on Lego, Materials, Ship, Bonsai, and
   Counter, with seeds 7, 17, 27: 30 runs. Generate one `init.ply` per
   synthetic scene and reuse it across kernels and seeds. Real scenes use the
   same COLMAP initialization for matched runs. Hold image resolution, camera
   splits, optimizer, densification, 8,000 training steps, and the 100,000
   splat cap fixed within each scene. The cap is a budget, not a guarantee of
   equal actual splat counts.
3. **Metrics:** Use the final 8,000-step checkpoint for the primary comparison.
   Evaluate only on held-out test views. Record mean PSNR and SSIM, splat count,
   training time, mean test-view render time, and tile pairs. Validation curves
   and saved snapshots diagnose convergence. If a validation-selected
   checkpoint becomes the primary endpoint, implement and predeclare that
   selection rule before examining test results.
4. **Analysis:** Pair the linear and Gaussian runs by scene and seed. Report
   paired PSNR/SSIM deltas and per-scene 95% confidence intervals. Treat the
   scene, rather than the individual view, as the unit for claims across
   scenes. Avoid interpreting a confidence interval crossing zero as proof of
   equivalence. Report both quality at equal steps and quality versus render
   time, since the kernels may perform different amounts of work.

Run `scripts/run_kernel_matrix.py --list` for array indices. Outputs are in
`outputs/kernels/<scene>/<kernel>/seed_<seed>/`. `analyze_kernels.py` writes
`outputs/kernel_analysis/{paired_runs,kernel_summary,validation_curves,missing_jobs}.csv`.
Historical filter results in `outputs/analysis/` remain separate.
When changing the real-scene guard, save new runs to a fresh output root so
they cannot silently replace the unguarded runs.

## Final presentation

- A scene-by-kernel table of test PSNR, SSIM, splat count, render time,
  training time, and tile pairs.
- A paired-delta plot with one point per scene/seed and per-scene uncertainty.
- A quality-versus-render-time plot and validation curves.
- Matched image crops of edges and thin structures, with references and both
  kernel outputs.
- A concise conclusion scoped to these scenes, resolution, kernel definitions,
  cap, and training budget. Published linear-kernel work is prior art, so this
  is a controlled comparison in this renderer.
