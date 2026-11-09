# Alternative screen-space kernels: project TODO

## Main objective

Explore how Gaussian, linear radial, and degree-2 polynomial **screen-space splat weights** affect held-out reconstruction quality, optimization behavior, and rendering cost in this renderer. This is an exploratory comparison, not a claim of a new kernel or a universally best one. The 3D ellipsoids, projection, color model, and compositing stay the same; the weight assigned to pixels within each projected splat changes.

**Final deliverable:** two reproducible, separately analyzed comparisons: eight NeRF-synthetic scenes at 100,000 splats and 8,000 steps, and nine real Mip-NeRF 360 scenes at 200,000 splats and 20,000 steps. Accompany them with a short presentation and a synchronized visual demo. Report benefits, losses, and uncertainty by scene and seed. The two matrices also differ in resolution, background, initialization, and scene type, so any difference between their outcomes is descriptive rather than an isolated effect of scene type.

## 1. Establish context and definitions

- [ ] **Literature matrix:** Read [3D Gaussian Splatting](https://doi.org/10.1145/3592433), [3D Linear Splatting](https://arxiv.org/abs/2411.12440), [Deformable Radial Kernel Splatting](https://arxiv.org/abs/2412.11752), and [From ex(p) to poly](https://arxiv.org/abs/2603.18707). Produce a one-page table of each paper's kernel, other pipeline changes, datasets, quality measures, and speed results. Note where this project's exact kernel definitions and evaluation differ.
- [ ] **Research questions and hypotheses:** Write a short statement of the expected quality, stability, and cost tradeoffs before examining the remaining scenes. Treat these as hypotheses, not promised outcomes.
- [x] **Kernel explainer:** The [Kernel Atlas page](../site/index.html) shows the exact weight curves, shared half-maximum radius, 2D footprints, 3D surfaces, and projected-ellipse distance.

## 2. Complete the 200k real-scene comparison

- [ ] **Stump and treehill runs:** Complete Gaussian, linear, and polynomial runs for seeds 7, 17, and 27 on each locally prepared scene; retain run configs, splits, validation logs, test summaries, and final renders under `outputs-cap200k/`.
- [ ] **Kitchen and room datasets:** Prepare both remaining Mip-NeRF 360 scenes in the [documented real-data layout](data.md), and verify that the loader can read their COLMAP cameras and points.
- [ ] **Kitchen and room runs:** Complete the same nine kernel/seed runs per scene under the 200k, 20k-step configuration.
- [ ] **Matrix audit:** Confirm all nine real scenes have nine complete runs, matching within-scene settings and paired seeds; record any failed or incomparable jobs in `missing_jobs.csv` and rerun them. The five scenes already present are bicycle, bonsai, counter, flowers, and garden.
- [ ] **Aggregate report:** Regenerate per-scene and cross-scene paired PSNR, SSIM, LPIPS, validation, render-time, tile-work, training-time, splat-count, and sampled GPU-memory summaries. Use scenes and paired seeds for conclusions; do not count test views as independent scenes.

## 3. Analyze the existing synthetic results

The saved `outputs/kernels/` matrix covers chair, drums, ficus, hotdog, lego, materials, mic, and ship, with three kernels and three seeds per scene. It uses 256 × 256 images, a white background, an 8,000-step schedule, a 100,000-splat cap, and no projected-radius guard. Preliminary inspection shows polynomial below Gaussian on mean test PSNR and LPIPS in all eight scenes; linear is mixed. Keep this result visible in the final report.

- [ ] **Synthetic matrix audit:** Verify all 72 run configs, paired seeds, train/validation/test splits, shared within-scene initialization, test metrics, and completed step counts. Deliver a coverage table and list any incomparable or missing runs.
- [ ] **Synthetic results report:** Generate per-scene paired Gaussian-versus-linear and Gaussian-versus-polynomial PSNR, SSIM, and LPIPS deltas, seed variation, validation curves, and cost measures from the saved runs. Deliver a compact summary table plus a plot with one point per scene/seed; keep the 8k/100k results separate from the real 20k/200k results.
- [ ] **Synthetic visual examples:** Export matched held-out views and crops from at least one scene where the alternatives lose and one where linear is competitive. Deliver a labeled comparison with reference images and all three kernels.
- [ ] **Cross-matrix discussion:** Write a short section comparing the *patterns* in synthetic and real results. List the changed settings explicitly and avoid attributing the difference to scene type, capacity, or kernel alone. If a causal explanation is needed, specify a new matched-settings experiment first.

## 4. Diagnose the observed tradeoffs

- [ ] **Capacity comparison:** Compare matched 100k and 200k runs on at least one outdoor scene in addition to the existing [bonsai study](../outputs-cap200k/capacity_report/bonsai/report.md). Deliver a table showing how each kernel's gap from Gaussian changes with capacity. Keep training steps, learning-rate schedule, resolution, split, and initialization matched.
- [ ] **Convergence review:** Produce a compact table and plot of final versus best validation PSNR for every scene/kernel/seed. Flag large late declines, including bonsai, and inspect their saved snapshots. Keep the fixed final checkpoint as the primary test endpoint unless a validation-based selection rule is specified before further test analysis.
- [ ] **Matched cost analysis:** Plot quality against rendering time for hardware-matched runs and report tile pairs separately from elapsed time. State the GPU model and omit unmatched timing comparisons from speed claims.
- [ ] **Visual error gallery:** Select representative gains, ties, and losses. Export identical held-out cameras and crops for the reference, Gaussian, linear, and polynomial renders; annotate visible edge, texture, blur, and artifact differences without claiming causes from appearance alone.
- [ ] **Optional resolution check:** If resources permit, rerun a small indoor/outdoor subset at a higher image resolution with all three kernels and paired seeds. Deliver a separate report; do not mix these scores with the 390-pixel matrix.

## 5. Present and demonstrate the findings

- [x] **Experiment page:** The [Procedure page](../site/procedure.html) describes scene splits, matched seeds, initialization, training settings, and evaluation rules.
- [ ] **Full results page:** Expand the existing Atlas summary into a dedicated page with the complete synthetic and real-scene matrices, uncertainty, matched image crops, and quality-versus-cost views.

- [ ] **Kernel appearance demo:** Show isolated and overlapping splats with identical geometry, opacity, and color under each weight function. Include the curve figure so the appearance has a precise explanation.
- [ ] **Scene comparison demo:** Extend the existing [snapshot comparison site](experiments.md#snapshot-comparison-site) to the 200k outputs. Deliver synchronized camera, seed, checkpoint, and zoom controls with the reference and three kernel renders side by side.
- [ ] **Presentation draft:** Prepare slides covering the question, prior work, kernel definitions, the two distinct experimental settings, paired results within each matrix, seed variation, quality versus cost, matched image crops, limitations, and conclusions. Show the synthetic/real contrast explicitly, including losses by polynomial on synthetic scenes.
- [ ] **Reproduction notes:** Update the experiment catalogue and README with the final matrix, exact commands/configuration, output paths, and demo launch instructions. Verify that a reader can regenerate the reports from saved run files.

## Completion criteria

- [ ] **Final handoff:** Deliver the literature matrix, synthetic and 200k real-scene reports, diagnostic follow-ups, presentation, and working demo. State conclusions only for each tested matrix's scenes, resolution, capacity, training budget, and kernel definitions; identify open questions rather than requiring a novel or uniformly positive result.
