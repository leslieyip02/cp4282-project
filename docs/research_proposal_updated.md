# Research proposal: screen-space kernel shape in 3D Gaussian splatting

*CP4282 individual project*

**Historical planning document.** For the implemented protocol and saved
results, read [methodology.md](methodology.md), [experiments.md](experiments.md),
and [final_report_draft.md](final_report_draft.md).

## Research question

In a 3D Gaussian splatting trainer built from the course implementation, how does replacing the Gaussian **screen-space splat weight** with a linear radial or degree-2 polynomial weight affect held-out reconstruction quality, optimization stability, and rendering cost? I will test the three kernels on eight NeRF-synthetic scenes and nine Mip-NeRF 360 scenes, holding the training configuration, data split, initialization, and splat cap fixed within each scene. I want to learn whether any quality gain is consistent across scenes and random seeds and whether it is worth its rendering cost.

## Motivation and course connection

The project builds on the course's treatment of Gaussian splats and rasterization (Units 2–3), differentiable rendering (Unit 5), evaluation metrics (Unit 6), and full 3DGS training (Units 7 and 9). A 3D ellipsoid projects to an ellipse in the image. Its screen-space weight determines how much opacity it contributes at each covered pixel. Changing that weight can alter rendered edges, the gradients used to optimize splats, the number of tiles touched, and rendering time. The course explains the Gaussian formulation but does not establish which footprint shape works best under a particular training and capacity budget.

[Kerbl et al.](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) establish the Gaussian baseline with anisotropic splats, density control, and visibility-aware rasterization. [Chen et al.](https://arxiv.org/abs/2411.12440) argue that linear kernels can give sharper results and report fidelity and frame-rate gains in their 3D Linear Splatting system. [Huang et al.](https://arxiv.org/abs/2412.11752) explore learnable deformable radial kernels together with other representation and rendering changes. [Mueller et al.](https://arxiv.org/abs/2603.18707) propose a polynomial approximation to the exponential kernel for more efficient culling while retaining compatibility with existing Gaussian-splat data. These are motivation and prior art, not direct predictions for my implementation: their full systems and kernel definitions differ. My question isolates three *fixed screen-space weights* in one trainer rather than claiming a new kernel.

## Problem formulation

The input is a set of posed training photographs and initial 3D points. The output is a trained splat scene that renders held-out camera views. The baseline uses a Gaussian weight. The two variants change the function that maps a pixel's elliptical distance from a splat center to its opacity weight. Let `q` be squared Mahalanobis distance in the projected ellipse:

| Variant | Screen-space weight `w(q)` | Support |
| --- | --- | --- |
| Gaussian baseline | `exp(−q/2)` | Truncated by the renderer's alpha/support rule |
| Linear radial | `max(0, 1 − √(q/(8 ln 2)))` | `q ≤ 8 ln 2` |
| Polynomial, degree 2 | `max(0, 1 − q/R²)²`, with `R² = 2 ln 2/(1 − 1/√2)` | `q ≤ R²` |

All three equal one-half at `q = 2 ln 2`. This matches their half-maximum radius while allowing support and total weight to differ. I will keep the projected 3D ellipsoids, color model, front-to-back compositing, optimizer, loss, densification settings, and evaluation procedure unchanged within each scene. Kernel-specific support will be used consistently by training, tile culling, validation, and test rendering. Polynomial degree is fixed at 2; I do not plan to sweep it or select it using test views. In real scenes, I will apply the same 0.25 projected-radius guard to every kernel to exclude splats whose support radius exceeds a quarter of the larger image dimension.

## Planned implementation and experiments

I will start from the Unit 9/course 3DGS trainer, which provides projection, splat optimization, densification, and held-out rendering. I will implement the linear and polynomial weights and their backward derivatives, update footprint bounds and tile culling for each support rule, and make the kernel selectable in training and evaluation. Before full runs, I will compare analytic gradients with finite differences and use small render tests to check that culling does not discard pixels with meaningful weight. I will save each run's configuration, split, validation history, final checkpoint, test metrics, and rendered examples.

The primary experiment has **153 planned training runs**:

| Matrix | Scenes | Kernels × seeds | Runs | Resolution | Splat cap | Training budget |
| --- | --- | --- | ---: | --- | ---: | ---: |
| NeRF-synthetic | chair, drums, ficus, hotdog, lego, materials, mic, ship | 3 × 3 | **72** | 256 × 256 | 100,000 | 8,000 steps |
| Mip-NeRF 360 | bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill | 3 × 3 | **81** | 390-pixel long edge | 200,000 | 20,000 steps |

The seeds will be 7, 17, and 27 for every scene and kernel. Synthetic scenes will use one generated initialization per scene, reused across the nine matched runs. Real scenes will use the same COLMAP point initialization across their nine runs. I will prepare fixed train, validation, and held-out test splits before comparing kernels. The synthetic scenes will use a white background and no projected-radius guard; the real scenes will use a black background and the common guard above. I will run one scene/kernel/seed job per CUDA GPU on the compute cluster and record GPU model. The two matrices have deliberately different budgets and will be analyzed separately; comparing them will show patterns, not isolate the effect of scene type.

If compute permits, I will add a secondary capacity study on bonsai, bicycle, and counter: **27 further 100,000-splat runs** (3 scenes × 3 kernels × 3 seeds), compared with their planned 200,000-splat runs. Steps, schedule, resolution, split, and initialization will be matched within each scene so that the cap is the intended change. This study will test whether an alternative kernel's gap from Gaussian changes with capacity. It will remain separate from the primary 153-run analysis.

## Evaluation plan and failure criterion

The main quality comparison is at **equal training steps and equal splat cap within each scene**, using the fixed final checkpoint. I will pair each alternative with the Gaussian run of the same scene and seed and report differences in test PSNR, SSIM, and VGG LPIPS. Higher PSNR and SSIM and lower LPIPS indicate better reconstruction under those measures. I will show per-seed values and scene means; across-scene conclusions will treat scenes, not individual test cameras, as the units. With only three seeds per scene, uncertainty estimates will be interpreted cautiously.

I will inspect validation curves and saved snapshots for late declines or unstable seeds. Side-by-side renders of identical held-out cameras and crops will document differences in edges, fine texture, blur, and artifacts, including cases where an alternative loses. The final checkpoint remains the primary test endpoint; any validation-selected checkpoint analysis would require a rule fixed before examining its test results.

For cost, I will report median single-view render time, total training time, tile-pair counts, final splat count, and sampled peak GPU memory. I will compare elapsed times only for runs on the same GPU model and show quality versus render time. Tile pairs will be presented as a measure of rasterization work, not as a substitute for elapsed time. Equal-iteration quality is the primary protocol; the quality-versus-time plot will reveal whether any gain comes with a speed tradeoff.

I will not regard an alternative as generally better if gains occur only in a few scenes or seeds, if one metric improves while structural/perceptual measures or matched images consistently worsen, or if the gain entails a substantial rendering cost. A negative or mixed result still answers the research question. The two matrices change resolution, background, initialization, cap, training budget, and radius guard together, so I will not claim that a cross-matrix difference is caused by scene type or kernel alone. A causal explanation would require a further matched-settings experiment.

## Intended deliverables

I will provide an implementation with reproducible run configurations and instructions; separate synthetic and real-scene results tables and paired plots; validation, capacity, and cost diagnostics; matched held-out image comparisons; a short written analysis of benefits, losses, and limits; and a 20-minute presentation with a synchronized visual comparison. The final report will cite the repository and explain how to regenerate the results. I will also declare any AI tools used and describe how their outputs were checked, as required by the [course project guidelines](https://weitsang.github.io/gaussian-splatting-notes/project/).
