# Linear and quadratic screen-space kernels for 3D Gaussian splatting

**Draft status:** Results and interpretation are based on the saved final-test summaries and generated reports. Add the required institutional format, literature comparison, and matched visual examples before submission.

## Abstract

We compare Gaussian, linear radial, and degree-2 polynomial screen-space weights in one splatting renderer. The projected 3D ellipsoids, color model, and compositing are held fixed within each experiment; the pixel weight inside a projected footprint changes. We evaluate two distinct matrices with three paired seeds per scene. The synthetic matrix contains eight NeRF-synthetic scenes at 256 × 256 pixels, 100,000 splats, and 8,000 training steps. The real-scene matrix contains nine Mip-NeRF 360 scenes at a 390-pixel long edge, 200,000 splats, and 20,000 steps. At the fixed final checkpoint, polynomial loses mean PSNR, SSIM, and LPIPS against Gaussian in every synthetic scene and seed. In the real-scene matrix, polynomial improves mean PSNR in seven of nine scenes and 23 of 27 paired seeds, with a mean scene difference of +0.579 dB but a median of only +0.084 dB. Linear gives mixed results in both matrices. The contrast is descriptive because the matrices differ in multiple settings. A kernel choice should therefore be scoped to the tested configuration and scene, with quality and rendering cost considered together.

## 1. Question and kernel definitions

The question is how the shape of a projected splat's screen-space weight affects held-out reconstruction quality, optimization behavior, and cost under a fixed within-scene budget. Let `q` be the squared Mahalanobis distance from a pixel to the center of the projected ellipse. The implemented weights are:

- Gaussian: `exp(-q/2)`.
- Linear radial: `max(0, 1 - sqrt(q / (8 ln 2)))`.
- Degree-2 polynomial: `max(0, 1 - q / R²)²`, where `R² = 2 ln 2 / (1 - 1/sqrt(2))`.

All three equal one-half at `q = 2 ln 2`. Their total mass and support differ. The linear and polynomial weights have finite support; the Gaussian is rendered under the implementation's alpha threshold and compact-box rule. The degree is fixed at 2. Full implementation details are in [methodology.md](methodology.md).

## 2. Experimental design

| Matrix | Scenes | Image size | Capacity | Steps | Background | Projected-radius guard |
| --- | ---: | --- | ---: | ---: | --- | --- |
| NeRF-synthetic | 8 | 256 × 256 | 100,000 | 8,000 | White | Disabled |
| Mip-NeRF 360 | 9 | 390-pixel long edge | 200,000 | 20,000 | Black | 0.25 of larger image dimension |

Each scene has Gaussian, linear, and polynomial runs with seeds 7, 17, and 27. Within a scene and matrix, runs share the camera split, settings, and initialization. Synthetic scenes use a fixed generated point initialization; real scenes use the same COLMAP initialization. The primary endpoint is the final checkpoint at the specified training step. We compare paired final-test means for PSNR and SSIM (higher is better) and VGG LPIPS (lower is better). The unit for cross-scene interpretation is a scene, not an individual test view. Validation curves diagnose convergence; they do not select the primary test checkpoint. Render time is a measured single-view elapsed time, while tile pairs count projected splat/tile records and are not a time proxy. Timing and sampled GPU memory are compared only where GPU models match.

The real matrix has 81 of 81 expected final-test summaries and no missing-job
entries in its generated `kernel_analysis/missing_jobs.csv`. The generated scene
reports verify the within-scene settings and splits. The synthetic matrix has
72 of 72 expected final-test summaries. A fuller audit of every synthetic
split and initialization file remains to be documented. Generated reports are
ignored by Git and must be regenerated from the corresponding run files.

## 3. Synthetic results: 8k steps, 100k splats

The table reports the mean of each scene's three paired seed differences from Gaussian. Positive PSNR and negative LPIPS favor the alternative.

| Scene | Linear ΔPSNR (dB) | Linear ΔLPIPS | Polynomial ΔPSNR (dB) | Polynomial ΔLPIPS |
| --- | ---: | ---: | ---: | ---: |
| Chair | −0.106 | +0.0029 | −0.441 | +0.0042 |
| Drums | −0.042 | +0.0024 | −0.201 | +0.0049 |
| Ficus | −0.111 | +0.0015 | −0.082 | +0.0035 |
| Hotdog | +0.221 | −0.0007 | −0.928 | +0.0102 |
| Lego | −0.255 | +0.0030 | −0.487 | +0.0030 |
| Materials | +0.024 | +0.0015 | −0.231 | +0.0096 |
| Mic | −0.069 | +0.0008 | −0.321 | +0.0033 |
| Ship | −0.133 | +0.0030 | −0.487 | +0.0072 |

Across scene means, linear changes PSNR by −0.059 dB and LPIPS by +0.0018; it improves PSNR in two of eight scenes. Polynomial changes PSNR by −0.397 dB and LPIPS by +0.0057. Its PSNR, SSIM, and LPIPS are worse than Gaussian in all eight scenes and all 24 paired seeds. These results should remain visible alongside the real-scene findings.

## 4. Real-scene results: 20k steps, 200k splats

The table again reports the mean paired difference across the three seeds in
each scene. The complete source table is `reports/overview.csv` under the
real-scene output root.

| Scene | Linear ΔPSNR (dB) | Linear ΔLPIPS | Polynomial ΔPSNR (dB) | Polynomial ΔLPIPS |
| --- | ---: | ---: | ---: | ---: |
| Bicycle | −0.070 | +0.0054 | +0.076 | −0.0014 |
| Bonsai | +0.992 | −0.0093 | +3.123 | −0.0520 |
| Counter | +0.208 | −0.0005 | +0.350 | −0.0167 |
| Flowers | −0.086 | +0.0064 | +0.084 | −0.0029 |
| Garden | −0.116 | +0.0017 | −0.036 | +0.0001 |
| Kitchen | +0.238 | +0.0001 | +1.098 | −0.0278 |
| Room | +0.524 | +0.0005 | +0.455 | −0.0100 |
| Stump | −0.151 | +0.0070 | +0.074 | −0.0115 |
| Treehill | −0.042 | +0.0048 | −0.018 | −0.0019 |

Polynomial improves mean PSNR in seven of nine scenes (23/27 paired seeds), mean SSIM in all nine, and mean LPIPS in eight. Its mean scene PSNR difference is +0.579 dB, while its median scene difference is +0.084 dB. Bonsai (+3.123 dB) and kitchen (+1.098 dB) drive much of the mean; garden and treehill are near ties with slightly negative mean PSNR. Linear improves mean PSNR in four scenes (13/27 seeds) and mean LPIPS in two scenes. Its mean scene PSNR difference is +0.166 dB, but its median is −0.042 dB. These are exploratory results from nine selected scenes and three seeds each, not evidence of a universal ranking.

## 5. Optimization and capacity diagnostics

All 81 real runs finish at the 200,000-splat cap. The largest gap between best and final validation PSNR is bonsai Gaussian seed 27: its best value occurs at step 8,500 and the 20,000-step value is 2.96 dB lower. Bonsai linear seed 17 is 1.14 dB below its best validation value at the final checkpoint. This makes the magnitude of the bonsai final-test advantage sensitive to optimization stability. The prespecified final checkpoint remains the primary endpoint; a different checkpoint rule would need to be specified before further test-based comparison.

Three matched 100k-versus-200k real-scene studies show that capacity response
is scene dependent. On bonsai, the polynomial-minus-Gaussian PSNR gap falls
from +5.18 to +3.12 dB. On counter, it falls from +0.67 to +0.35 dB. On
bicycle, it changes only from +0.04 to +0.08 dB. These figures come from the
three generated `capacity_report/<scene>/report.md` files under the
higher-capacity output root. The comparisons match training steps, schedule,
resolution, split, and initialization within each scene. They show that the
effect of increasing the splat cap is not uniform across scenes.

## 6. Cost and interpretation

The real-scene reports compare rendering and training time using same-GPU seed pairs. Linear's scene-level render-time changes range from −5.9% to +4.8%; polynomial's range from −13.6% to +32.0%. Several scene estimates have only one or two GPU-matched pairs, so these ranges do not establish a stable speed ranking. Tile-pair counts and elapsed time should be reported separately. Sampled peak training GPU memory is unchanged between kernel choices in the matched real-scene pairs, at the resolution of the 250 ms NVML sampling method.

The synthetic and real matrices change several settings together: scene set and type, image resolution, background, initialization, splat cap, training duration, learning-rate schedule, and projected-radius guard. Their differing outcomes cannot be attributed to any one of those factors without a matched-settings experiment. The results establish a configuration-specific quality tradeoff in this renderer: polynomial performs worse throughout the tested synthetic matrix and often better in the tested 200k real-scene matrix; linear is inconsistent. Matched held-out image crops are still needed to describe visible differences in edges, texture, blur, and artifacts.

## 7. Work needed for the submission version

1. Add a sourced prior-work table and distinguish each paper's pipeline from the exact kernels and evaluation here.
2. Add a kernel-curve figure, paired scene/seed plot, quality-versus-render-time plot, and matched reference/render crops.
3. Complete the synthetic config, split, and initialization audit and report its coverage explicitly.
4. Insert the required report format, citation style, authorship details, and any course-specific sections once the submission brief is available.
