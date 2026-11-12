# Experiment catalogue

This page identifies experiments by their settings and purpose. Output roots are
local storage choices, not experiment identifiers. All complete matrices pair
`gaussian`, `linear`, and `polynomial` (quadratic) at seeds 7, 17, and 27.
The [methodology](methodology.md) defines the comparison, and the
[report draft](final_report_draft.md) summarizes the findings.

| Experiment | Scenes | Steps | Splat cap | Role |
| --- | --- | ---: | ---: | --- |
| NeRF-synthetic main matrix | chair, drums, ficus, hotdog, lego, materials, mic, ship | 8,000 | 100,000 | Main synthetic result; 72 runs |
| Mip-NeRF 360 main matrix | bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill | 20,000 | 200,000 | Main real-scene result; 81 runs |
| Matched real capacity comparator | bicycle, bonsai, counter | 20,000 | 100,000 | Secondary capacity diagnostic |
| Earlier real matrix | bicycle, bonsai, flowers, garden | 8,000 | 100,000 | Historical, excluded from the main comparison |
| Earlier counter run | counter | 16,000 | 100,000 | Historical, excluded from the main comparison |
| Bonsai pilot | bonsai | 16,000 | 100,000 | Historical, excluded from the main comparison |

The synthetic main matrix uses `config/synthetic.yaml`; the real main matrix
uses `config/real_20k_200k.yaml`. The capacity comparator uses
`config/real_20k.yaml`. `config/real.yaml` belongs to the earlier 8k/100k
real runs. The earlier counter run configs say 16k; no 8k counter checkpoints
are present in that group. Do not pool runs across these budgets in one
aggregate analysis. Real runs use the 0.25 projected-radius guard; synthetic
runs do not.

## Reproducing and locating results

Prepare datasets as in [data.md](data.md), then follow the [README](../README.md)
for setup, training, analysis, and site building. Use a distinct output root
for each configuration. Repeat the nine-task kernel/seed array for each scene;
`scripts/run_kernel_matrix.py --list` prints the index mapping.

A run is stored at `<output-root>/kernels/<scene>/<kernel>/seed_<seed>/`.
`run_config.json` identifies its settings independently of the root's name.
Real-scene runs also contain `split.json`; synthetic scenes use the dataset's
train/validation/test manifests, which must be kept with the source data.
Analysis writes to
`<output-root>/kernel_analysis/`; reporting writes to
`<output-root>/reports/`; the site builder writes to
`<output-root>/site/` by default. Optional view exports live in
`<output-root>/view_renders/`. These generated directories are ignored by Git,
so a fresh clone needs its own run files to regenerate numerical reports and
the interactive site.

The checked-in Analysis example uses four static images from bonsai DSCF5648,
seed 17, step 20,000. They illustrate this study without requiring the dataset.
Numerical charts and explorer images depend on the selected output roots; the
Analysis prose must be checked if a different matrix is supplied.

## Capacity diagnostic

`scripts/report_capacity.py` compares a lower and higher splat cap for one
real scene at a time. It verifies matching configurations apart from capacity,
camera splits, held-out view names, LPIPS backbone, and COLMAP initialization
PLYs. It then reports paired final-test changes, the change in each alternative
kernel's gap from Gaussian, validation curves, per-view counts, and matched-GPU
cost. It writes `capacity_comparison.csv`, `per_view_capacity.csv`, plots, and
`report.md` under the higher-capacity root's `capacity_report/<scene>/` by
default. It does not train or alter checkpoints.

For a saved 20k-step, 100k-versus-200k pair, pass the two `kernels/` trees:

```sh
LOW_ROOT=outputs-my-real-100k
HIGH_ROOT=outputs-my-real-200k
.venv/bin/python scripts/report_capacity.py \
  --low-root "$LOW_ROOT/kernels" --high-root "$HIGH_ROOT/kernels" \
  --scene bonsai
```

The existing matched comparison covers bicycle, bonsai, and counter. On bonsai,
the mean polynomial-minus-Gaussian PSNR gap changes from +5.18 to +3.12 dB as
capacity rises; the linear gap changes from +4.54 to +0.99 dB. On counter,
the polynomial gap changes from +0.67 to +0.35 dB; on bicycle, from +0.04 to
+0.08 dB. This is supporting evidence that capacity can interact with kernel
choice, not an estimate of a general kernel advantage. The script is useful for
reproducing that claim, but it is optional for the primary kernel comparison.
