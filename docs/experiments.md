# Experiment catalogue and workflows

This catalogue describes the run files currently copied into this repository.
Unless noted, every listed scene has three kernels (`gaussian`, `linear`,
`polynomial`) and paired seeds 7, 17, and 27: nine completed runs per scene.
Output directories are ignored by Git. “Historical” means
retain the data for provenance; it does not require moving or deleting files.

| Matrix | Scenes | Saved training steps | Location | Role |
| --- | --- | ---: | --- | --- |
| Synthetic | chair, drums, ficus, hotdog, lego, materials, mic, ship | 8,000 | `outputs/kernels/` | Synthetic comparison |
| Real, early | bicycle, bonsai, flowers, garden | 8,000 | `outputs/kernels/` | Historical real matrix |
| Real, counter | counter | **16,000** | `outputs/kernels/counter/` | Historical; the copied run configs are 16k, not 8k |
| Bonsai pilot | bonsai | 16,000 | `outputs-pilot/kernels/bonsai/` | Historical budget pilot |
| Real, 100k capacity | bicycle, bonsai, counter | 20,000 | `outputs-20k/kernels/` | Historical; matched capacity comparator |
| Real, 200k capacity | bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill | 20,000 | `outputs-cap200k/kernels/` | Current real matrix and comparison site |

The copied `counter` runs in `outputs/kernels/` are 16k. No 8k counter
checkpoints are present there. All listed real runs except the 200k capacity matrix
use 100,000 splat capacity; every real run uses the 0.25 projected-radius guard.
Both 20k real matrices extend the
position learning-rate schedule to 20k, so comparisons across budgets change
both training duration and schedule. The old and current bicycle heights are
260 and 259 pixels respectively; do not interpret their cross-budget metric
change as a strictly matched comparison.

## Synthetic scene workflow

Provide the train, validation, and test manifests and images under
`data/<scene>/`. Generate one shared initialization if it is absent, then
submit one nine-task array per scene. The fixed initialization is reused by
every kernel and seed in that scene.

```sh
.venv/bin/python generate_synthetic_init.py --data data/lego --seed 7 --points 50000
mkdir -p logs
sbatch scripts/slurm_train_synthetic.sh data/lego config/synthetic.yaml
```

Run the initialization command only when `data/lego/init.ply` is missing.
Repeat the array for chair, drums, ficus, hotdog, materials, mic, and ship.
The default output root is `outputs/`. After arrays finish, check missing
jobs and generate aggregate analysis:

```sh
sbatch scripts/slurm_analyze_kernels.sh
cat outputs/kernel_analysis/missing_jobs.csv
```

## Real scene workflow

Provide a prepared Mip-NeRF 360 COLMAP dataset under `data/<scene>/`.
The dataset's points initialize all matched runs; no synthetic `init.ply`
generation is needed. The original 8k real configuration is `config/real.yaml`:

```sh
mkdir -p logs
sbatch scripts/slurm_train_real.sh data/bonsai config/real.yaml
```

For the current 200k-capacity matrix, set a separate output root and pass the
200k-capacity 20k-step config. Repeat for all nine real scenes:

```sh
export OUTPUT_ROOT="$PWD/outputs-cap200k"
sbatch scripts/slurm_train_real.sh data/bonsai config/real_20k_200k.yaml
```

After the training arrays finish, submit analysis:

```sh
sbatch scripts/slurm_analyze_kernels.sh
```

The matrix runner's
`--list` option shows the nine kernel/seed array indices, allowing failed
indices to be retried without replacing completed runs.

## Snapshot comparison site

The generated site has Motivation, Kernels, Procedure, Explore runs, Results,
and Analysis pages. The kernel guide compares radial profiles, 2D footprints,
and 3D weight surfaces. The explorer synchronizes camera, seed, and checkpoint.
The current build uses `outputs-cap200k/kernels/` for nine real scenes (81
runs) and `outputs/kernels/` for eight synthetic scenes (72 runs). The Results
tab shows both matrices separately. The Analysis page reads these summaries for
scene-level quality and matched-GPU cost charts, and discusses convergence and
limitations for the current matrices. The real-scene explorer uses saved snapshot
PNGs for one fixed validation view per scene; additional cameras and source
photos require the render export below.

In the commands below, replace `<outputs>` with the directory that holds the
selected `kernels/` tree (for this matrix, `outputs-cap200k`). Keep
`kernels/`, `view_renders/`, and `site/` under that same directory.

Run the GPU render array once per scene on the cluster. The Slurm script only
exports images and source photographs; it does not build the site:

```sh
export OUTPUT_ROOT="$PWD/<outputs>"
mkdir -p logs
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
  sbatch scripts/slurm_render_views.sh "data/$scene"
done
```

Each array has three seed tasks. By default, each task exports three training
cameras and eight reserved validation cameras at saved steps from 6k onward.
Existing image exports are skipped, so a failed task can be resubmitted.
After the render arrays finish, copy `<outputs>/view_renders/` to the local
machine beside the copied `<outputs>/kernels/` directory. Build the site
locally:

```sh
.venv/bin/python scripts/build_kernel_site.py --input-root "<outputs>/kernels" \
  --synthetic-root outputs/kernels
python -m http.server 8765 --directory "<outputs>/site"
```

The motivation, kernel, and procedure pages work as static files. The explorer
and Results tab fetch `data.json`; Results and Analysis also fetch
`synthetic-results.json`. Serve the site directory over local HTTP for those
pages. Synthetic summaries are optional for a partial build, but Analysis
requires both matrices. Its written interpretation refers to the current 8k
synthetic and 20k, 200k-capacity real experiments and should be revised if
those experiments change.

## Capacity follow-up decision

The bonsai COLMAP initialization contains 50,000 splats. All nine 100k-capacity
bonsai runs reach the 100,000 cap by the first validation log at step 500 and
remain there at every saved PLY. The cap is therefore active, but the current
100k matrix alone cannot show whether it hurts Gaussian more than the other kernels.

The three-seed 200k-capacity matrix is in `outputs-cap200k/`. Compare it
against the matched 100k runs with:

```sh
.venv/bin/python scripts/report_capacity.py \
  --low-root outputs-20k/kernels --high-root outputs-cap200k/kernels \
  --scene bonsai
```

This verifies configuration, split, test views, LPIPS backbone, and identical
COLMAP initialization, then writes `outputs-cap200k/capacity_report/bonsai/`.
The prepared 200k config changes only `model.capacity` from
`config/real_20k.yaml`. Across seeds, Gaussian gains 0.67–4.93 dB PSNR,
linear changes by −2.89 to +0.66 dB, and polynomial gains 0.15–1.35 dB.
The mean linear-minus-Gaussian gap falls from +4.54 to +0.99 dB, and the
polynomial-minus-Gaussian gap from +5.18 to +3.12 dB. The interaction is:

`(alternative − Gaussian at high capacity) − (alternative − Gaussian at 100k)`.

The shrinking gaps support a capacity-dependent kernel effect on bonsai, but
the response varies substantially by seed. Polynomial still leads Gaussian
on all three seeds at 200k; linear is mixed. All three 200k kernels hit their
cap by step 1,500. Gaussian seed 27's validation PSNR peaks at 8,500 and is
2.96 dB lower at 20,000, so optimization stability is also part of the
diagnosis. This is one scene; do not generalize the capacity interaction to
other real scenes without a matched capacity study there. The higher-capacity
runs use 96 MiB more sampled training GPU memory on matched GPU models and
about 1.6–1.9× as much training time.
