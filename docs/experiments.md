# Experiment catalogue and workflows

This catalogue describes the run files currently copied into this repository.
Every listed scene has three kernels (`gaussian`, `linear`, `polynomial`) and
paired seeds 7, 17, and 27: nine completed runs per scene. `outputs/`,
`outputs-pilot/`, and `outputs-20k/` are ignored by Git. “Historical” means
retain the data for provenance; it does not require moving or deleting files.

| Matrix | Scenes | Saved training steps | Location | Role |
| --- | --- | ---: | --- | --- |
| Synthetic | chair, drums, ficus, hotdog, lego, materials, mic, ship | 8,000 | `outputs/kernels/` | Synthetic comparison |
| Real, early | bicycle, bonsai, flowers, garden | 8,000 | `outputs/kernels/` | Historical real matrix |
| Real, counter | counter | **16,000** | `outputs/kernels/counter/` | Historical; the copied run configs are 16k, not 8k |
| Bonsai pilot | bonsai | 16,000 | `outputs-pilot/kernels/bonsai/` | Historical budget pilot |
| Real, current | bicycle, bonsai, counter | 20,000 | `outputs-20k/kernels/` | Current real matrix and comparison site |

The copied `counter` runs in `outputs/kernels/` are 16k. No 8k counter
checkpoints are present there. All listed real runs use 100,000 splat capacity
and the 0.25 projected-radius guard. The current 20k matrix also extends the
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

For the current matrix, set a separate output root and pass the 20k config.
Use the same command for bicycle and counter:

```sh
export OUTPUT_ROOT="$PWD/outputs-20k"
sbatch scripts/slurm_train_real.sh data/bonsai config/real_20k.yaml
```

After the training arrays finish, submit analysis:

```sh
sbatch scripts/slurm_analyze_kernels.sh
```

The matrix runner's
`--list` option shows the nine kernel/seed array indices, allowing failed
indices to be retried without replacing completed runs.

## Snapshot comparison site

Run the GPU render array once per scene on the cluster. The Slurm script only
exports images and source photographs; it does not build the site:

```sh
export OUTPUT_ROOT="$PWD/outputs-20k"
mkdir -p logs
sbatch scripts/slurm_render_views.sh data/bicycle
sbatch scripts/slurm_render_views.sh data/bonsai
sbatch scripts/slurm_render_views.sh data/counter
```

Each array has three seed tasks. By default, each task exports three training
cameras and eight reserved validation cameras at saved steps from 6k onward.
Existing image exports are skipped, so a failed task can be resubmitted.
After the render arrays finish, copy `outputs-20k/view_renders/` to the local
machine beside the copied `outputs-20k/kernels/` directory. Build the site
locally:

```sh
.venv/bin/python scripts/build_kernel_site.py --input-root outputs-20k/kernels
python -m http.server 8765 --directory outputs-20k/site
```

The site uses pre-rendered images; opening `index.html` directly from the
filesystem may block its JSON fetch, so serve the directory over local HTTP.

## Capacity follow-up decision

The bonsai COLMAP initialization contains 50,000 splats. All nine current
bonsai runs reach the 100,000 cap by the first validation log at step 500 and
remain there at every saved PLY. The cap is therefore active, but the current
data cannot show whether it hurts Gaussian more than the other kernels.

First compare matched train and validation views across steps to identify
whether Gaussian detail is absent from the start or degrades later. Then run
a **capacity pilot on bonsai** at 20k steps: 100k versus a higher capacity
(for example, 200k), all three kernels, and one paired seed. If the higher
capacity changes the result materially, run the same capacity comparison
for seeds 7, 17, and 27. Keep resolution, initialization, position schedule,
guard, optimizer, split, and evaluation protocol fixed. Save higher-capacity
runs to a new output root. Report the kernel-by-capacity interaction:

The prepared 200k config changes only `model.capacity` from
`config/real_20k.yaml`. For a seed-17 pilot, matrix indices 1, 4, and 7
select Gaussian, linear, and polynomial respectively:

```sh
export OUTPUT_ROOT="$PWD/outputs-cap200k"
sbatch --array=1,4,7 scripts/slurm_train_real.sh data/bonsai config/real_20k_200k.yaml
```

Check the GPU memory and completion status before expanding to all three
seeds. Compare each 200k run against its seed-17 counterpart in
`outputs-20k/kernels/bonsai/`, then compute the interaction:

`(alternative − Gaussian at high capacity) − (alternative − Gaussian at 100k)`.

A smaller alternative-versus-Gaussian gap at higher capacity would support
the idea that alternative kernels fit better under the constrained splat
budget. If all three kernels improve similarly, the cap limits the renderer
but does not explain the relative bonsai result. Inspect actual splat counts
and GPU memory because higher capacity increases both training resource use
and the number of primitives that may be active.
