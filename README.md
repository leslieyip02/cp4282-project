# Linear and quadratic screen-space kernels for 3D Gaussian splatting

This project tests whether replacing the Gaussian screen-space weight in a 3DGS
renderer with a linear radial or quadratic weight changes held-out image quality
and cost. The quadratic kernel is named `polynomial` in the code and run files.
Within each experiment, the three kernels share the scene, initialization,
camera split, training schedule, and splat cap. The main comparisons are an
eight-scene NeRF-synthetic matrix and a nine-scene Mip-NeRF 360 matrix; their
different budgets mean their outcomes should be interpreted separately.

**Read the study:** [method and kernel definitions](docs/methodology.md) →
[experiment catalogue](docs/experiments.md) →
[results and report draft](docs/final_report_draft.md). The
[Kernel Atlas source](site/index.html) presents the results interactively after
building it from run files. [Dataset preparation](docs/data.md) and
[cluster operations](docs/cluster.md) support reproduction.

**Repository map:** `shared/`, `3dgs_trainer.py`, and `3dgs_evaluator.py` contain
the renderer; `kernel_experiment.py` and `scripts/run_kernel_matrix.py` run the
paired comparison; `analyze_kernels.py` and `scripts/report_kernels.py` summarize
it; `scripts/build_kernel_site.py` builds the Atlas. The capacity report is a
secondary, matched-budget diagnostic described in the
[catalogue](docs/experiments.md#capacity-diagnostic). `config/as2_reference.yaml`
and [the earlier proposal](docs/research_proposal_updated.md) document the
starting point, not the final experimental protocol.

Datasets and generated results are ignored by Git. Choose a distinct output
root for each experimental configuration. This README uses example roots;
folder names carry no methodological meaning. Keep each run's
`run_config.json`, metrics, and checkpoint together. Real-scene runs also save
`split.json`; synthetic runs use the dataset's train/validation/test manifests.

## 1. Set up

Run commands from the repository root. Python 3.12 is used on the cluster; a
CUDA GPU is needed for the full experiment. Install the dependencies, then run
the GPU environment check on a CUDA machine before submitting jobs:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_training_env.py --gpu
```

The CPU gradient check can also run without CUDA:

```sh
WARP_CACHE_PATH=/tmp/warp-cache .venv/bin/python 3dgs_gradient_check_gpu.py --device cpu
```

Put each dataset in `data/<scene>/` as described in [dataset preparation](docs/data.md).
For a NeRF-synthetic scene, generate `init.ply` once and reuse it across all
kernel and seed runs:

```sh
.venv/bin/python generate_synthetic_init.py --data data/lego --seed 7 --points 50000
```

Real scenes initialize from their COLMAP points and do not need this step.
Each run records its resolved settings. Preserve the source dataset and, for
synthetic scenes, the generated `init.ply` to reproduce the exact inputs.

## 2. Train a scene

The runner maps array indices 0–8 to the three kernels and paired seeds 7, 17,
and 27. Inspect that mapping, then run one task directly on a GPU or submit the
nine-task Slurm array:

```sh
.venv/bin/python scripts/run_kernel_matrix.py --list
.venv/bin/python scripts/run_kernel_matrix.py --data data/bonsai \
  --config config/real_20k_200k.yaml --index 0 --arch gpu \
  --output-root outputs-my-real
```

For the full matrices on Slurm, create `logs/` first. Use a separate output root
for each configuration; the current real comparison uses 20,000 steps and a
200,000-splat cap, while synthetic uses 8,000 steps and 100,000 splats:

```sh
mkdir -p logs
OUTPUT_ROOT="$PWD/outputs-my-synthetic" sbatch scripts/slurm_train_synthetic.sh \
  data/lego config/synthetic.yaml
OUTPUT_ROOT="$PWD/outputs-my-real" sbatch scripts/slurm_train_real.sh \
  data/bonsai config/real_20k_200k.yaml
```

Repeat the appropriate array for every scene. Array indices can be retried
individually with `sbatch --array=INDEX ...`; see [cluster operations](docs/cluster.md).
Runs land in `<outputs>/kernels/<scene>/<kernel>/seed_<seed>/`, where
`<outputs>` means the output root you chose above. The scene lists and earlier
matrices are in the [run catalogue](docs/experiments.md).

## 3. Analyze completed runs

Analyze each matrix separately, using the same output roots as training. Set
the two paths below to the roots holding the completed main matrices. The
explicit scene lists prevent historical runs in a shared root from entering
the final tables:

```sh
REAL_ROOT=outputs-my-real
SYNTHETIC_ROOT=outputs-my-synthetic
.venv/bin/python analyze_kernels.py --input-root "$REAL_ROOT/kernels" \
  --output-dir "$REAL_ROOT/kernel_analysis" \
  --scenes bicycle bonsai counter flowers garden kitchen room stump treehill
.venv/bin/python scripts/report_kernels.py --input-root "$REAL_ROOT/kernels" \
  --scenes bicycle bonsai counter flowers garden kitchen room stump treehill
.venv/bin/python analyze_kernels.py --input-root "$SYNTHETIC_ROOT/kernels" \
  --output-dir "$SYNTHETIC_ROOT/kernel_analysis" \
  --scenes chair drums ficus hotdog lego materials mic ship
.venv/bin/python scripts/report_kernels.py --input-root "$SYNTHETIC_ROOT/kernels" \
  --scenes chair drums ficus hotdog lego materials mic ship
cat "$REAL_ROOT/kernel_analysis/missing_jobs.csv"
cat "$SYNTHETIC_ROOT/kernel_analysis/missing_jobs.csv"
```

Both `missing_jobs.csv` files should contain only a header. The scene-level
figures and `overview.csv` used by the report are under each root's `reports/`;
`paired_runs.csv` and validation curves are under `kernel_analysis/`. The
`gaussian_seed_baseline.csv` file gives the sample SD across Gaussian seeds
for each scene; `kernel_summary.csv` adds the SD of paired kernel differences
and their descriptive 95% intervals. The `reports/` directory also contains
the seed-variability tables in its scene reports and overview. The
[capacity diagnostic](docs/experiments.md#capacity-diagnostic) is separate.
These commands regenerate summaries from saved runs without retraining. A
fresh clone does not contain those runs or datasets. Retraining follows
Sections 1–2, but exact numerical replication also depends on the original
dataset files and software environment; `requirements.txt` is not a lockfile.

To export report-ready per-scene seed-variability figures for PSNR, SSIM, and
VGG LPIPS after generating both `kernel_analysis/` directories, run
`.venv/bin/python scripts/plot_seed_variability.py`. It writes PNG and PDF
versions to `docs/figures/`. Each figure separates the synthetic and real
matrices and shows Gaussian run SD alongside the paired-difference SD for
linear and quadratic kernels.

For paired per-scene SSIM and LPIPS plots matching the PSNR figure, run
`.venv/bin/python scripts/plot_paired_metrics.py`. It uses each matrix's
`paired_runs.csv` and writes PNG and PDF figures to `docs/figures/`.

## 4. Build and view the site

The site builder reads completed real runs, and optionally the eight synthetic
summaries. If additional real-scene cameras are wanted in the explorer, first
export them on the GPU with `scripts/slurm_render_views.sh` or
`scripts/export_kernel_views.py` (see [cluster operations](docs/cluster.md)).
Keep `view_renders/` beside `kernels/` under the selected output root.

Choose the real and synthetic run roots you want to present. For example:

```sh
REAL_ROOT=outputs-my-real
SYNTHETIC_ROOT=outputs-my-synthetic
.venv/bin/python scripts/build_kernel_site.py \
  --input-root "$REAL_ROOT/kernels" --synthetic-root "$SYNTHETIC_ROOT/kernels"
.venv/bin/python -m http.server 8765 --directory "$REAL_ROOT/site"
```

Open `http://localhost:8765/`. The generated site is written beside
the real `kernels/` directory by default; `--output-dir` changes that location.
The Analysis page's numerical interpretation describes the saved matrices and
must be updated for different data. Its bonsai DSCF5648 comparison images are
checked-in assets under `site/assets/analysis/`, so that example loads even
when a reader has no local dataset or render exports.
