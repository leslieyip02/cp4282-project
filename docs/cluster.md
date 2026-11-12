# Cluster operations

Use the [README](../README.md) for the complete setup-to-report sequence and
the [catalogue](experiments.md) for scene lists and saved configurations. Run
Slurm commands from the repository root. Create `logs/` before `sbatch`, since
Slurm opens log paths before the job script starts. The training scripts
activate `.venv` when present and check CUDA/Warp on the allocated GPU.

The scripts request one `gpu:nv` on the `xgpd` or `xgpf` node families. The
cluster environment uses Python 3.12 and `warp-lang==1.17.0`. Before a matrix,
run `.venv/bin/python scripts/check_training_env.py --gpu` on a GPU node and
check the gradients with `3dgs_gradient_check_gpu.py --device cuda:0`. Job logs
include the node, GPU allocation, `nvidia-smi`, and the environment check.

## Submit and retry

Pass the configuration as the second argument. In particular, the real script's
default is the **earlier** 8k/100k `config/real.yaml`; the current real matrix
requires the explicit 20k/200k config. Choose a new `OUTPUT_ROOT` for a new
matrix. These examples submit all nine kernel/seed tasks for one scene:

```sh
mkdir -p logs
OUTPUT_ROOT="$PWD/outputs-my-real" sbatch scripts/slurm_train_real.sh \
  data/bonsai config/real_20k_200k.yaml
OUTPUT_ROOT="$PWD/outputs-my-synthetic" sbatch scripts/slurm_train_synthetic.sh \
  data/lego config/synthetic.yaml
```

The same array indexes work for both scripts. To retry a failed task, find its
index with `.venv/bin/python scripts/run_kernel_matrix.py --list`, then submit
only that index, preserving the original output root and config:

```sh
OUTPUT_ROOT="$PWD/outputs-my-real" sbatch --array=INDEX \
  scripts/slurm_train_real.sh data/bonsai config/real_20k_200k.yaml
```

Use `sacct -j JOB_ID --format=JobID,NodeList,State,ExitCode` to inspect a
failed allocation. If the node is faulty, add `--exclude=BAD_NODE` to the
retry. `DATASET_NAME` can disambiguate dataset directories with the same
basename; `PYTHON_BIN` selects another existing interpreter.

After all scene arrays finish, run analysis for the chosen root. This writes
`kernel_analysis/missing_jobs.csv`; check and retry any missing jobs before
generating final reports:

```sh
OUTPUT_ROOT="$PWD/outputs-my-real" sbatch scripts/slurm_analyze_kernels.sh
cat outputs-my-real/kernel_analysis/missing_jobs.csv
.venv/bin/python scripts/report_kernels.py --input-root outputs-my-real/kernels
```

If a run tree contains more than one configuration, pass an explicit scene list
to the analysis script. Its output CSVs are overwritten when you re-run it for
a different subset, so give each subset a separate `--output-dir` when running
`analyze_kernels.py` directly.

## Export more camera views for the site

The builder can use existing snapshot PNGs without the original datasets.
To show more train and validation cameras in the explorer, export them on a
CUDA node with the datasets and saved PLYs available. For bonsai seed 17:

```sh
OUTPUT_ROOT="$PWD/outputs-my-real" STEPS="10000 20000" TRAIN_VIEWS=0 VALIDATION_VIEWS=3 \
  sbatch --array=1 scripts/slurm_render_views.sh data/bonsai
```

Each array index corresponds to a seed (0, 1, 2 for 7, 17, 27). By default,
the exporter selects three train cameras and eight reserved validation
cameras, with saved steps from 6k onward. It skips images already written, so
failed tasks can be rerun. The direct alternative is:

```sh
.venv/bin/python scripts/export_kernel_views.py --input-root outputs-my-real/kernels \
  --scene bonsai --data data/bonsai --seeds 17 --arch gpu
```

Copy `<outputs>/view_renders/` beside `<outputs>/kernels/` when building the
site on another machine, then run the README's site command. The four bonsai
DSCF5648 Analysis images are checked in under `site/assets/analysis/` and are
copied by the builder independently of view exports.

LPIPS uses the VGG backbone on final test views. To backfill it for an older
checkpoint without retraining, run
`.venv/bin/python scripts/backfill_lpips.py RUN_DIR DATA_DIR --arch gpu`, then
repeat the analysis. Keep final test views out of checkpoint selection.
