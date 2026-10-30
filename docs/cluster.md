# Cluster notes

The README contains the full local-to-cluster playbook. Run Slurm commands
from the repository root on the cluster. Create `logs/` before submitting:
Slurm opens its log files before the job script starts.

Install `requirements.txt` into a Python 3.12 environment on the cluster and
run `.venv/bin/python scripts/check_training_env.py` before submitting jobs.
Kernel experiments calculate VGG LPIPS on the final test views by default.
Set `reporting.lpips_net: none` only when intentionally skipping that metric.
For existing checkpoints, backfill LPIPS without retraining:

```sh
.venv/bin/python scripts/backfill_lpips.py outputs/kernels/bonsai/gaussian/seed_7 data/bonsai --arch gpu
```

This updates that run's `test_metrics.csv` and `test_summary.json`. Re-run
`analyze_kernels.py` afterward to add paired LPIPS columns to the analysis.
The training scripts activate `.venv` and check Warp CUDA access inside the
allocated job. `warp-lang` is pinned to 1.17.0 for the cluster's `sm_70` GPUs;
see [NVIDIA's installation guide](https://nvidia.github.io/warp/v1.17/user_guide/installation.html).

GPU scripts request one `gpu:nv` on the `xgpd` or `xgpf` node families. This
selects supported hardware but cannot guarantee that a node's driver is
healthy. The job log includes the node name, GPU allocation, `nvidia-smi`
output, and the Warp CUDA check. To inspect previous allocations and avoid a
faulty node on the next submission:

```sh
sacct -j JOB_ID --format=JobID,NodeList,State,ExitCode
sbatch --exclude=BAD_NODE --array=INDEX_LIST scripts/slurm_train_real.sh data/bonsai
```

After an array finishes, inspect `outputs/kernel_analysis/missing_jobs.csv`. Match
its kernel/seed rows with the matrix listing, then resubmit only those array
indices using the real or synthetic training script. Run analysis again after
the retries. The baseline pilot is index 0:

```sh
.venv/bin/python scripts/run_kernel_matrix.py --list
sbatch --array=0 scripts/slurm_train_real.sh data/bonsai
```

For a synthetic scene, select `config/synthetic.yaml` and
`scripts/slurm_train_synthetic.sh` in the README playbook. Its dataset
directory must contain the three transforms manifests, their images, and
`init.ply`. To analyze several scenes together, wait for every training array
and run the analysis script without `--scenes`; it discovers all scenes under
`outputs/kernels/`. A per-scene analysis writes to the same `outputs/kernel_analysis/` CSVs,
so save those results before analyzing another scene separately.

Pass a modified YAML as the second argument to the training scripts.
Set `OUTPUT_ROOT` to move outputs, or
`DATASET_NAME` when two dataset directories have the same final name. The
scripts accept `PYTHON_BIN` to use a different existing interpreter.

Generate a report for any copied kernel matrix. The script discovers scenes and
seeds under `--input-root`; pass `--scene` or `--scenes` to select a subset. An
optional reference root adds a budget comparison for matching configurations:

```sh
.venv/bin/python scripts/report_kernels.py --input-root outputs-20k/kernels \
  --reference-root outputs/kernels
```

The overview is written to `outputs-20k/reports/overview.md`, with per-scene
reports and plots in `outputs-20k/reports/<scene>/`. The script excludes
reference runs from budget comparisons when settings or held-out splits differ.

## Snapshot comparison site

Build the static site from copied runs. It needs no web dependencies or server
backend. The existing snapshot PNGs supply one fixed validation camera per
run; PLY files supply splat counts and scale/opacity distributions at every
saved step. `data/<scene>` is not required for this first build.

```sh
.venv/bin/python scripts/build_kernel_site.py --input-root outputs-20k/kernels
cd outputs-20k/site
python -m http.server 8765
```

Open `http://localhost:8765/`. The builder writes a self-contained directory
at `outputs-20k/site/` and can be rerun after more rendered views are copied.
The site compares the same step, seed, and camera across kernels and has a
summary tab for the final held-out test metrics. Do not use test views to
select a training checkpoint or tune settings.

To add train and validation cameras, run the exporter on the cluster where the
dataset and CUDA GPU are available. It selects three spread-out train cameras
and all eight reserved validation cameras by default, renders the same camera
from each saved PLY from 6k onward, and saves resized source photos.

```sh
.venv/bin/python scripts/export_kernel_views.py --input-root outputs-20k/kernels \
  --scene bonsai --data data/bonsai --seeds 17 --arch gpu
```

Copy `outputs-20k/view_renders/` alongside the copied `kernels/` directory,
then rerun `build_kernel_site.py`. Use `--steps`, `--train-views`, and
`--validation-views` to limit export size. The exporter skips images already
written, so a timed-out job can be rerun. Snapshot distributions are computed
from saved PLYs: scale is the geometric mean of the three world-space axes,
opacity is the sigmoid of the stored logit, and the displayed opacity share
counts splats with opacity at least 0.5. These are scene-level diagnostics,
not projected pixel footprint measurements.

For the 20,000-step real-scene matrix, use `config/real_20k.yaml` and a fresh
output root. For example:

```sh
OUTPUT_ROOT="$PWD/outputs-20k" sbatch scripts/slurm_train_real.sh data/bonsai config/real_20k.yaml
```

Repeat for each real scene. The same array indexes cover all three kernels and
seeds 7, 17, and 27. Analyze the matrix with `--input-root outputs-20k/kernels`
and `--output-dir outputs-20k/kernel_analysis`. The new runs record per-process
sampled peak GPU memory during training and final test evaluation. Check that
the Slurm environment can import `pynvml` and query NVML before submitting;
`scripts/check_training_env.py --gpu` performs this check.
