# CP4282: image filtering during 3D Gaussian splatting training

This project builds on the Assignment 2 Warp trainer and Assignment 1's rendering
work. It runs the proposal's controlled experiment on any dataset with the
NeRF-synthetic layout, including **lego**, **materials**, and **ship**. The same
posed training views are used in every method. Validation and test images are
always unfiltered.
Training-camera draws use a separate seeded random stream from densification,
so runs with the same seed see the same camera sequence even if their splat
populations evolve differently.

## Setup and data

Install `requirements.txt` in a Python environment with a working Warp device.
Warp is pinned to 1.17.0 because its Linux x86-64 PyPI build uses CUDA 12.9
and supports `sm_70` GPUs. Newer default PyPI builds use CUDA 13.4 and require `sm_75` or
newer; see [NVIDIA's Warp installation guide](https://nvidia.github.io/warp/v1.17/user_guide/installation.html)
and [current CUDA requirements](https://nvidia.github.io/warp/latest/user_guide/installation.html#cuda-requirements).
Each scene directory must contain `transforms_train.json`, `transforms_val.json`,
`transforms_test.json`, their referenced images, and `init.ply` (or supply
`--init-ply`). The supplied Lego data from Assignment 2 can be used directly:

```sh
python synthetic_experiment.py --data ../cp4282-as2/data/lego \
  --method gaussian --seed 7 --arch cpu
```

The runner defaults to 256 × 256, 8,000 training steps, and a phase switch after
step 2,000. Set these in `config/synthetic.yaml` or override the switch with
`--phase-switch-step`. Step `N` is the last filtered-target step; step `N + 1`
starts original-target refinement. `--phase-switch-step 0` uses original targets
throughout. The `none` method is the original-target baseline regardless of the
configured switch. The switch replaces only view-dependent buffers and camera
intrinsics; splats, Adam optimizers, and their moments stay in place. The trainer's
ordinary densification behavior is otherwise unchanged, including its own
optimizer reset when the splat population changes. If a scheduled densification,
late prune, or opacity reset falls exactly on the configured switch step, that
event is skipped for **every** method so no run loses Adam state at the boundary.

## Methods and measurements

`none`, `downsample`, `mean`, `gaussian`, `bilateral`, `kuwahara`, and `unsharp`
implement the proposal's seven methods. The five spatial filters operate on
full-resolution RGB targets after alpha compositing. Downsampling uses a 2 × 2
area average, half-resolution training buffers, and half focal length during
warm-up. Its held-out validation still renders at full resolution. Both image
dimensions must be even for this method.

Prepared filtered training arrays are cached in `outputs/preprocessed/` and
shared across seeds. To prepare all six alternatives before scheduling jobs:

```sh
python prepare_synthetic.py --data ../cp4282-as2/data/lego
```

`validation_metrics.csv` records per-view PSNR and SSIM on the same fixed
validation cameras at step 0 and every 500 steps. `test_metrics.csv` and
`test_summary.json` cover the full unfiltered test split after step 8,000.
The runner also saves the final PLY, timing logs, run configuration, and periodic
images/PLYs under `outputs/<scene>/<method>/seed_<seed>/`.

Each dataset uses a 21-job matrix (7 methods × 3 seeds). The array mapping is
independent of the dataset name:

```sh
python scripts/run_synthetic_matrix.py --list
python scripts/run_synthetic_matrix.py --data /path/to/scene --index 0 --arch gpu
```

After the jobs finish, `python analyze_synthetic.py` discovers dataset labels
under `outputs/` and writes four CSVs in `outputs/analysis/`:

- `paired_runs.csv`: one row per completed run, with final test PSNR/SSIM,
  same-dataset/same-seed changes from the `none` baseline, training time, and
  final Gaussian count.
- `method_summary.csv`: per-dataset/per-method mean results across paired seeds,
  including mean and standard deviation of the PSNR change.
- `validation_curves.csv`: mean validation PSNR, SSIM, and elapsed time at each
  sampled iteration.
- `missing_jobs.csv`: expected dataset/method/seed combinations without a
  complete result, for reruns.

The analysis reads CSV and JSON files only; it needs one CPU and no GPU. It
prints the number of expected jobs that are missing or incomplete. It does not
currently create plots.

## SoC Slurm scripts

Submit from the project root after setting up `.venv` on the cluster. Pass one
dataset directory to each preprocessing and training submission. The output and
cache label defaults to that directory's name; set `DATASET_NAME` if two dataset
paths share the same final directory name. The scripts use the GPU request and
submit-directory convention from Assignment 1's Slurm examples. Training
requests one NVIDIA GPU, four CPUs, 16 GB, and eight hours per task; adjust
these with `sbatch` options after a cluster pilot. Slurm writes `.out` and
`.err` files under `logs/`. Keep that directory present on the cluster before
submitting: Slurm opens its log paths before the script body runs, so the job
cannot create the directory for itself.

```sh
sbatch scripts/slurm_prepare_synthetic.sh /path/to/scene
sbatch --dependency=afterok:<prep_job_id> scripts/slurm_train_synthetic.sh /path/to/scene
sbatch --dependency=afterany:<train_array_job_id> scripts/slurm_analyze_synthetic.sh
```

Preprocessing is one CPU job per dataset. Training is a 21-task GPU array per
dataset, with at most four tasks running at once. Submit the same pair again for
each additional scene, then run analysis after all training arrays finish. Pass
the same `OUTPUT_ROOT` environment variable to all jobs if results should be
stored outside the project.

`config/synthetic.yaml` holds shared training settings. You can pass a modified
copy as the second argument to both the preprocessing and training scripts:
`sbatch scripts/slurm_train_synthetic.sh /path/to/scene config/my_scene.yaml`.
The array overrides the YAML's dataset path, method, and seed for each task.
Every run saves its resolved configuration in `run_config.json`, which avoids maintaining
21 near-identical YAML files per dataset. For a special one-off run, copy
`config/synthetic.yaml`, set `paths.data`, `experiment.method`, and `runtime.seed`,
then submit `sbatch scripts/slurm_train_one.sh config/my_run.yaml`. Paths inside
YAML are resolved relative to that YAML file. The runner derives its output path
from `OUTPUT_ROOT/<dataset>/<method>/seed_<seed>/`; `paths.output` in the input
YAML is a placeholder and is replaced with that path. If you use seeds beyond the array's
7, 17, and 27, pass them to the CPU analysis script, for example
`sbatch scripts/slurm_analyze_synthetic.sh --seeds 7 17 27 42`.

If a batch job reports a missing Python package even though `pip list` shows it,
check the exact interpreter used by the job:

```sh
.venv/bin/python -m pip show numpy
.venv/bin/python -c 'import numpy; print(numpy.__version__)'
.venv/bin/python -m pip install -r requirements.txt
```

The GPU scripts now print their interpreter path and check imports before
starting an array task. Create the virtual environment on the cluster rather
than copying a local machine's `.venv`.

Validate the renderer gradients and the short end-to-end phase-switch fixture:

```sh
python 3dgs_gradient_check_gpu.py --device cpu
python -m unittest discover -s tests -v
```

## TODO: real scenes

Revisit Mip-NeRF 360 **bonsai** and **counter** after the synthetic methodology is
working. Their COLMAP data needs a loader and point-cloud initialization, and
their intrinsics/distortion and image aspect ratios must be handled consistently
by the renderer, warm-up switch, and evaluator. The current runner accepts
NeRF-synthetic-format scenes only.
