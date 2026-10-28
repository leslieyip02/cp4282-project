# Screen-space kernels for 3D Gaussian splatting

This project compares Gaussian and linear radial splat footprints under the
same projected 3D ellipsoids and training budget. It measures held-out PSNR and
SSIM, training time, rendering time, splat count, and tile work. The full
methodology is in [docs/experiment.md](docs/experiment.md).

## Local setup

Install `requirements.txt` into `.venv`, then check the renderer:

```sh
.venv/bin/python scripts/check_training_env.py
WARP_CACHE_PATH=/tmp/warp-cache .venv/bin/python 3dgs_gradient_check_gpu.py --device cpu
```

A NeRF-synthetic scene needs its train/val/test manifests and images. Generate
one shared initialization if `init.ply` is missing:

```sh
.venv/bin/python generate_synthetic_init.py --data data/lego --seed 7 --points 50000
```

A Mip-NeRF 360 scene uses its COLMAP points; see [docs/data.md](docs/data.md).

## Run the comparison

List the six array indices for one scene:

```sh
.venv/bin/python scripts/run_kernel_matrix.py --list
```

Run a one-seed pilot on a CUDA machine (substitute `data/bonsai` and
`config/real.yaml` for a real scene):

```sh
.venv/bin/python kernel_experiment.py --data data/lego --config config/synthetic.yaml \
  --kernel gaussian --seed 7 --arch gpu
.venv/bin/python kernel_experiment.py --data data/lego --config config/synthetic.yaml \
  --kernel linear --seed 7 --arch gpu
.venv/bin/python analyze_kernels.py --scenes lego --seeds 7
```

On the cluster, submit each scene's six-task GPU array after the project code
and dataset are present. Create `logs/` before submitting:

```sh
mkdir -p logs
sbatch scripts/slurm_train_synthetic.sh data/lego config/synthetic.yaml
sbatch scripts/slurm_train_real.sh data/bonsai config/real.yaml
```

After all arrays finish, run:

```sh
sbatch scripts/slurm_analyze_kernels.sh
cat outputs/kernel_analysis/missing_jobs.csv
```

Results are stored under `outputs/kernels/<scene>/<kernel>/seed_<seed>/`.
Historical filter outputs in `outputs/analysis/` are retained as prior data
and are not read by the kernel analysis. See [docs/cluster.md](docs/cluster.md)
for Slurm retries and environment checks.
