# Cluster notes

The README contains the full local-to-cluster playbook. Run Slurm commands
from the repository root on the cluster. Create `logs/` before submitting:
Slurm opens its log files before the job script starts.

Install `requirements.txt` into a Python 3.12 environment on the cluster and
run `.venv/bin/python scripts/check_training_env.py` before submitting jobs.
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
