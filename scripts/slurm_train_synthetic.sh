#!/bin/bash
#SBATCH --job-name=gs-filter
#SBATCH --array=0-20%4
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=xgpd|xgpf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f synthetic_experiment.py ]]; then
  echo "Submit this script from the cp4282-project root." >&2
  exit 1
fi
if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

data_dir="${1:?Usage: sbatch scripts/slurm_train_synthetic.sh DATASET_DIR [CONFIG_YAML]}"
config="${2:-config/synthetic.yaml}"
output_root="${OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/outputs}"
if [[ ! -d "$data_dir" ]]; then
  echo "Dataset directory missing: $data_dir" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "Training configuration missing: $config" >&2
  exit 1
fi
index="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
if [[ ! "$index" =~ ^[0-9]+$ ]]; then
  echo "Invalid Slurm array index: $index" >&2
  exit 1
fi
name_arg=()
if [[ -n "${DATASET_NAME:-}" ]]; then
  name_arg=(--name "$DATASET_NAME")
fi

python_bin="${PYTHON_BIN:-python}"
echo "Compute node: $(hostname)"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi is unavailable on this node."
fi
"$python_bin" scripts/check_training_env.py --gpu
"$python_bin" scripts/run_synthetic_matrix.py \
  --data "$data_dir" \
  --config "$config" \
  --index "$index" \
  --arch gpu \
  --output-root "$output_root" \
  "${name_arg[@]}"
