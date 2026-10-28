#!/bin/bash
#SBATCH --job-name=gs-one
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=xgpd|xgpf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f kernel_experiment.py ]]; then
  echo "Submit this script from the cp4282-project root." >&2
  exit 1
fi
if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

config="${1:?Usage: sbatch scripts/slurm_train_one.sh RUN_CONFIG_YAML}"
if [[ ! -f "$config" ]]; then
  echo "Run configuration missing: $config" >&2
  exit 1
fi
output_root="${OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/outputs}"
name_arg=()
if [[ -n "${DATASET_NAME:-}" ]]; then
  name_arg=(--scene "$DATASET_NAME")
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
"$python_bin" kernel_experiment.py \
  --config "$config" \
  --arch gpu \
  --output-root "$output_root" \
  "${name_arg[@]}"
