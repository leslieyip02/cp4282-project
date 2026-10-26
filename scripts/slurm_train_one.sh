#!/bin/bash
#SBATCH --job-name=gs-one
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f synthetic_experiment.py ]]; then
  echo "Submit this script from the cp4282-project root." >&2
  exit 1
fi
if [[ -f .venv/bin/activate ]]; then
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
"$python_bin" scripts/check_training_env.py
"$python_bin" synthetic_experiment.py \
  --config "$config" \
  --arch gpu \
  --output-root "$output_root" \
  "${name_arg[@]}"
