#!/bin/bash
#SBATCH --job-name=gs-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f prepare_synthetic.py ]]; then
  echo "Submit this script from the cp4282-project root." >&2
  exit 1
fi
if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

data_dir="${1:?Usage: sbatch scripts/slurm_prepare_synthetic.sh DATASET_DIR [CONFIG_YAML]}"
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
name_arg=()
if [[ -n "${DATASET_NAME:-}" ]]; then
  name_arg=(--scene "$DATASET_NAME")
fi

python_bin="${PYTHON_BIN:-python}"
"$python_bin" scripts/check_training_env.py
"$python_bin" prepare_synthetic.py \
  --data "$data_dir" \
  --config "$config" \
  --init-seed "${INIT_SEED:-7}" \
  --cache-dir "$output_root/preprocessed" \
  "${name_arg[@]}"
