#!/bin/bash
#SBATCH --job-name=gs-views
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=xgpd|xgpf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f scripts/export_kernel_views.py ]]; then
  echo "Submit this script from the cp4282-project root." >&2
  exit 1
fi
if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

data_dir="${1:?Usage: sbatch scripts/slurm_render_views.sh DATASET_DIR}"
if [[ ! -f "$data_dir/nb-info.json" ]]; then
  echo "Real-scene dataset missing: $data_dir" >&2
  exit 1
fi
seeds=(7 17 27)
index="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
if [[ ! "$index" =~ ^[0-2]$ ]]; then
  echo "Invalid Slurm array index: $index (expected 0-2)" >&2
  exit 1
fi
scene="${DATASET_NAME:-$(basename "$data_dir")}"
output_root="${OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/outputs-20k}"
python_bin="${PYTHON_BIN:-python}"
"$python_bin" scripts/check_training_env.py --gpu
"$python_bin" scripts/export_kernel_views.py \
  --input-root "$output_root/kernels" \
  --output-root "$output_root/view_renders" \
  --scene "$scene" --data "$data_dir" --seeds "${seeds[$index]}" \
  --train-views "${TRAIN_VIEWS:-3}" \
  --validation-views "${VALIDATION_VIEWS:-8}" --arch gpu
