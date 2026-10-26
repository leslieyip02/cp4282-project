#!/bin/bash
#SBATCH --job-name=gs-analyze
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f analyze_synthetic.py ]]; then
  echo "Submit this script from the cp4282-project root." >&2
  exit 1
fi
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

output_root="${OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/outputs}"
"${PYTHON_BIN:-python}" analyze_synthetic.py \
  --input-root "$output_root" \
  --output-dir "$output_root/analysis" \
  "$@"
