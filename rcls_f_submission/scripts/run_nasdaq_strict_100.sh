#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SUBMISSION_ROOT}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${SUBMISSION_ROOT}/logs" "${SUBMISSION_ROOT}/results"

MODELS=("stockmixer" "rcls_f_k1" "rcls_f_k3")

for MODEL in "${MODELS[@]}"; do
  echo "Running strict 100-epoch NASDAQ ${MODEL}"
  python "${SUBMISSION_ROOT}/code/src/train.py" \
    --dataset NASDAQ \
    --model "${MODEL}" \
    --seed 0 \
    --numpy-seed 123456789 \
    --torch-seed 12345678 \
    --epochs 100 \
    --patience 0 \
    --require-gpu 3090 \
    --dataset-root "${REPO_ROOT}/dataset" \
    --output-root "${SUBMISSION_ROOT}" \
    --activation hardswish \
    --main-mixer-activation hardswish \
    --scale-mixer-activation gelu \
    --stock-activation hardswish \
    2>&1 | tee "${SUBMISSION_ROOT}/logs/strict100_${MODEL}_NASDAQ_seed0.log"
done

python "${SCRIPT_DIR}/summarize_results.py" --output-root "${SUBMISSION_ROOT}"
python "${SCRIPT_DIR}/evaluate_stress.py" --output-root "${SUBMISSION_ROOT}"
python "${SCRIPT_DIR}/evaluate_selective.py" --output-root "${SUBMISSION_ROOT}"
python "${SCRIPT_DIR}/profile_efficiency.py" --output-root "${SUBMISSION_ROOT}"
