#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_compare_raw_pure_soft_server_2gpu.sh <seed> [base_port]"
  exit 1
fi

SEED=$1
BASE_PORT=${2:-8012}

bash run_raw_eval_server_2gpu.sh "${SEED}" "${BASE_PORT}"
bash run_pure_grpo_train_eval_server_2gpu.sh "${SEED}" "$((BASE_PORT + 4))"
bash run_soft_grpo_train_eval_server_2gpu.sh "${SEED}" "$((BASE_PORT + 8))"
