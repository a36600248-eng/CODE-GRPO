#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash run_dual_gpu_server.sh <config_path> <seed> [port]"
  exit 1
fi

CONFIG_PATH=$1
SEED=$2
PORT=${3:-8000}

cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
SERVER_LOG=/root/autodl-tmp/CODE-GRPO/trl/vllm_server_${PORT}.log

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

echo "Starting vLLM server on GPU0, port=${PORT}"
CUDA_VISIBLE_DEVICES=0 python -m trl.cli.main vllm-serve \
  --model "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor_parallel_size 1 \
  --data_parallel_size 1 \
  --gpu_memory_utilization 0.85 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM server to become healthy..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health/" > /dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${PORT}/health/" > /dev/null; then
  echo "vLLM server failed to start. Tail of ${SERVER_LOG}:"
  tail -n 100 "${SERVER_LOG}" || true
  exit 1
fi

echo "Starting training on GPU1 with config=${CONFIG_PATH}"
CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --vllm_server_base_url "http://127.0.0.1:${PORT}"
