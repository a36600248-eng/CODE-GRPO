#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_all_server_2gpu.sh <seed> [port]"
  exit 1
fi

SEED=$1
PORT=${2:-8000}

cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
SERVER_LOG=/root/autodl-tmp/CODE-GRPO/trl/vllm_server_${PORT}.log

healthcheck() {
  curl -sf --connect-timeout 2 --max-time 3 "http://127.0.0.1:${PORT}/health/" > /dev/null
}

port_in_use() {
  python - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(0)
else:
    sys.exit(1)
finally:
    s.close()
PY
}

echo "Checking whether port ${PORT} already has a live server"
if healthcheck; then
  echo "Port ${PORT} already has a live server. Stop it or use another port."
  exit 1
fi
if port_in_use; then
  echo "Port ${PORT} is already in use by another process. Stop it or use another port."
  exit 1
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping vLLM server process group: ${SERVER_PID}"
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

echo "Starting official TRL vLLM server sync path on GPU0, port=${PORT}"
setsid env CUDA_VISIBLE_DEVICES=0 python -m trl.cli.main vllm-serve \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor_parallel_size 1 \
  --data_parallel_size 1 \
  --gpu_memory_utilization 0.85 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM server health endpoint..."
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server process exited before becoming healthy. Tail of ${SERVER_LOG}:"
    tail -n 100 "${SERVER_LOG}" || true
    exit 1
  fi
  if healthcheck; then
    break
  fi
  sleep 2
done

if ! healthcheck; then
  echo "vLLM server failed to start. Tail of ${SERVER_LOG}:"
  tail -n 100 "${SERVER_LOG}" || true
  exit 1
fi

run_train() {
  local cfg="$1"
  echo "Running train config: ${cfg}"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${PORT}"
}

run_eval() {
  local cfg="$1"
  echo "Running eval config: ${cfg}"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${PORT}"
}

CONFIG_ROOT=configs/comparison/server_2gpu

run_eval ${CONFIG_ROOT}/raw_qwen7b_eval_mbpp.yaml
run_train ${CONFIG_ROOT}/codegrpo_method_mbpp.yaml
run_train ${CONFIG_ROOT}/vanilla_grpo_multiround_k2_mbpp.yaml
run_train ${CONFIG_ROOT}/vanilla_grpo_single_round_k8_mbpp.yaml
