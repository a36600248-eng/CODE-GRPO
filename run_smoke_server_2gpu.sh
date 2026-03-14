#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_smoke_server_2gpu.sh <seed> [port]"
  exit 1
fi

SEED=$1
PORT=${2:-8010}
ORCH_LOG=/root/autodl-tmp/CODE-GRPO/trl/smoke_orchestrator_${PORT}.log

log() {
  echo "[$(date '+%F %T')] $*"
}

exec > >(tee -a "${ORCH_LOG}") 2>&1

log "Smoke orchestrator starting: seed=${SEED} port=${PORT}"

cd ~/autodl-tmp/CODE-GRPO/trl
log "Changed directory to $(pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
log "Activated conda env: codegrpo"

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
SERVER_LOG=/root/autodl-tmp/CODE-GRPO/trl/vllm_server_smoke_${PORT}.log

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

log "Checking whether port ${PORT} already has a live server"
if healthcheck; then
  log "Port ${PORT} already has a live server. Stop it or use another port."
  exit 1
fi
if port_in_use; then
  log "Port ${PORT} is already in use by another process. Stop it or use another port."
  exit 1
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping smoke vLLM server process group: ${SERVER_PID}"
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

log "Starting smoke vLLM server on GPU0, port=${PORT}"
setsid env CUDA_VISIBLE_DEVICES=0 python -m trl.cli.main vllm-serve \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor_parallel_size 1 \
  --data_parallel_size 1 \
  --gpu_memory_utilization 0.75 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

log "Waiting for vLLM server health endpoint..."
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "vLLM server process exited before becoming healthy. Tail of ${SERVER_LOG}:"
    tail -n 100 "${SERVER_LOG}" || true
    exit 1
  fi
  if healthcheck; then
    break
  fi
  sleep 2
done

if ! healthcheck; then
  log "vLLM server failed to start. Tail of ${SERVER_LOG}:"
  tail -n 100 "${SERVER_LOG}" || true
  exit 1
fi

log "vLLM server healthy on port ${PORT}"

run_train() {
  local cfg="$1"
  log "Running smoke train config: ${cfg}"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${PORT}"
}

run_eval() {
  local cfg="$1"
  log "Running smoke eval config: ${cfg}"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${PORT}"
}

CONFIG_ROOT=configs/comparison/server_2gpu_smoke

run_eval ${CONFIG_ROOT}/raw_qwen7b_eval_mbpp.yaml
run_train ${CONFIG_ROOT}/codegrpo_method_mbpp.yaml
run_train ${CONFIG_ROOT}/vanilla_grpo_multiround_k2_mbpp.yaml
run_train ${CONFIG_ROOT}/vanilla_grpo_single_round_k8_mbpp.yaml

log "Smoke orchestrator finished successfully"
