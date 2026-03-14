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

log "Smoke orchestrator starting: seed=${SEED} base_port=${PORT}"

cd ~/autodl-tmp/CODE-GRPO/trl
log "Changed directory to $(pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
log "Activated conda env: codegrpo"

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct

healthcheck() {
  local port="$1"
  curl -sf --connect-timeout 2 --max-time 3 "http://127.0.0.1:${port}/health/" > /dev/null
}

port_in_use() {
  local port="$1"
  python - "$port" <<'PY'
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

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping smoke vLLM server process group: ${SERVER_PID}"
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT

start_server() {
  local port="$1"
  local server_log="/root/autodl-tmp/CODE-GRPO/trl/vllm_server_smoke_${port}.log"

  log "Checking whether port ${port} already has a live server"
  if healthcheck "$port"; then
    log "Port ${port} already has a live server. Stop it or use another port."
    exit 1
  fi
  if port_in_use "$port"; then
    log "Port ${port} is already in use by another process. Stop it or use another port."
    exit 1
  fi

  log "Starting smoke vLLM server on GPU0, port=${port}"
  setsid env CUDA_VISIBLE_DEVICES=0 python -m trl.cli.main vllm-serve \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tensor_parallel_size 1 \
    --data_parallel_size 1 \
    --gpu_memory_utilization 0.75 \
    > "${server_log}" 2>&1 &
  SERVER_PID=$!

  log "Waiting for vLLM server health endpoint on port ${port}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "vLLM server process exited before becoming healthy. Tail of ${server_log}:"
      tail -n 100 "${server_log}" || true
      exit 1
    fi
    if healthcheck "$port"; then
      break
    fi
    sleep 2
  done

  if ! healthcheck "$port"; then
    log "vLLM server failed to start. Tail of ${server_log}:"
    tail -n 100 "${server_log}" || true
    exit 1
  fi

  log "vLLM server healthy on port ${port}"
}

run_train() {
  local cfg="$1"
  local port="$2"
  local group_port="$((port + 40000))"
  log "Running smoke train config: ${cfg} (port=${port}, group_port=${group_port})"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${port}" \
    --vllm_group_port "${group_port}"
}

run_eval() {
  local cfg="$1"
  local port="$2"
  local group_port="$((port + 40000))"
  log "Running smoke eval config: ${cfg} (port=${port}, group_port=${group_port})"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${port}" \
    --vllm_group_port "${group_port}"
}

CONFIG_ROOT=configs/comparison/server_2gpu_smoke

STAGE_PORT=$PORT

start_server "${STAGE_PORT}"
run_eval ${CONFIG_ROOT}/raw_qwen7b_eval_mbpp.yaml "${STAGE_PORT}"
cleanup

STAGE_PORT=$((PORT + 1))
start_server "${STAGE_PORT}"
run_train ${CONFIG_ROOT}/codegrpo_method_mbpp.yaml "${STAGE_PORT}"
cleanup

STAGE_PORT=$((PORT + 2))
start_server "${STAGE_PORT}"
run_train ${CONFIG_ROOT}/vanilla_grpo_multiround_k2_mbpp.yaml "${STAGE_PORT}"
cleanup

STAGE_PORT=$((PORT + 3))
start_server "${STAGE_PORT}"
run_train ${CONFIG_ROOT}/vanilla_grpo_single_round_k8_mbpp.yaml "${STAGE_PORT}"
cleanup

log "Smoke orchestrator finished successfully"
