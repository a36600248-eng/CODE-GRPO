#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_raw_eval_server_2gpu.sh <seed> [port]"
  exit 1
fi

SEED=$1
PORT=${2:-8012}
ORCH_LOG=/root/autodl-tmp/CODE-GRPO/trl/raw_eval_orchestrator_${PORT}.log

log() {
  echo "[$(date '+%F %T')] $*"
}

exec > >(tee -a "${ORCH_LOG}") 2>&1

log "Raw eval orchestrator starting: seed=${SEED} base_port=${PORT}"

cd ~/autodl-tmp/CODE-GRPO/trl
log "Changed directory to $(pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
log "Activated conda env: codegrpo"

pkill -f "trl.cli.main vllm-serve" || true
pkill -f "VLLM::EngineCore" || true
pkill -f "python -m trl.cli.main code_grpo" || true
pkill -f "python -m trl.cli.main code_grpo_eval" || true
sleep 3

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export GLOO_SOCKET_IFNAME=lo
export VLLM_HOST_IP=127.0.0.1

MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
CONFIG_PATH=configs/comparison/server_2gpu_smoke/raw_qwen7b_eval_mbpp.yaml

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
    log "Stopping raw eval vLLM server process group: ${SERVER_PID}"
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" || true
  fi
  SERVER_PID=""
  sleep 2
}
trap cleanup EXIT

choose_stage_port() {
  local candidate="$1"
  local candidate_group_port
  while true; do
    candidate_group_port=$((candidate + 40000))
    if healthcheck "$candidate"; then
      log "Port ${candidate} already has a live server, trying next port" >&2
      candidate=$((candidate + 1))
      continue
    fi
    if port_in_use "$candidate"; then
      log "Port ${candidate} is already in use, trying next port" >&2
      candidate=$((candidate + 1))
      continue
    fi
    if port_in_use "$candidate_group_port"; then
      log "Group port ${candidate_group_port} is already in use, trying next port" >&2
      candidate=$((candidate + 1))
      continue
    fi
    echo "$candidate"
    return
  done
}

start_server() {
  local port="$1"
  local server_log="/root/autodl-tmp/CODE-GRPO/trl/vllm_server_raw_eval_${port}.log"

  log "Starting raw eval vLLM server on GPU0, port=${port}"
  setsid env CUDA_VISIBLE_DEVICES=0 python -m trl.cli.main vllm-serve     --model "${MODEL_PATH}"     --tokenizer "${MODEL_PATH}"     --host 127.0.0.1     --port "${port}"     --tensor_parallel_size 1     --data_parallel_size 1     --gpu_memory_utilization 0.75     > "${server_log}" 2>&1 &
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

run_eval() {
  local port="$1"
  local group_port="$((port + 40000))"
  local output_dir="/root/autodl-tmp/CODE-GRPO/trl/runs_raw_eval_seed${SEED}"
  log "Running raw eval: ${CONFIG_PATH}"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval     --config "${CONFIG_PATH}"     --seed "${SEED}"     --data_seed "${SEED}"     --output_dir "${output_dir}"     --vllm_server_base_url "http://127.0.0.1:${port}"     --vllm_group_port "${group_port}"
  log "Raw eval finished. Output dir: ${output_dir}"
}

STAGE_PORT=$(choose_stage_port "$PORT")
start_server "${STAGE_PORT}"
run_eval "${STAGE_PORT}"
cleanup

log "Raw eval orchestrator finished successfully"
