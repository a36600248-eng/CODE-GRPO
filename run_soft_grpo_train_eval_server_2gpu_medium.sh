#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_soft_grpo_train_eval_server_2gpu_medium.sh <seed> [port]"
  exit 1
fi

SEED=$1
PORT=${2:-8022}
ORCH_LOG=/root/autodl-tmp/CODE-GRPO/trl/soft_grpo_train_eval_medium_orchestrator_${PORT}.log

log() {
  echo "[$(date '+%F %T')] $*"
}

exec > >(tee -a "${ORCH_LOG}") 2>&1

log "Soft-reward GRPO medium train+eval orchestrator starting: seed=${SEED} base_port=${PORT}"

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
TRAIN_CONFIG=configs/comparison/server_2gpu/codegrpo_single_round_zero_pass_soft_reward_mbpp_medium.yaml
EVAL_CONFIG=configs/comparison/server_2gpu/raw_qwen7b_eval_mbpp_full.yaml
TRAIN_OUTPUT_DIR=/root/autodl-tmp/CODE-GRPO/trl/runs_single_round_soft_grpo_medium_seed${SEED}
EVAL_OUTPUT_DIR=/root/autodl-tmp/CODE-GRPO/trl/runs_single_round_soft_grpo_medium_eval_seed${SEED}

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
    log "Stopping soft-reward GRPO medium vLLM server process group: ${SERVER_PID}"
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

resolve_latest_train_out() {
  local train_root="${1}"
  local latest
  latest=$(find "${train_root}/train" -mindepth 2 -maxdepth 2 -type d -name train_out | sort | tail -n 1)
  if [ -z "${latest}" ]; then
    log "Could not locate train_out under ${train_root}/train"
    exit 1
  fi
  echo "${latest}"
}

start_server() {
  local port="$1"
  local server_log="/root/autodl-tmp/CODE-GRPO/trl/vllm_server_soft_grpo_medium_${port}.log"

  log "Starting soft-reward GRPO medium vLLM server on GPU0, port=${port}"
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
  local port="$1"
  local group_port=$((port + 40000))
  log "Running soft-reward GRPO medium train"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
    --config "${TRAIN_CONFIG}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --vllm_server_base_url "http://127.0.0.1:${port}" \
    --vllm_group_port "${group_port}"
}

run_eval() {
  local port="$1"
  local group_port=$((port + 40000))
  local trained_adapter_dir
  trained_adapter_dir=$(resolve_latest_train_out "${TRAIN_OUTPUT_DIR}")
  log "Running soft-reward GRPO medium eval on trained adapter"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval \
    --config "${EVAL_CONFIG}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --model_name_or_path "${trained_adapter_dir}" \
    --use_peft true \
    --vllm_server_base_url "http://127.0.0.1:${port}" \
    --vllm_group_port "${group_port}"
  log "Soft-reward GRPO medium eval finished. Train adapter: ${trained_adapter_dir} Eval dir: ${EVAL_OUTPUT_DIR}"
}

STAGE_PORT=$(choose_stage_port "$PORT")
start_server "${STAGE_PORT}"
run_train "${STAGE_PORT}"
run_eval "${STAGE_PORT}"
cleanup

log "Soft-reward GRPO medium train+eval orchestrator finished successfully"
