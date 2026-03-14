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
    echo "Stopping vLLM server process group: ${SERVER_PID}"
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
      echo "Port ${candidate} already has a live server, trying next port" >&2
      candidate=$((candidate + 1))
      continue
    fi
    if port_in_use "$candidate"; then
      echo "Port ${candidate} is already in use, trying next port" >&2
      candidate=$((candidate + 1))
      continue
    fi
    if port_in_use "$candidate_group_port"; then
      echo "Group port ${candidate_group_port} is already in use, trying next port" >&2
      candidate=$((candidate + 1))
      continue
    fi
    echo "$candidate"
    return
  done
}

start_server() {
  local port="$1"
  local group_port="$((port + 40000))"
  local server_log="/root/autodl-tmp/CODE-GRPO/trl/vllm_server_${port}.log"

  echo "Checking whether port ${port} already has a live server (group_port=${group_port})"
  if healthcheck "$port"; then
    echo "Port ${port} already has a live server. Stop it or use another port."
    exit 1
  fi
  if port_in_use "$port"; then
    echo "Port ${port} is already in use by another process. Stop it or use another port."
    exit 1
  fi
  if port_in_use "$group_port"; then
    echo "Group port ${group_port} is already in use by another process. Stop it or use another port."
    exit 1
  fi

  echo "Starting official TRL vLLM server sync path on GPU0, port=${port}"
  setsid env CUDA_VISIBLE_DEVICES=0 python -m trl.cli.main vllm-serve \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tensor_parallel_size 1 \
    --data_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    > "${server_log}" 2>&1 &
  SERVER_PID=$!

  echo "Waiting for vLLM server health endpoint on port ${port}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM server process exited before becoming healthy. Tail of ${server_log}:"
      tail -n 100 "${server_log}" || true
      exit 1
    fi
    if healthcheck "$port"; then
      break
    fi
    sleep 2
  done

  if ! healthcheck "$port"; then
    echo "vLLM server failed to start. Tail of ${server_log}:"
    tail -n 100 "${server_log}" || true
    exit 1
  fi
}

run_train() {
  local cfg="$1"
  local port="$2"
  local group_port="$((port + 40000))"
  echo "Running train config: ${cfg} (port=${port}, group_port=${group_port})"
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
  echo "Running eval config: ${cfg} (port=${port}, group_port=${group_port})"
  CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval \
    --config "${cfg}" \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --vllm_server_base_url "http://127.0.0.1:${port}" \
    --vllm_group_port "${group_port}"
}

CONFIG_ROOT=configs/comparison/server_2gpu

STAGE_PORT=$(choose_stage_port "$PORT")
start_server "${STAGE_PORT}"
run_eval ${CONFIG_ROOT}/raw_qwen7b_eval_mbpp.yaml "${STAGE_PORT}"
cleanup

STAGE_PORT=$(choose_stage_port "$((STAGE_PORT + 1))")
start_server "${STAGE_PORT}"
run_train ${CONFIG_ROOT}/codegrpo_method_mbpp.yaml "${STAGE_PORT}"
cleanup

STAGE_PORT=$(choose_stage_port "$((STAGE_PORT + 1))")
start_server "${STAGE_PORT}"
run_train ${CONFIG_ROOT}/vanilla_grpo_multiround_k2_mbpp.yaml "${STAGE_PORT}"
cleanup

STAGE_PORT=$(choose_stage_port "$((STAGE_PORT + 1))")
start_server "${STAGE_PORT}"
run_train ${CONFIG_ROOT}/vanilla_grpo_single_round_k8_mbpp.yaml "${STAGE_PORT}"
cleanup
