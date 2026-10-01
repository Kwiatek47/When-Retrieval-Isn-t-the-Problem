#!/usr/bin/env bash
# Start multiple Ollama servers, each pinned to a different GPU.
# Usage:
#   scripts/agents/start_multi_ollama.sh
#   scripts/agents/start_multi_ollama.sh 1,2,3 11434
#
# Args:
#   $1  comma-separated nvidia-smi / Vulkan device IDs (default: 1,2,3)
#   $2  first host port (default: 11434); each next GPU gets +1
#
# This Ollama build uses Vulkan by default. CUDA_VISIBLE_DEVICES alone does
# NOT isolate Vulkan devices (all instances pile onto one GPU). We pin with
# GGML_VK_VISIBLE_DEVICES and leave Vulkan enabled.
#
# Then run:
#   --ollama-base-urls http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436

set -euo pipefail

GPU_LIST="${1:-1}"
BASE_PORT="${2:-11434}"
NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
MAX_LOADED="${OLLAMA_MAX_LOADED_MODELS:-2}"
LOG_DIR="${OLLAMA_MULTI_LOG_DIR:-/tmp/ollama_multi}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
PORT="${BASE_PORT}"

echo "Starting ${#GPUS[@]} Ollama instance(s); OLLAMA_NUM_PARALLEL=${NUM_PARALLEL}"
echo "Pinning via GGML_VK_VISIBLE_DEVICES (Vulkan); do NOT set CUDA_VISIBLE_DEVICES"
for gpu in "${GPUS[@]}"; do
  gpu_trimmed="$(echo "${gpu}" | xargs)"
  log="${LOG_DIR}/ollama_gpu${gpu_trimmed}_p${PORT}.log"
  echo "GPU ${gpu_trimmed} -> 127.0.0.1:${PORT} (log: ${log})"
  # Unset CUDA_VISIBLE_DEVICES so Vulkan discovery is not confused.
  env -u CUDA_VISIBLE_DEVICES \
  GGML_VK_VISIBLE_DEVICES="${gpu_trimmed}" \
  OLLAMA_VULKAN=true \
  OLLAMA_HOST="127.0.0.1:${PORT}" \
  OLLAMA_NUM_PARALLEL="${NUM_PARALLEL}" \
  OLLAMA_MAX_LOADED_MODELS="${MAX_LOADED}" \
    nohup ollama serve >"${log}" 2>&1 &
  echo $! > "${LOG_DIR}/ollama_gpu${gpu_trimmed}_p${PORT}.pid"
  PORT=$((PORT + 1))
done

echo "Waiting for health..."
sleep 3
PORT="${BASE_PORT}"
for gpu in "${GPUS[@]}"; do
  if curl -sf "http://127.0.0.1:${PORT}/api/tags" >/dev/null; then
    echo "OK  http://127.0.0.1:${PORT}"
  else
    echo "WARN http://127.0.0.1:${PORT} not ready yet (check ${LOG_DIR})"
  fi
  PORT=$((PORT + 1))
done

echo
echo "Each log should show ONE discrete GPU under 'inference compute', e.g.:"
echo "  grep 'inference compute' ${LOG_DIR}/ollama_gpu*_p*.log"
echo "Example:"
echo "  --ollama-base-urls http://127.0.0.1:${BASE_PORT},http://127.0.0.1:$((BASE_PORT+1)),http://127.0.0.1:$((BASE_PORT+2))"
echo "Stop later with:  kill \$(cat ${LOG_DIR}/ollama_*.pid)"
