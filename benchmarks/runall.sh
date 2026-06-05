#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/benchmarks/embedding_benchmark}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
RUNALL_CUDA_VISIBLE_DEVICES="${RUNALL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
RUNALL_DEVICE_0="${RUNALL_DEVICE_0:-cuda:0}"
RUNALL_DEVICE_1="${RUNALL_DEVICE_1:-cuda:1}"
RUNALL_EVAL_DEVICE="${RUNALL_EVAL_DEVICE:-cuda:0}"
RUNALL_CROSS_ENCODER_DEVICE="${RUNALL_CROSS_ENCODER_DEVICE:-${RUNALL_EVAL_DEVICE}}"
RUNALL_AUTOTUNE_SAMPLE_SIZE="${RUNALL_AUTOTUNE_SAMPLE_SIZE:-2048}"
RUNALL_AUTOTUNE_MAX_BATCH="${RUNALL_AUTOTUNE_MAX_BATCH:-2048}"
RUNALL_AUTOTUNE_SAFETY_FACTOR="${RUNALL_AUTOTUNE_SAFETY_FACTOR:-0.75}"
RUNALL_UPSERT_BATCH_SIZE="${RUNALL_UPSERT_BATCH_SIZE:-128}"
RUNALL_LOG_DIR="${RUNALL_LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUNALL_LOG_FILE="${RUNALL_LOG_FILE:-${RUNALL_LOG_DIR}/runall_$(date -u +%Y%m%dT%H%M%SZ).log}"
RUNALL_CLEANUP_MODEL_DATA="${RUNALL_CLEANUP_MODEL_DATA:-1}"
RUNALL_RESUME="${RUNALL_RESUME:-1}"
RUNALL_STOP_ON_ERROR="${RUNALL_STOP_ON_ERROR:-1}"
AUTO_START_SERVICES="${AUTO_START_SERVICES:-1}"
AUTO_START_QDRANT="${AUTO_START_QDRANT:-${AUTO_START_SERVICES}}"
AUTO_START_OLLAMA="${AUTO_START_OLLAMA:-${AUTO_START_SERVICES}}"
AUTO_PULL_OLLAMA_MODEL="${AUTO_PULL_OLLAMA_MODEL:-1}"
QDRANT_START_MODE="${QDRANT_START_MODE:-local}"
QDRANT_ALLOW_DOCKER="${QDRANT_ALLOW_DOCKER:-0}"
QDRANT_BIN="${QDRANT_BIN:-}"
QDRANT_STORAGE_DIR="${QDRANT_STORAGE_DIR:-${OUTPUT_ROOT}/qdrant_storage}"

mkdir -p "${RUNALL_LOG_DIR}"
exec > >(tee -a "${RUNALL_LOG_FILE}") 2>&1
echo "Writing runall log: ${RUNALL_LOG_FILE}"

export PYTHONPATH="${PROJECT_ROOT}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${RUNALL_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${RUNALL_CUDA_VISIBLE_DEVICES}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
export OLLAMA_BASE_URL
export OLLAMA_MODEL

cd "${PROJECT_ROOT}"

url_ok() {
  local url="$1"
  "${PYTHON_BIN}" - "$url" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=5) as response:
        raise SystemExit(0 if 200 <= response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local timeout_seconds="$3"
  local started_at
  started_at="$(date +%s)"
  while true; do
    if url_ok "${url}"; then
      echo "${name} is ready: ${url}"
      return 0
    fi
    if (( "$(date +%s)" - started_at >= timeout_seconds )); then
      echo "${name} did not become ready within ${timeout_seconds}s: ${url}" >&2
      return 1
    fi
    sleep 2
  done
}

has_arg() {
  local expected="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${expected}" ]]; then
      return 0
    fi
  done
  return 1
}

append_default_arg() {
  local flag="$1"
  local value="$2"
  if ! has_arg "${flag}" "${RUNALL_ARGS[@]}"; then
    RUNALL_ARGS+=("${flag}" "${value}")
  fi
}

append_default_arg_list() {
  local flag="$1"
  shift
  if ! has_arg "${flag}" "${RUNALL_ARGS[@]}"; then
    RUNALL_ARGS+=("${flag}" "$@")
  fi
}

append_default_flag() {
  local flag="$1"
  if ! has_arg "${flag}" "${RUNALL_ARGS[@]}"; then
    RUNALL_ARGS+=("${flag}")
  fi
}

truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_qdrant_bin() {
  if [[ -n "${QDRANT_BIN}" && -x "${QDRANT_BIN}" ]]; then
    echo "${QDRANT_BIN}"
    return 0
  fi
  if command -v qdrant >/dev/null 2>&1; then
    command -v qdrant
    return 0
  fi
  if [[ -x "${HOME}/bin/qdrant" ]]; then
    echo "${HOME}/bin/qdrant"
    return 0
  fi
  return 1
}

start_qdrant_local() {
  local bin
  if ! bin="$(resolve_qdrant_bin)"; then
    return 1
  fi
  mkdir -p "${QDRANT_STORAGE_DIR}" "${OUTPUT_ROOT}/logs"
  echo "Starting Qdrant locally: ${bin} --storage-dir ${QDRANT_STORAGE_DIR}"
  nohup "${bin}" --storage-dir "${QDRANT_STORAGE_DIR}" >"${OUTPUT_ROOT}/logs/qdrant_local.log" 2>&1 &
  echo "$!" >"${OUTPUT_ROOT}/logs/qdrant_local.pid"
  wait_for_url "Qdrant" "${QDRANT_URL}/readyz" 120
}

ensure_qdrant() {
  if url_ok "${QDRANT_URL}/readyz"; then
    echo "Qdrant already running: ${QDRANT_URL}"
    return
  fi
  case "${AUTO_START_QDRANT}" in
    1|true|TRUE|yes|YES|on|ON)
      case "${QDRANT_START_MODE}" in
        local|LOCAL)
          if start_qdrant_local; then
            return
          fi
          echo "Qdrant binary not found. Put it at ~/bin/qdrant, set QDRANT_BIN, or set QDRANT_ALLOW_DOCKER=1 for Docker fallback." >&2
          exit 1
          ;;
        auto|AUTO)
          if start_qdrant_local; then
            return
          fi
          ;;
        docker|DOCKER)
          ;;
        *)
          echo "Invalid QDRANT_START_MODE=${QDRANT_START_MODE}. Use local, auto, or docker." >&2
          exit 2
          ;;
      esac
      if ! truthy "${QDRANT_ALLOW_DOCKER}" && [[ "${QDRANT_START_MODE}" != "docker" && "${QDRANT_START_MODE}" != "DOCKER" ]]; then
        echo "Qdrant local startup failed and Docker fallback is disabled. Set QDRANT_ALLOW_DOCKER=1 to allow docker compose." >&2
        exit 1
      fi
      if ! command -v docker >/dev/null 2>&1; then
        echo "Qdrant is not running and docker is unavailable. Start Qdrant at ${QDRANT_URL} or install a local qdrant binary." >&2
        exit 1
      fi
      echo "Starting Qdrant with docker compose..."
      docker compose up -d qdrant
      wait_for_url "Qdrant" "${QDRANT_URL}/readyz" 120
      ;;
    *)
      echo "Qdrant is not running at ${QDRANT_URL}. Start it or set AUTO_START_QDRANT=1." >&2
      exit 1
      ;;
  esac
}

ensure_ollama() {
  if url_ok "${OLLAMA_BASE_URL}/api/tags"; then
    echo "Ollama already running: ${OLLAMA_BASE_URL}"
  else
    case "${AUTO_START_OLLAMA}" in
      1|true|TRUE|yes|YES|on|ON)
        if ! command -v ollama >/dev/null 2>&1; then
          echo "Ollama is not running and the ollama command is unavailable. Install/start Ollama at ${OLLAMA_BASE_URL} or set AUTO_START_OLLAMA=0 and --skip-pubmedqa-pipeline-eval." >&2
          exit 1
        fi
        mkdir -p "${OUTPUT_ROOT}/logs"
        echo "Starting Ollama in background..."
        nohup ollama serve >"${OUTPUT_ROOT}/logs/ollama_serve.log" 2>&1 &
        echo "$!" >"${OUTPUT_ROOT}/logs/ollama_serve.pid"
        wait_for_url "Ollama" "${OLLAMA_BASE_URL}/api/tags" 120
        ;;
      *)
        echo "Ollama is not running at ${OLLAMA_BASE_URL}. Start it or set AUTO_START_OLLAMA=1." >&2
        exit 1
        ;;
    esac
  fi

  case "${AUTO_PULL_OLLAMA_MODEL}" in
    1|true|TRUE|yes|YES|on|ON)
      if command -v ollama >/dev/null 2>&1; then
        echo "Ensuring Ollama model is available: ${OLLAMA_MODEL}"
        ollama pull "${OLLAMA_MODEL}"
      fi
      ;;
  esac
}

ensure_qdrant
if has_arg "--skip-pubmedqa-pipeline-eval" "$@"; then
  echo "Skipping Ollama startup because --skip-pubmedqa-pipeline-eval was passed."
else
  ensure_ollama
fi

RUNALL_ARGS=("$@")
append_default_arg "--device-0" "${RUNALL_DEVICE_0}"
append_default_arg "--device-1" "${RUNALL_DEVICE_1}"
append_default_arg "--eval-device" "${RUNALL_EVAL_DEVICE}"
append_default_arg "--pubmedqa-cross-encoder-device" "${RUNALL_CROSS_ENCODER_DEVICE}"
append_default_arg "--autotune-sample-size" "${RUNALL_AUTOTUNE_SAMPLE_SIZE}"
append_default_arg "--autotune-max-batch" "${RUNALL_AUTOTUNE_MAX_BATCH}"
append_default_arg "--autotune-safety-factor" "${RUNALL_AUTOTUNE_SAFETY_FACTOR}"
append_default_arg "--upsert-batch-size" "${RUNALL_UPSERT_BATCH_SIZE}"

case "${RUNALL_RESUME}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "Resume enabled: completed models and per-stage checkpoints will be reused."
    ;;
  0|false|FALSE|no|NO|off|OFF)
    append_default_flag "--no-resume"
    append_default_flag "--no-resume-completed-models"
    echo "Resume disabled: RUNALL_RESUME=${RUNALL_RESUME}."
    ;;
  *)
    echo "Invalid RUNALL_RESUME=${RUNALL_RESUME}. Use 1/0, true/false, yes/no, or on/off." >&2
    exit 2
    ;;
esac

case "${RUNALL_STOP_ON_ERROR}" in
  1|true|TRUE|yes|YES|on|ON)
    append_default_flag "--stop-on-error"
    echo "Stop-on-error enabled: rerun the same command to continue from checkpoint."
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "Stop-on-error disabled: failed models will be logged and later models may continue."
    ;;
  *)
    echo "Invalid RUNALL_STOP_ON_ERROR=${RUNALL_STOP_ON_ERROR}. Use 1/0, true/false, yes/no, or on/off." >&2
    exit 2
    ;;
esac

case "${RUNALL_CLEANUP_MODEL_DATA}" in
  1|true|TRUE|yes|YES|on|ON)
    if ! has_arg "--cleanup-model-data" "$@" && ! has_arg "--no-cleanup-model-data" "$@"; then
      RUNALL_ARGS+=("--cleanup-model-data")
    fi
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Invalid RUNALL_CLEANUP_MODEL_DATA=${RUNALL_CLEANUP_MODEL_DATA}. Use 1/0, true/false, yes/no, or on/off." >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" -m embedding_benchmark.run_all \
  --qdrant-url "${QDRANT_URL}" \
  --output-root "${OUTPUT_ROOT}" \
  "${RUNALL_ARGS[@]}"
