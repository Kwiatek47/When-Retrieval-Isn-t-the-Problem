#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/benchmarks/embedding_benchmark}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
AUTO_START_SERVICES="${AUTO_START_SERVICES:-1}"
AUTO_START_QDRANT="${AUTO_START_QDRANT:-${AUTO_START_SERVICES}}"
AUTO_START_OLLAMA="${AUTO_START_OLLAMA:-${AUTO_START_SERVICES}}"
AUTO_PULL_OLLAMA_MODEL="${AUTO_PULL_OLLAMA_MODEL:-1}"
RUN_CLASSIFIER_FIRST="${RUN_CLASSIFIER_FIRST:-1}"
CLASSIFIER_SCRIPT="${CLASSIFIER_SCRIPT:-${PROJECT_ROOT}/scripts/classifier/run_pubmedqa_2x4080_full_experiments.sh}"
CLASSIFIER_PYTHON_BIN="${CLASSIFIER_PYTHON_BIN:-}"

if [[ -z "${CLASSIFIER_PYTHON_BIN}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    CLASSIFIER_PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    CLASSIFIER_PYTHON_BIN="${PYTHON_BIN}"
  fi
fi

export PYTHONPATH="${PROJECT_ROOT}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
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

ensure_qdrant() {
  if url_ok "${QDRANT_URL}/readyz"; then
    echo "Qdrant already running: ${QDRANT_URL}"
    return
  fi
  case "${AUTO_START_QDRANT}" in
    1|true|TRUE|yes|YES|on|ON)
      if ! command -v docker >/dev/null 2>&1; then
        echo "Qdrant is not running and docker is unavailable. Start Qdrant at ${QDRANT_URL} or set AUTO_START_QDRANT=0." >&2
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

case "${RUN_CLASSIFIER_FIRST}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ ! -f "${CLASSIFIER_SCRIPT}" ]]; then
      echo "Classifier script not found: ${CLASSIFIER_SCRIPT}" >&2
      exit 1
    fi
    echo "Starting PubMedQA classifier pre-run: ${CLASSIFIER_SCRIPT}"
    PYTHON_BIN="${CLASSIFIER_PYTHON_BIN}" bash "${CLASSIFIER_SCRIPT}"
    echo "PubMedQA classifier pre-run complete. Starting embedding benchmark."
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "Skipping PubMedQA classifier pre-run."
    ;;
  *)
    echo "Invalid RUN_CLASSIFIER_FIRST=${RUN_CLASSIFIER_FIRST}. Use 1/0, true/false, yes/no, or on/off." >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" -m embedding_benchmark.run_all \
  --qdrant-url "${QDRANT_URL}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
