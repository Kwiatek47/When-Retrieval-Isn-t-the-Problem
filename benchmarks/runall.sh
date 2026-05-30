#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/benchmarks/embedding_benchmark}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

export PYTHONPATH="${PROJECT_ROOT}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m embedding_benchmark.run_all \
  --qdrant-url "${QDRANT_URL}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

