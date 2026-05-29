#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

DATA_DIR="data/benchmarks/pubmedqa/official_pqal_test"
QUICK_DIR="${QUICK_PQAL_DATA_DIR:-${DATA_DIR}/quick}"
REPORT_DIR="${REPORT_DIR:-reports/pqal_quick}"
RUN_LABEL="${QUICK_PQAL_LABEL:-quick_pqal_$(date -u +%Y%m%dT%H%M%SZ)}"

RAG_API_URL="${RAG_API_URL:-http://127.0.0.1:8000}"
MODEL="${PUBMEDQA_EVAL_MODEL:-${OLLAMA_MODEL:-qwen2.5:7b}}"
CANDIDATE_K="${PUBMEDQA_EVAL_CANDIDATE_K:-20}"
TOP_K="${PUBMEDQA_EVAL_TOP_K:-1}"
TEMPERATURE="${PUBMEDQA_EVAL_TEMPERATURE:-0.0}"
MODE="${PUBMEDQA_EVAL_MODE:-benchmark_pqal}"
PER_LABEL="${QUICK_PQAL_PER_LABEL:-30}"
YES_COUNT="${QUICK_PQAL_YES_COUNT:-100}"

BALANCED_COUNT=$((PER_LABEL * 3))
BALANCED_DATASET="${QUICK_DIR}/balanced${BALANCED_COUNT}.json"
YES_DATASET="${QUICK_DIR}/first${YES_COUNT}_yes.json"

BALANCED_JSON="${REPORT_DIR}/${RUN_LABEL}_balanced${BALANCED_COUNT}.json"
BALANCED_MD="${REPORT_DIR}/${RUN_LABEL}_balanced${BALANCED_COUNT}.md"
YES_JSON="${REPORT_DIR}/${RUN_LABEL}_first${YES_COUNT}_yes.json"
YES_MD="${REPORT_DIR}/${RUN_LABEL}_first${YES_COUNT}_yes.md"
SUMMARY_JSON="${REPORT_DIR}/${RUN_LABEL}.summary.json"
SUMMARY_MD="${REPORT_DIR}/${RUN_LABEL}.summary.md"

mkdir -p "${REPORT_DIR}"

echo "==> Building quick PQA-L datasets"
"${PYTHON_BIN}" scripts/eval/build_pqal_quick_sets.py \
  --dataset "${DATA_DIR}/eval.json" \
  --out-dir "${QUICK_DIR}" \
  --per-label "${PER_LABEL}" \
  --yes-count "${YES_COUNT}"

echo "==> Running balanced PQA-L ${BALANCED_COUNT} eval in mode=${MODE}"
"${PYTHON_BIN}" scripts/rag/06_evaluate_pubmedqa_benchmark.py \
  --dataset "${BALANCED_DATASET}" \
  --api-url "${RAG_API_URL}" \
  --model "${MODEL}" \
  --candidate-k "${CANDIDATE_K}" \
  --top-k "${TOP_K}" \
  --temperature "${TEMPERATURE}" \
  --mode "${MODE}" \
  --json-out "${BALANCED_JSON}" \
  --md-out "${BALANCED_MD}"

echo "==> Running first-${YES_COUNT}-yes PQA-L eval in mode=${MODE}"
"${PYTHON_BIN}" scripts/rag/06_evaluate_pubmedqa_benchmark.py \
  --dataset "${YES_DATASET}" \
  --api-url "${RAG_API_URL}" \
  --model "${MODEL}" \
  --candidate-k "${CANDIDATE_K}" \
  --top-k "${TOP_K}" \
  --temperature "${TEMPERATURE}" \
  --mode "${MODE}" \
  --json-out "${YES_JSON}" \
  --md-out "${YES_MD}"

echo "==> Writing quick PQA-L summary"
"${PYTHON_BIN}" scripts/eval/write_quick_pqal_summary.py \
  --balanced-report "${BALANCED_JSON}" \
  --yes-report "${YES_JSON}" \
  --json-out "${SUMMARY_JSON}" \
  --md-out "${SUMMARY_MD}"

echo "Done."
echo "Balanced report: ${BALANCED_JSON}"
echo "First-${YES_COUNT}-yes report: ${YES_JSON}"
echo "Summary: ${SUMMARY_JSON}"
