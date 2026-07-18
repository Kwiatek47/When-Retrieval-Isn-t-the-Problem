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

LABEL="${CLASSIFIER_EVAL_LABEL:-classifier_balanced90}"
MODEL_DIR="${CLASSIFIER_MODEL_DIR:-artifacts/classifier/pubmedqa_deberta/best}"
DATASET="${CLASSIFIER_EVAL_DATASET:-data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json}"
CORPUS="${CLASSIFIER_EVAL_CORPUS:-data/benchmarks/pubmedqa/official_pqal_test/corpus.json}"
REPORT_DIR="${CLASSIFIER_REPORT_DIR:-reports/classifier}"
JSON_OUT="${REPORT_DIR}/${LABEL}.json"
MD_OUT="${REPORT_DIR}/${LABEL}.md"

mkdir -p "${REPORT_DIR}"

echo "==> Classifier eval on balanced90"
echo "Model: ${MODEL_DIR}"
echo "Dataset: ${DATASET}"

"${PYTHON_BIN}" scripts/classifier/evaluate_pubmedqa_classifier.py \
  --dataset "${DATASET}" \
  --corpus "${CORPUS}" \
  --model-dir "${MODEL_DIR}" \
  --temperature-path "${MODEL_DIR}/calibration.json" \
  --json-out "${JSON_OUT}" \
  --md-out "${MD_OUT}"

echo
echo "==> Summary"
"${PYTHON_BIN}" - <<'PY' "${JSON_OUT}"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
classifier = report.get("classifier") or {}
metrics = classifier.get("metrics") or {}
per_label = metrics.get("per_label") or {}
print(f"accuracy={metrics.get('accuracy', 0):.3f} macro_f1={metrics.get('macro_f1', 0):.3f}")
for label in ("yes", "no", "maybe"):
    row = per_label.get(label) or {}
    print(
        f"  {label}: recall={row.get('recall', 0):.3f} "
        f"precision={row.get('precision', 0):.3f} f1={row.get('f1', 0):.3f}"
    )
predicted = classifier.get("predicted_labels") or {}
if predicted:
    print("predicted_labels:", predicted)
PY

echo
echo "Wrote:"
echo "  ${JSON_OUT}"
echo "  ${MD_OUT}"

if [[ "${RUN_RAG_BALANCED90:-0}" == "1" ]]; then
  echo
  echo "==> End-to-end RAG eval on balanced90"
  PUBMEDQA_EVAL_LABEL="${LABEL}_rag" \
  RAG_EVIDENCE_JUDGE_METHOD="${RAG_EVIDENCE_JUDGE_METHOD:-classifier}" \
  RAG_EVIDENCE_CLASSIFIER_ENABLED="${RAG_EVIDENCE_CLASSIFIER_ENABLED:-true}" \
  "${PYTHON_BIN}" scripts/rag/06_evaluate_pubmedqa_benchmark.py \
    --dataset "${DATASET}" \
    --mode benchmark_pqal \
    --model "${PUBMEDQA_EVAL_MODEL:-qwen2.5:7b}" \
    --json-out "reports/${LABEL}_rag.json" \
    --md-out "reports/${LABEL}_rag.md"
fi
