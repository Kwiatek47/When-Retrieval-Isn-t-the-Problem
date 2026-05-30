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
REPORT_DIR="${REPORT_DIR:-reports/official_pqal500}"
RUN_LABEL="${PUBMEDQA_EVAL_LABEL:-official_pqal500_$(date -u +%Y%m%dT%H%M%SZ)}"

DATASET="${OFFICIAL_PQAL500_DATASET:-${DATA_DIR}/eval.json}"
CHUNKS="${OFFICIAL_PQAL500_CHUNKS:-${DATA_DIR}/chunks.parquet}"
BM25_STATS="${BM25_STATS_PATH:-${DATA_DIR}/bm25_stats.json}"
INDEX_MANIFEST="${OFFICIAL_PQAL500_INDEX_MANIFEST:-${DATA_DIR}/index_manifest.json}"
INDEX_CHECKPOINT="${OFFICIAL_PQAL500_INDEX_CHECKPOINT:-${DATA_DIR}/index_checkpoint.json}"
CHUNK_STORE="${OFFICIAL_PQAL500_CHUNK_STORE:-data/indexes/pubmedqa_official_pqal_test/chunk_store.sqlite}"
BASELINE="${OFFICIAL_PQAL500_BASELINE:-${DATA_DIR}/previous_best.json}"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-MedicalChunk_pubmed_reviews_v1_medcpt_20260518}"
QDRANT_VECTOR_NAME="${QDRANT_VECTOR_NAME:-medcpt_dense}"
QDRANT_SPARSE_VECTOR_NAME="${QDRANT_SPARSE_VECTOR_NAME:-bm25_sparse}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-medcpt-ncbi-v1}"

RAG_API_URL="${RAG_API_URL:-http://127.0.0.1:8000}"
MODEL="${PUBMEDQA_EVAL_MODEL:-${OLLAMA_MODEL:-qwen2.5:7b}}"
CANDIDATE_K="${PUBMEDQA_EVAL_CANDIDATE_K:-20}"
TOP_K="${PUBMEDQA_EVAL_TOP_K:-1}"
TEMPERATURE="${PUBMEDQA_EVAL_TEMPERATURE:-0.0}"
MODE="${PUBMEDQA_EVAL_MODE:-benchmark_pqal}"
METRIC_PATH="${OFFICIAL_PQAL500_GATE_METRIC:-summary.label_accuracy}"
ALLOWED_DROP="${OFFICIAL_PQAL500_ALLOWED_DROP:-0.01}"
BM25_STATS_VERSION="${BM25_STATS_VERSION:-bm25-v1}"
CORPUS_VERSION="${RAG_CORPUS_VERSION:-pubmedqa-official-pqal-test-v1}"

JSON_REPORT="${REPORT_DIR}/${RUN_LABEL}.json"
MD_REPORT="${REPORT_DIR}/${RUN_LABEL}.md"
GATE_REPORT="${REPORT_DIR}/${RUN_LABEL}.gate.json"
LOCKFILE="${REPORT_DIR}/${RUN_LABEL}.lock.json"
CANONICAL_LOCKFILE="${DATA_DIR}/eval_lock.json"
ERROR_ANALYSIS_OUT="${OFFICIAL_PQAL500_ERROR_ANALYSIS_OUT:-data/error_analysis_pqal500.json}"

export RAG_CORPUS_VERSION="${CORPUS_VERSION}"
export BM25_STATS_PATH="${BM25_STATS}"
export QDRANT_COLLECTION
export RAG_EVIDENCE_JUDGE_METHOD="${RAG_EVIDENCE_JUDGE_METHOD:-classifier}"
export RAG_EVIDENCE_CLASSIFIER_ENABLED="${RAG_EVIDENCE_CLASSIFIER_ENABLED:-true}"

mkdir -p "${REPORT_DIR}"

echo "==> Indexing official PQA-L 500 corpus into Qdrant collection ${QDRANT_COLLECTION}"
INDEX_ARGS=(
  scripts/rag/01_build_index.py
  --chunks "${CHUNKS}"
  --collection "${QDRANT_COLLECTION}"
  --qdrant-url "${QDRANT_URL}"
  --dense-vector-name "${QDRANT_VECTOR_NAME}"
  --sparse-vector-name "${QDRANT_SPARSE_VECTOR_NAME}"
  --corpus-version "${CORPUS_VERSION}"
  --bm25-stats-out "${BM25_STATS}"
  --manifest-out "${INDEX_MANIFEST}"
  --checkpoint-out "${INDEX_CHECKPOINT}"
  --chunk-store "${CHUNK_STORE}"
  --embedding-batch-size "${OFFICIAL_PQAL500_EMBEDDING_BATCH_SIZE:-32}"
  --upsert-batch-size "${OFFICIAL_PQAL500_UPSERT_BATCH_SIZE:-128}"
)
if [[ "${OFFICIAL_PQAL500_RECREATE_INDEX:-0}" == "1" ]]; then
  INDEX_ARGS+=(--recreate)
else
  INDEX_ARGS+=(--resume)
fi
"${PYTHON_BIN}" "${INDEX_ARGS[@]}"

echo "==> Running official PQA-L 500 eval against ${RAG_API_URL}"
"${PYTHON_BIN}" scripts/rag/06_evaluate_pubmedqa_benchmark.py \
  --dataset "${DATASET}" \
  --api-url "${RAG_API_URL}" \
  --model "${MODEL}" \
  --candidate-k "${CANDIDATE_K}" \
  --top-k "${TOP_K}" \
  --temperature "${TEMPERATURE}" \
  --mode "${MODE}" \
  --json-out "${JSON_REPORT}" \
  --md-out "${MD_REPORT}"

if [[ "${OFFICIAL_PQAL500_REQUIRE_CLASSIFIER:-1}" == "1" ]]; then
  echo "==> Verifying official PQA-L eval used classifier decisions"
  "${PYTHON_BIN}" scripts/eval/check_pubmedqa_evidence_methods.py \
    --report "${JSON_REPORT}" \
    --out "${REPORT_DIR}/${RUN_LABEL}.methods.json" \
    --forbid-only-method rules \
    --require-method-prefix deberta_classifier \
    --min-required-count 1
fi

echo "==> Seeding manual error-analysis file"
"${PYTHON_BIN}" scripts/eval/write_pqal500_error_analysis_seed.py \
  --report "${JSON_REPORT}" \
  --out "${ERROR_ANALYSIS_OUT}"

echo "==> Checking regression gate: ${METRIC_PATH} >= previous best - ${ALLOWED_DROP}"
GATE_STATUS=0
"${PYTHON_BIN}" scripts/eval/check_regression_gate.py \
  --report "${JSON_REPORT}" \
  --baseline "${BASELINE}" \
  --metric-path "${METRIC_PATH}" \
  --allowed-drop "${ALLOWED_DROP}" \
  --out "${GATE_REPORT}" || GATE_STATUS=$?

echo "==> Writing eval lockfile"
"${PYTHON_BIN}" scripts/eval/write_official_pqal500_lock.py \
  --out "${LOCKFILE}" \
  --dataset "${DATASET}" \
  --chunks "${CHUNKS}" \
  --bm25-stats "${BM25_STATS}" \
  --index-manifest "${INDEX_MANIFEST}" \
  --report "${JSON_REPORT}" \
  --gate-result "${GATE_REPORT}" \
  --model "${MODEL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --collection "${QDRANT_COLLECTION}" \
  --qdrant-url "${QDRANT_URL}" \
  --dense-vector-name "${QDRANT_VECTOR_NAME}" \
  --sparse-vector-name "${QDRANT_SPARSE_VECTOR_NAME}" \
  --api-url "${RAG_API_URL}" \
  --candidate-k "${CANDIDATE_K}" \
  --top-k "${TOP_K}" \
  --temperature "${TEMPERATURE}" \
  --mode "${MODE}" \
  --metric-path "${METRIC_PATH}" \
  --allowed-drop "${ALLOWED_DROP}" \
  --bm25-stats-version "${BM25_STATS_VERSION}"
cp "${LOCKFILE}" "${CANONICAL_LOCKFILE}"

echo "Done."
echo "JSON report: ${JSON_REPORT}"
echo "Markdown report: ${MD_REPORT}"
echo "Lockfile: ${LOCKFILE}"

exit "${GATE_STATUS}"
