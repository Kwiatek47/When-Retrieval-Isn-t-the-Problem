#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-/content/drive/MyDrive/pubmedqa_diagnostic_runs/diagnostic_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${RUN_ROOT}/logs"
REPORT_ROOT="${RUN_ROOT}/reports"
DATA_DIR="data/benchmarks/pubmedqa/official_pqal_test"
QUICK_DIR="${DATA_DIR}/quick"

DIAGNOSTIC_SCOPE="${DIAGNOSTIC_SCOPE:-quick}" # quick | official
QUICK_PQAL_PER_LABEL="${QUICK_PQAL_PER_LABEL:-30}"
QUICK_PQAL_YES_COUNT="${QUICK_PQAL_YES_COUNT:-100}"
BALANCED_COUNT=$((QUICK_PQAL_PER_LABEL * 3))

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-MedicalChunk_pubmed_reviews_v1_medcpt_20260518}"
QDRANT_VECTOR_NAME="${QDRANT_VECTOR_NAME:-medcpt_dense}"
QDRANT_SPARSE_VECTOR_NAME="${QDRANT_SPARSE_VECTOR_NAME:-bm25_sparse}"
EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL:-http://127.0.0.1:8081}"
RAG_API_URL="${RAG_API_URL:-http://127.0.0.1:8000}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
if [[ -d /content ]]; then
  QDRANT_STORAGE_PATH="${QDRANT_STORAGE_PATH:-/content/qdrant_pubmedqa_diag_storage}"
else
  QDRANT_STORAGE_PATH="${QDRANT_STORAGE_PATH:-${RUN_ROOT}/qdrant_storage}"
fi

DIAGNOSTIC_LLM_MODELS="${DIAGNOSTIC_LLM_MODELS:-qwen2.5:7b}"
DIAGNOSTIC_RAG_MODELS="${DIAGNOSTIC_RAG_MODELS:-${DIAGNOSTIC_LLM_MODELS}}"
DIAGNOSTIC_TOP_KS="${DIAGNOSTIC_TOP_KS:-1 3 5}"
DIAGNOSTIC_PROMPT_STYLES="${DIAGNOSTIC_PROMPT_STYLES:-compact definitions cite_then_answer sufficiency_first}"
DIAGNOSTIC_CLASSIFIER_MODEL_PATH="${DIAGNOSTIC_CLASSIFIER_MODEL_PATH:-${RAG_EVIDENCE_CLASSIFIER_MODEL_PATH:-}}"

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
INSTALL_PY_DEPS="${INSTALL_PY_DEPS:-1}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"
INSTALL_QDRANT="${INSTALL_QDRANT:-1}"
OLLAMA_PULL_MODELS="${OLLAMA_PULL_MODELS:-1}"
START_QDRANT="${START_QDRANT:-1}"
START_OLLAMA="${START_OLLAMA:-1}"
START_EMBEDDING_SERVICE="${START_EMBEDDING_SERVICE:-1}"
RECREATE_INDEX="${RECREATE_INDEX:-1}"

EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
CROSS_ENCODER_DEVICE="${CROSS_ENCODER_DEVICE:-cuda}"
CROSS_ENCODER_MODEL_ON="${CROSS_ENCODER_MODEL_ON:-ncbi/MedCPT-Cross-Encoder}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-600}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-4096}"
OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-120}"
PUBMEDQA_EVAL_CANDIDATE_K="${PUBMEDQA_EVAL_CANDIDATE_K:-20}"
PUBMEDQA_EVAL_TEMPERATURE="${PUBMEDQA_EVAL_TEMPERATURE:-0.0}"

mkdir -p "${LOG_DIR}" "${REPORT_ROOT}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

wait_http() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-240}"
  local start
  start="$(date +%s)"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( "$(date +%s)" - start > timeout_s )); then
      echo "Timed out waiting for ${name} at ${url}" >&2
      return 1
    fi
    sleep 2
  done
}

sanitize_label() {
  echo "$1" | tr '/:@. ' '_____' | tr -cd 'A-Za-z0-9_-'
}

install_system_deps() {
  if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
    return
  fi
  if ! command -v zstd >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    log "Installing system dependency: zstd"
    apt-get update -qq
    apt-get install -y -qq zstd
  fi
}

install_python_deps() {
  if [[ "${INSTALL_PY_DEPS}" != "1" ]]; then
    return
  fi
  log "Installing Python dependencies"
  "${PYTHON_BIN}" -m pip install -q --upgrade pip
  "${PYTHON_BIN}" -m pip install -q -r requirements.txt -r services/embedding-service/requirements.txt
}

ensure_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    if [[ "${INSTALL_OLLAMA}" != "1" ]]; then
      echo "Ollama is not installed. Set INSTALL_OLLAMA=1 or install it manually." >&2
      return 1
    fi
    log "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  if [[ "${START_OLLAMA}" == "1" ]] && ! curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
    log "Starting Ollama"
    nohup ollama serve >"${LOG_DIR}/ollama.log" 2>&1 &
  fi
  wait_http "${OLLAMA_BASE_URL}/api/tags" "ollama" 240
}

ensure_qdrant_binary() {
  if command -v qdrant >/dev/null 2>&1; then
    return
  fi
  if [[ "${INSTALL_QDRANT}" != "1" ]]; then
    echo "qdrant binary not found. Set INSTALL_QDRANT=1, install qdrant, or start Docker Qdrant manually." >&2
    return 1
  fi
  local install_dir="${RUN_ROOT}/bin"
  local archive="${RUN_ROOT}/qdrant.tar.gz"
  local url="${QDRANT_ARCHIVE_URL:-https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz}"
  mkdir -p "${install_dir}"
  log "Downloading Qdrant binary"
  curl -fL "${url}" -o "${archive}"
  tar -xzf "${archive}" -C "${install_dir}"
  chmod +x "${install_dir}/qdrant"
  export PATH="${install_dir}:${PATH}"
}

ensure_qdrant() {
  if curl -fsS "${QDRANT_URL}" >/dev/null 2>&1; then
    log "Qdrant already running"
    return
  fi
  if [[ "${START_QDRANT}" != "1" ]]; then
    echo "Qdrant is not running at ${QDRANT_URL} and START_QDRANT=0." >&2
    return 1
  fi

  ensure_qdrant_binary
  log "Starting Qdrant binary"
  QDRANT__SERVICE__HTTP_PORT=6333 \
  QDRANT__SERVICE__GRPC_PORT=6334 \
  QDRANT__STORAGE__STORAGE_PATH="${QDRANT_STORAGE_PATH}" \
  nohup qdrant >"${LOG_DIR}/qdrant.log" 2>&1 &
  wait_http "${QDRANT_URL}" "qdrant" 240
}

start_embedding_service() {
  if curl -fsS "${EMBEDDING_SERVICE_URL}/health" >/dev/null 2>&1; then
    log "Embedding service already running"
    return
  fi
  if [[ "${START_EMBEDDING_SERVICE}" != "1" ]]; then
    echo "Embedding service is not running and START_EMBEDDING_SERVICE=0." >&2
    return 1
  fi
  log "Starting embedding service on ${EMBEDDING_SERVICE_URL}"
  (
    cd services/embedding-service
    PYTHONUNBUFFERED=1 \
    EMBEDDING_DEVICE="${EMBEDDING_DEVICE}" \
    EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}" \
    QDRANT_HOST="127.0.0.1" \
    QDRANT_PORT="6333" \
    QDRANT_COLLECTION="${QDRANT_COLLECTION}" \
    QDRANT_VECTOR_NAME="${QDRANT_VECTOR_NAME}" \
    QDRANT_SPARSE_VECTOR_NAME="${QDRANT_SPARSE_VECTOR_NAME}" \
    BM25_STATS_PATH="${PROJECT_ROOT}/${DATA_DIR}/bm25_stats.json" \
    "${PYTHON_BIN}" -m uvicorn app.main:app --host 127.0.0.1 --port 8081
  ) >"${LOG_DIR}/embedding-service.log" 2>&1 &
  wait_http "${EMBEDDING_SERVICE_URL}/health" "embedding-service" 600
}

stop_api() {
  if [[ -f "${RUN_ROOT}/rag_api.pid" ]]; then
    local pid
    pid="$(cat "${RUN_ROOT}/rag_api.pid")"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 2
    fi
  fi
  pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8000" >/dev/null 2>&1 || true
}

start_api() {
  local log_name="$1"
  local judge_method="$2"
  local classifier_enabled="$3"
  local cross_encoder_model="$4"

  stop_api
  start_embedding_service
  log "Starting RAG API log=${log_name} judge=${judge_method} classifier=${classifier_enabled} reranker=${cross_encoder_model:-off}"
  PYTHONUNBUFFERED=1 \
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
  OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT}" \
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}" \
  OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX}" \
  OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT}" \
  QUERY_REWRITE_MODEL="$(echo "${DIAGNOSTIC_LLM_MODELS}" | awk '{print $1}')" \
  EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL}" \
  EMBEDDING_TIMEOUT="${EMBEDDING_TIMEOUT:-120}" \
  QDRANT_HOST="127.0.0.1" \
  QDRANT_PORT="6333" \
  QDRANT_COLLECTION="${QDRANT_COLLECTION}" \
  QDRANT_VECTOR_NAME="${QDRANT_VECTOR_NAME}" \
  QDRANT_SPARSE_VECTOR_NAME="${QDRANT_SPARSE_VECTOR_NAME}" \
  RAG_RETRIEVER="${RAG_RETRIEVER:-qdrant_hybrid}" \
  RAG_CORPUS_VERSION="pubmedqa-official-pqal-test-v1" \
  BM25_STATS_PATH="${PROJECT_ROOT}/${DATA_DIR}/bm25_stats.json" \
  CROSS_ENCODER_MODEL="${cross_encoder_model}" \
  CROSS_ENCODER_DEVICE="${CROSS_ENCODER_DEVICE}" \
  RAG_EVIDENCE_JUDGE_METHOD="${judge_method}" \
  RAG_EVIDENCE_CLASSIFIER_ENABLED="${classifier_enabled}" \
  RAG_EVIDENCE_CLASSIFIER_MODEL_PATH="${DIAGNOSTIC_CLASSIFIER_MODEL_PATH}" \
  RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH="${DIAGNOSTIC_CLASSIFIER_MODEL_PATH:+${DIAGNOSTIC_CLASSIFIER_MODEL_PATH}/calibration.json}" \
  RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0 \
  RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD="${RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD:-2.0}" \
  RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD="${RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD:-2.0}" \
  RAG_ANSWER_QUALITY_GATE_ENABLED=false \
  "${PYTHON_BIN}" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 \
    >"${LOG_DIR}/rag-api-${log_name}.log" 2>&1 &
  echo $! > "${RUN_ROOT}/rag_api.pid"
  wait_http "${RAG_API_URL}/api/health" "rag-api ${log_name}" 600
}

pull_models() {
  if [[ "${OLLAMA_PULL_MODELS}" != "1" ]]; then
    return
  fi
  for model in ${DIAGNOSTIC_LLM_MODELS} ${DIAGNOSTIC_RAG_MODELS}; do
    log "Pulling Ollama model: ${model}"
    ollama pull "${model}"
  done
}

build_quick_sets() {
  log "Building quick PQA-L sets"
  "${PYTHON_BIN}" scripts/eval/build_pqal_quick_sets.py \
    --dataset "${DATA_DIR}/eval.json" \
    --out-dir "${QUICK_DIR}" \
    --per-label "${QUICK_PQAL_PER_LABEL}" \
    --yes-count "${QUICK_PQAL_YES_COUNT}"
}

dataset_path() {
  if [[ "${DIAGNOSTIC_SCOPE}" == "official" ]]; then
    echo "${DATA_DIR}/eval.json"
  else
    echo "${QUICK_DIR}/balanced${BALANCED_COUNT}.json"
  fi
}

build_index_once() {
  log "Indexing official PQA-L 500 corpus once"
  local args=(
    scripts/rag/01_build_index.py
    --chunks "${DATA_DIR}/chunks.parquet"
    --collection "${QDRANT_COLLECTION}"
    --qdrant-url "${QDRANT_URL}"
    --dense-vector-name "${QDRANT_VECTOR_NAME}"
    --sparse-vector-name "${QDRANT_SPARSE_VECTOR_NAME}"
    --corpus-version "pubmedqa-official-pqal-test-v1"
    --bm25-stats-out "${DATA_DIR}/bm25_stats.json"
    --manifest-out "${DATA_DIR}/index_manifest.json"
    --checkpoint-out "${DATA_DIR}/index_checkpoint.json"
    --chunk-store "data/indexes/pubmedqa_official_pqal_test/chunk_store.sqlite"
    --embedding-batch-size "${OFFICIAL_PQAL500_EMBEDDING_BATCH_SIZE:-32}"
    --upsert-batch-size "${OFFICIAL_PQAL500_UPSERT_BATCH_SIZE:-128}"
  )
  if [[ "${RECREATE_INDEX}" == "1" ]]; then
    args+=(--recreate)
  else
    args+=(--resume)
  fi
  "${PYTHON_BIN}" "${args[@]}"
}

run_direct() {
  local model="$1"
  local evidence_mode="$2"
  local prompt_style="$3"
  local safe
  safe="$(sanitize_label "${model}")"
  local label="direct_${safe}_${evidence_mode}_${prompt_style}_${DIAGNOSTIC_SCOPE}"
  log "Running direct judge ${label}"
  "${PYTHON_BIN}" scripts/eval/run_pubmedqa_direct_judge.py \
    --dataset "$(dataset_path)" \
    --corpus "${DATA_DIR}/corpus.json" \
    --ollama-url "${OLLAMA_BASE_URL}" \
    --model "${model}" \
    --temperature "${PUBMEDQA_EVAL_TEMPERATURE}" \
    --evidence-mode "${evidence_mode}" \
    --prompt-style "${prompt_style}" \
    --json-out "${REPORT_ROOT}/direct/${label}.json" \
    --md-out "${REPORT_ROOT}/direct/${label}.md"
}

run_rag() {
  local model="$1"
  local label="$2"
  local top_k="$3"
  local judge_method="$4"
  local classifier_enabled="$5"
  local cross_encoder_model="$6"
  local safe
  safe="$(sanitize_label "${model}")"
  local run_label="rag_${label}_${safe}_top${top_k}_${DIAGNOSTIC_SCOPE}"
  log "Running RAG eval ${run_label}"
  CROSS_ENCODER_MODEL="${cross_encoder_model}" \
  REPORT_DIR="${REPORT_ROOT}/rag" \
  PUBMEDQA_EVAL_LABEL="${run_label}" \
  RAG_API_URL="${RAG_API_URL}" \
  PUBMEDQA_EVAL_MODEL="${model}" \
  PUBMEDQA_EVAL_CANDIDATE_K="${PUBMEDQA_EVAL_CANDIDATE_K}" \
  PUBMEDQA_EVAL_TOP_K="${top_k}" \
  PUBMEDQA_EVAL_TEMPERATURE="${PUBMEDQA_EVAL_TEMPERATURE}" \
  RAG_EVIDENCE_JUDGE_METHOD="${judge_method}" \
  RAG_EVIDENCE_CLASSIFIER_ENABLED="${classifier_enabled}" \
  "${PYTHON_BIN}" scripts/rag/06_evaluate_pubmedqa_benchmark.py \
    --dataset "$(dataset_path)" \
    --api-url "${RAG_API_URL}" \
    --model "${model}" \
    --candidate-k "${PUBMEDQA_EVAL_CANDIDATE_K}" \
    --top-k "${top_k}" \
    --temperature "${PUBMEDQA_EVAL_TEMPERATURE}" \
    --mode benchmark_pqal \
    --json-out "${REPORT_ROOT}/rag/${run_label}.json" \
    --md-out "${REPORT_ROOT}/rag/${run_label}.md"
}

write_summary() {
  log "Writing diagnostic summary"
  "${PYTHON_BIN}" scripts/eval/write_pubmedqa_diagnostic_summary.py \
    --report-root "${REPORT_ROOT}" \
    --json-out "${RUN_ROOT}/pubmedqa_diagnostic_summary.json" \
    --md-out "${RUN_ROOT}/pubmedqa_diagnostic_summary.md"
}

main() {
  log "RUN_ROOT=${RUN_ROOT}"
  log "DIAGNOSTIC_SCOPE=${DIAGNOSTIC_SCOPE}"
  install_system_deps
  install_python_deps
  ensure_qdrant
  ensure_ollama
  start_embedding_service
  pull_models
  build_quick_sets
  build_index_once

  for model in ${DIAGNOSTIC_LLM_MODELS}; do
    run_direct "${model}" none compact
    run_direct "${model}" oracle compact
    for prompt_style in ${DIAGNOSTIC_PROMPT_STYLES}; do
      if [[ "${prompt_style}" == "compact" ]]; then
        continue
      fi
      run_direct "${model}" oracle "${prompt_style}"
    done
  done

  start_api "llm_reranker_on" llm false "${CROSS_ENCODER_MODEL_ON}"
  for model in ${DIAGNOSTIC_RAG_MODELS}; do
    for top_k in ${DIAGNOSTIC_TOP_KS}; do
      run_rag "${model}" "llm_reranker_on" "${top_k}" llm false "${CROSS_ENCODER_MODEL_ON}"
    done
  done

  start_api "llm_reranker_off" llm false ""
  for model in ${DIAGNOSTIC_RAG_MODELS}; do
    run_rag "${model}" "llm_reranker_off" 1 llm false ""
  done

  if [[ -n "${DIAGNOSTIC_CLASSIFIER_MODEL_PATH}" && -d "${DIAGNOSTIC_CLASSIFIER_MODEL_PATH}" ]]; then
    start_api "classifier_reranker_on" classifier true "${CROSS_ENCODER_MODEL_ON}"
    for model in ${DIAGNOSTIC_RAG_MODELS}; do
      run_rag "${model}" "classifier_reranker_on" 1 classifier true "${CROSS_ENCODER_MODEL_ON}"
    done
  else
    log "Skipping classifier ablation. Set DIAGNOSTIC_CLASSIFIER_MODEL_PATH=/path/to/best to enable it."
  fi

  write_summary
  log "Done. Summary: ${RUN_ROOT}/pubmedqa_diagnostic_summary.md"
}

main "$@"
