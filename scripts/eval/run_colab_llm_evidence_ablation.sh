#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-/content/drive/MyDrive/pubmedqa_llm_ablation_runs/llm_ablation_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${RUN_ROOT}/logs"
REPORT_ROOT="${RUN_ROOT}/reports"
DATA_DIR="data/benchmarks/pubmedqa/official_pqal_test"

EVAL_SCOPE="${LLM_ABLATION_EVAL_SCOPE:-quick}" # quick | official | both
QUICK_PQAL_PER_LABEL="${QUICK_PQAL_PER_LABEL:-30}"
QUICK_PQAL_YES_COUNT="${QUICK_PQAL_YES_COUNT:-100}"

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-MedicalChunk_pubmed_reviews_v1_medcpt_20260518}"
QDRANT_VECTOR_NAME="${QDRANT_VECTOR_NAME:-medcpt_dense}"
QDRANT_SPARSE_VECTOR_NAME="${QDRANT_SPARSE_VECTOR_NAME:-bm25_sparse}"
EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL:-http://127.0.0.1:8081}"
RAG_API_URL="${RAG_API_URL:-http://127.0.0.1:8000}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

SMALL_MEDICAL_MODEL="${SMALL_MEDICAL_MODEL:-hf.co/itlwas/BioMistral-7B-Q4_K_M-GGUF}"
LARGE_MEDICAL_MODEL="${LARGE_MEDICAL_MODEL:-hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_M}"
LLM_ABLATION_MODELS="${LLM_ABLATION_MODELS:-qwen2.5:7b ${SMALL_MEDICAL_MODEL} ${LARGE_MEDICAL_MODEL}}"

INSTALL_PY_DEPS="${INSTALL_PY_DEPS:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"
INSTALL_QDRANT="${INSTALL_QDRANT:-1}"
OLLAMA_PULL_MODELS="${OLLAMA_PULL_MODELS:-1}"
START_QDRANT="${START_QDRANT:-1}"
START_OLLAMA="${START_OLLAMA:-1}"
START_EMBEDDING_SERVICE="${START_EMBEDDING_SERVICE:-1}"
START_API="${START_API:-1}"
RECREATE_INDEX="${RECREATE_INDEX:-0}"

EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
CROSS_ENCODER_DEVICE="${CROSS_ENCODER_DEVICE:-cuda}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-600}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-4096}"
OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-120}"

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

install_python_deps() {
  if [[ "${INSTALL_PY_DEPS}" != "1" ]]; then
    return
  fi
  log "Installing Python dependencies"
  "${PYTHON_BIN}" -m pip install -q --upgrade pip
  "${PYTHON_BIN}" -m pip install -q -r requirements.txt -r services/embedding-service/requirements.txt
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

  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    log "Starting Qdrant with Docker"
    docker rm -f pubmedqa-qdrant-colab >/dev/null 2>&1 || true
    docker run -d --name pubmedqa-qdrant-colab \
      -p 6333:6333 -p 6334:6334 \
      -v "${RUN_ROOT}/qdrant_storage:/qdrant/storage" \
      qdrant/qdrant:latest >/dev/null
  else
    ensure_qdrant_binary
    log "Starting Qdrant binary"
    QDRANT__SERVICE__HTTP_PORT=6333 \
    QDRANT__STORAGE__STORAGE_PATH="${RUN_ROOT}/qdrant_storage" \
    nohup qdrant >"${LOG_DIR}/qdrant.log" 2>&1 &
  fi
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

start_api() {
  if curl -fsS "${RAG_API_URL}/api/health" >/dev/null 2>&1; then
    log "RAG API already running"
    return
  fi
  if [[ "${START_API}" != "1" ]]; then
    echo "RAG API is not running and START_API=0." >&2
    return 1
  fi
  log "Starting RAG API on ${RAG_API_URL}"
  PYTHONUNBUFFERED=1 \
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
  OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT}" \
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}" \
  OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX}" \
  OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT}" \
  QUERY_REWRITE_MODEL="qwen2.5:7b" \
  EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL}" \
  EMBEDDING_TIMEOUT="${EMBEDDING_TIMEOUT:-120}" \
  QDRANT_HOST="127.0.0.1" \
  QDRANT_PORT="6333" \
  QDRANT_COLLECTION="${QDRANT_COLLECTION}" \
  QDRANT_VECTOR_NAME="${QDRANT_VECTOR_NAME}" \
  QDRANT_SPARSE_VECTOR_NAME="${QDRANT_SPARSE_VECTOR_NAME}" \
  RAG_RETRIEVER="embedding_service" \
  RAG_CORPUS_VERSION="pubmedqa-official-pqal-test-v1" \
  BM25_STATS_PATH="${PROJECT_ROOT}/${DATA_DIR}/bm25_stats.json" \
  CROSS_ENCODER_DEVICE="${CROSS_ENCODER_DEVICE}" \
  RAG_EVIDENCE_JUDGE_METHOD="llm" \
  RAG_EVIDENCE_CLASSIFIER_ENABLED="false" \
  RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD="2.0" \
  RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD="2.0" \
  RAG_ANSWER_QUALITY_GATE_ENABLED="false" \
  "${PYTHON_BIN}" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 \
    >"${LOG_DIR}/rag-api.log" 2>&1 &
  wait_http "${RAG_API_URL}/api/health" "rag-api" 600
}

pull_models() {
  if [[ "${OLLAMA_PULL_MODELS}" != "1" ]]; then
    return
  fi
  for model in ${LLM_ABLATION_MODELS}; do
    log "Pulling Ollama model: ${model}"
    ollama pull "${model}"
  done
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

run_quick_for_model() {
  local model="$1"
  local safe
  safe="$(sanitize_label "${model}")"
  log "Running quick eval for ${model}"
  REPORT_DIR="${REPORT_ROOT}/quick_${safe}" \
  QUICK_PQAL_LABEL="quick_${safe}" \
  QUICK_PQAL_REQUIRE_CLASSIFIER=0 \
  QUICK_PQAL_PER_LABEL="${QUICK_PQAL_PER_LABEL}" \
  QUICK_PQAL_YES_COUNT="${QUICK_PQAL_YES_COUNT}" \
  RAG_API_URL="${RAG_API_URL}" \
  RAG_EVIDENCE_JUDGE_METHOD=llm \
  RAG_EVIDENCE_CLASSIFIER_ENABLED=false \
  PUBMEDQA_EVAL_MODEL="${model}" \
  PUBMEDQA_EVAL_CANDIDATE_K="${PUBMEDQA_EVAL_CANDIDATE_K:-20}" \
  PUBMEDQA_EVAL_TOP_K="${PUBMEDQA_EVAL_TOP_K:-1}" \
  PUBMEDQA_EVAL_TEMPERATURE="${PUBMEDQA_EVAL_TEMPERATURE:-0.0}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  scripts/eval/run_quick_pqal_eval.sh
}

run_official_for_model() {
  local model="$1"
  local safe
  safe="$(sanitize_label "${model}")"
  log "Running official PQA-L 500 eval for ${model}"
  REPORT_DIR="${REPORT_ROOT}/official_${safe}" \
  PUBMEDQA_EVAL_LABEL="official_${safe}_llm_evidence" \
  OFFICIAL_PQAL500_REQUIRE_CLASSIFIER=0 \
  OFFICIAL_PQAL500_RECREATE_INDEX=0 \
  RAG_API_URL="${RAG_API_URL}" \
  RAG_EVIDENCE_JUDGE_METHOD=llm \
  RAG_EVIDENCE_CLASSIFIER_ENABLED=false \
  PUBMEDQA_EVAL_MODEL="${model}" \
  PUBMEDQA_EVAL_CANDIDATE_K="${PUBMEDQA_EVAL_CANDIDATE_K:-20}" \
  PUBMEDQA_EVAL_TOP_K="${PUBMEDQA_EVAL_TOP_K:-1}" \
  PUBMEDQA_EVAL_TEMPERATURE="${PUBMEDQA_EVAL_TEMPERATURE:-0.0}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  scripts/eval/run_official_pqal500.sh
}

write_summary() {
  log "Writing ablation summary"
  SUMMARY_JSON="${RUN_ROOT}/llm_ablation_summary.json" \
  SUMMARY_MD="${RUN_ROOT}/llm_ablation_summary.md" \
  REPORT_ROOT="${REPORT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
json_out = Path(os.environ["SUMMARY_JSON"])
md_out = Path(os.environ["SUMMARY_MD"])

rows = []
for path in sorted(report_root.rglob("*.json")):
    if path.name.endswith((".summary.json", ".gate.json", ".methods.json", ".lock.json")):
        continue
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    summary = data.get("summary")
    if not isinstance(summary, dict) or "label_accuracy" not in summary:
        continue
    labels = summary.get("labels") or {}
    rows.append(
        {
            "report": str(path),
            "model": summary.get("model"),
            "cases": summary.get("case_count"),
            "label_accuracy": summary.get("label_accuracy"),
            "source_hit_at_1": summary.get("source_hit_at_1"),
            "source_hit_at_3": summary.get("source_hit_at_3"),
            "citation_pass_rate": summary.get("citation_pass_rate"),
            "mean_latency_ms": summary.get("mean_latency_ms"),
            "yes_acc": (labels.get("yes") or {}).get("accuracy"),
            "no_acc": (labels.get("no") or {}).get("accuracy"),
            "maybe_acc": (labels.get("maybe") or {}).get("accuracy"),
            "evidence_methods": summary.get("evidence_decision", {}).get("methods")
            or summary.get("evidence_decision", {}).get("evidence_method"),
        }
    )

rows.sort(key=lambda r: ((r["cases"] or 0), (r["label_accuracy"] or 0)), reverse=True)
json_out.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

lines = ["# LLM Evidence Judge Ablation", ""]
lines.append("| Model | Cases | Accuracy | Maybe acc/recall | No acc/recall | Hit@1 | Citation | Latency ms | Report |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    lines.append(
        "| {model} | {cases} | {acc:.3f} | {maybe:.3f} | {no:.3f} | {hit:.3f} | {cit:.3f} | {lat:.1f} | `{report}` |".format(
            model=r["model"] or "",
            cases=r["cases"] or 0,
            acc=r["label_accuracy"] or 0.0,
            maybe=r["maybe_acc"] or 0.0,
            no=r["no_acc"] or 0.0,
            hit=r["source_hit_at_1"] or 0.0,
            cit=r["citation_pass_rate"] or 0.0,
            lat=r["mean_latency_ms"] or 0.0,
            report=r["report"],
        )
    )
md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {json_out}")
print(f"Wrote {md_out}")
PY
}

main() {
  log "RUN_ROOT=${RUN_ROOT}"
  install_system_deps
  install_python_deps
  ensure_qdrant
  ensure_ollama
  start_embedding_service
  start_api
  pull_models
  build_index_once

  for model in ${LLM_ABLATION_MODELS}; do
    if [[ "${EVAL_SCOPE}" == "quick" || "${EVAL_SCOPE}" == "both" ]]; then
      run_quick_for_model "${model}"
    fi
    if [[ "${EVAL_SCOPE}" == "official" || "${EVAL_SCOPE}" == "both" ]]; then
      run_official_for_model "${model}"
    fi
  done

  write_summary
  log "Done. Summary: ${RUN_ROOT}/llm_ablation_summary.md"
}

main "$@"
