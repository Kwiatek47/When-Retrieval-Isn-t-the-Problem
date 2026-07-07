#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/content/drive/MyDrive/pubmedqa_option_ranker_runs/option_ranker_$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_NAME="${OPTION_RANKER_MODEL_NAME:-michiyasunaga/BioLinkBERT-large}"
SEED="${OPTION_RANKER_SEED:-47}"
TRAIN_JSONL="${OPTION_RANKER_TRAIN_JSONL:-/content/pubmedqa_interim/pqaa_pqal_long_answer_aux/train.jsonl}"
DEV_JSONL="${OPTION_RANKER_DEV_JSONL:-/content/pubmedqa_interim/pqaa_pqal_long_answer_aux/dev.jsonl}"
INTERIM_DIR="${OPTION_RANKER_INTERIM_DIR:-/content/pubmedqa_interim/pqaa_pqal_long_answer_aux}"
RAW_DIR="${OPTION_RANKER_RAW_DIR:-/content/pubmedqa_raw/pubmedqa_official}"
INSTALL_PY_DEPS="${INSTALL_PY_DEPS:-1}"
PREPARE_DATA_IF_MISSING="${PREPARE_DATA_IF_MISSING:-1}"

MAX_LENGTH="${OPTION_RANKER_MAX_LENGTH:-512}"
BATCH_SIZE="${OPTION_RANKER_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${OPTION_RANKER_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${OPTION_RANKER_GRADIENT_ACCUMULATION:-16}"
EPOCHS="${OPTION_RANKER_EPOCHS:-5}"
LR="${OPTION_RANKER_LEARNING_RATE:-1e-5}"
AMP="${OPTION_RANKER_AMP:-bf16}"

mkdir -p "${RUN_ROOT}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

safe_model_name() {
  echo "$1" | tr '/:@. ' '_____' | tr -cd 'A-Za-z0-9_-'
}

if [[ "${INSTALL_PY_DEPS}" == "1" ]]; then
  log "Installing Python dependencies"
  "${PYTHON_BIN}" -m pip install -q --upgrade pip
  "${PYTHON_BIN}" -m pip install -q -r requirements.txt
fi

if [[ ! -s "${TRAIN_JSONL}" || ! -s "${DEV_JSONL}" ]]; then
  if [[ "${PREPARE_DATA_IF_MISSING}" != "1" ]]; then
    echo "Missing train/dev JSONL and PREPARE_DATA_IF_MISSING=0:" >&2
    echo "  ${TRAIN_JSONL}" >&2
    echo "  ${DEV_JSONL}" >&2
    exit 1
  fi
  log "Preparing PubMedQA train/dev JSONL"
  "${PYTHON_BIN}" scripts/classifier/prepare_pubmedqa_deberta_dataset.py \
    --source-dir "${RAW_DIR}" \
    --download \
    --download-pqaa \
    --heldout-eval data/benchmarks/pubmedqa/official_pqal_test/eval.json \
    --out-dir "${INTERIM_DIR}" \
    --seed "${SEED}" \
    --max-train-per-label "${OPTION_RANKER_MAX_TRAIN_PER_LABEL:-20000}" \
    --max-dev-per-label "${OPTION_RANKER_MAX_DEV_PER_LABEL:-500}" \
    --priority-source-name ori_pqal.json
  TRAIN_JSONL="${INTERIM_DIR}/train.jsonl"
  DEV_JSONL="${INTERIM_DIR}/dev.jsonl"
fi

log "Using train JSONL: ${TRAIN_JSONL}"
log "Using dev JSONL: ${DEV_JSONL}"

MODEL_SAFE="$(safe_model_name "${MODEL_NAME}")"
OUT_DIR="${RUN_ROOT}/${MODEL_SAFE}/seed_${SEED}"

log "Training option-ranker"
"${PYTHON_BIN}" scripts/classifier/train_pubmedqa_option_ranker.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --model-name "${MODEL_NAME}" \
  --out-dir "${OUT_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LR}" \
  --seed "${SEED}" \
  --amp "${AMP}" \
  --selection-metric "${OPTION_RANKER_SELECTION_METRIC:-accuracy_macro_f1}" \
  --log-every "${OPTION_RANKER_LOG_EVERY:-100}"

BEST_DIR="${OUT_DIR}/best"

log "Evaluating option-ranker on dev split"
"${PYTHON_BIN}" scripts/classifier/evaluate_pubmedqa_option_ranker.py \
  --jsonl "${DEV_JSONL}" \
  --model-dir "${BEST_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --amp "${AMP}" \
  --json-out "${OUT_DIR}/option_ranker_dev_eval.json" \
  --md-out "${OUT_DIR}/option_ranker_dev_eval.md"

log "Evaluating option-ranker on official PQA-L 500"
"${PYTHON_BIN}" scripts/classifier/evaluate_pubmedqa_option_ranker.py \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/eval.json \
  --corpus data/benchmarks/pubmedqa/official_pqal_test/corpus.json \
  --model-dir "${BEST_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --amp "${AMP}" \
  --json-out "${OUT_DIR}/official_pqal500_option_ranker.json" \
  --md-out "${OUT_DIR}/official_pqal500_option_ranker.md"

log "Done"
echo "BEST_DIR=${BEST_DIR}"
echo "DEV_REPORT=${OUT_DIR}/option_ranker_dev_eval.md"
echo "OFFICIAL_REPORT=${OUT_DIR}/official_pqal500_option_ranker.md"
