#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-${PY:-python3}}"
RAW_SOURCE_DIR="${RAW_SOURCE_DIR:-data/raw/pubmedqa_official}"
RAW_DIR="${RAW_DIR:-${RAW_SOURCE_DIR}/data}"
RUN_ROOT="${RUN_ROOT:-artifacts/classifier/pubmedqa_biolinkbert_v3_$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_ROOT="${DATA_ROOT:-data/interim/classifier/pubmedqa_biolinkbert_v3}"
MODEL_NAME="${MODEL_NAME:-michiyasunaga/BioLinkBERT-large}"
SEEDS="${SEEDS:-47 123 2026}"
EPOCHS_PHASE_I="${EPOCHS_PHASE_I:-5}"
EPOCHS_FINAL="${EPOCHS_FINAL:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
LEARNING_RATE_PHASE_I="${LEARNING_RATE_PHASE_I:-2e-5}"
LEARNING_RATE_FINAL="${LEARNING_RATE_FINAL:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
DEV_FRACTION="${DEV_FRACTION:-0.20}"
MAX_TRAIN_PER_LABEL="${MAX_TRAIN_PER_LABEL:-20000}"
MAX_DEV_PER_LABEL="${MAX_DEV_PER_LABEL:-500}"
MIN_DEV_PER_LABEL="${MIN_DEV_PER_LABEL:-10}"
AUX_WEIGHT="${AUX_WEIGHT:-0.10}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}"
COMMAND_LOG="${RUN_ROOT}/command_log.jsonl"
SAFE_MODEL="$(echo "${MODEL_NAME}" | tr '/:' '__')"

echo "==> PubMedQA BioLinkBERT v3 multi-phase run"
echo "    RUN_ROOT=${RUN_ROOT}"
echo "    DATA_ROOT=${DATA_ROOT}"
echo "    MODEL_NAME=${MODEL_NAME}"
echo "    SEEDS=${SEEDS}"
echo "    Phase I: PQA-A question+context -> ${EPOCHS_PHASE_I} epochs"
echo "    Final: PQA-L question+context + LONG_ANSWER BoW aux -> ${EPOCHS_FINAL} epochs"
echo "    Decision path: argmax softmax; no threshold tuning"
echo "    Loss: standard CE + optional BoW aux; no class weights; no focal loss"
echo "    COMMAND_LOG=${COMMAND_LOG}"

log_command() {
  local name="$1"
  shift
  "${PYTHON_BIN}" - "$COMMAND_LOG" "$name" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

path = sys.argv[1]
name = sys.argv[2]
cmd = sys.argv[3:]
with open(path, "a", encoding="utf-8") as file:
    file.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "name": name, "cmd": cmd}) + "\n")
PY
}

ensure_raw_pubmedqa_sources() {
  local pqal_path="${RAW_DIR}/ori_pqal.json"
  local pqaa_path="${RAW_DIR}/ori_pqaa.json"
  if [[ -s "${pqal_path}" && -s "${pqaa_path}" ]]; then
    echo "==> Found PubMedQA raw sources"
    return
  fi

  local bootstrap_out="${DATA_ROOT}/_pubmedqa_raw_bootstrap"
  local cmd=(
    "${PYTHON_BIN}" scripts/classifier/prepare_pubmedqa_deberta_dataset.py
    --download
    --download-pqaa
    --source-dir "${RAW_SOURCE_DIR}"
    --heldout-eval data/benchmarks/pubmedqa/official_pqal_test/eval.json
    --out-dir "${bootstrap_out}"
    --max-train-per-label 1
    --max-dev-per-label 1
    --min-dev-per-label 1
  )
  log_command "bootstrap_pubmedqa_raw_sources" "${cmd[@]}"
  echo "==> Downloading PubMedQA raw sources"
  "${cmd[@]}"

  if [[ ! -s "${pqal_path}" || ! -s "${pqaa_path}" ]]; then
    cat <<EOF
Missing required raw PubMedQA files:
  ${pqal_path}
  ${pqaa_path}

If Google Drive blocks PQA-A, manually copy official ori_pqaa.json to:
  ${pqaa_path}
EOF
    exit 1
  fi
}

run_prepare() {
  local name="$1"
  shift
  local out_dir="${DATA_ROOT}/${name}"
  local cmd=(
    "${PYTHON_BIN}" scripts/classifier/prepare_pubmedqa_deberta_dataset.py
    --heldout-eval data/benchmarks/pubmedqa/official_pqal_test/eval.json
    --out-dir "${out_dir}"
    --seed 47
    --dev-fraction "${DEV_FRACTION}"
    --min-dev-per-label "${MIN_DEV_PER_LABEL}"
    --max-train-per-label "${MAX_TRAIN_PER_LABEL}"
    --max-dev-per-label "${MAX_DEV_PER_LABEL}"
    "$@"
  )
  log_command "prepare_${name}" "${cmd[@]}"
  echo "==> Preparing dataset ${name}"
  "${cmd[@]}"
}

run_train() {
  local stage="$1"
  local train_jsonl="$2"
  local dev_jsonl="$3"
  local init_model="$4"
  local out_dir="$5"
  local seed="$6"
  local epochs="$7"
  local learning_rate="$8"
  local aux_flag="$9"

  if [[ "${SKIP_COMPLETED}" == "1" && -s "${out_dir}/best/dev_metrics.json" ]]; then
    echo "==> SKIP completed ${stage} seed=${seed}"
    echo "    best=${out_dir}/best"
    return
  fi

  local cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    scripts/classifier/train_deberta_pubmedqa.py
    --train-jsonl "${train_jsonl}"
    --dev-jsonl "${dev_jsonl}"
    --out-dir "${out_dir}"
    --model-name "${init_model}"
    --max-length "${MAX_LENGTH}"
    --batch-size "${BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --gradient-accumulation "${GRADIENT_ACCUMULATION}"
    --epochs "${epochs}"
    --learning-rate "${learning_rate}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"
    --early-stopping-patience 4
    --seed "${seed}"
    --selection-metric accuracy_macro_f1
    --amp bf16
    --no-class-weighted-loss
    --no-balanced-sampling
    --focal-loss-gamma 0.0
    --no-tune-thresholds
    --num-workers "${NUM_WORKERS}"
    --log-every 25
  )
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    cmd+=(--gradient-checkpointing)
  else
    cmd+=(--no-gradient-checkpointing)
  fi
  if [[ "${aux_flag}" == "aux" ]]; then
    cmd+=(--aux-long-answer-bow --aux-long-answer-bow-weight "${AUX_WEIGHT}")
  fi

  log_command "train_${stage}_${SAFE_MODEL}_seed_${seed}" "${cmd[@]}"
  mkdir -p "$(dirname "${out_dir}")"
  echo "==> TRAIN ${stage} seed=${seed}"
  echo "    init_model=${init_model}"
  echo "    out_dir=${out_dir}"
  echo "    live_log=${out_dir}.log"
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" "${cmd[@]}" 2>&1 | tee "${out_dir}.log"
}

ensure_raw_pubmedqa_sources

run_prepare "pqaa_only" \
  --source-json "${RAW_DIR}/ori_pqaa.json"

run_prepare "pqal_only" \
  --source-json "${RAW_DIR}/ori_pqal.json" \
  --priority-source-name ori_pqal.json

for seed in ${SEEDS}; do
  phase_i_dir="${RUN_ROOT}/phase_i_pqaa/${SAFE_MODEL}/seed_${seed}"
  final_dir="${RUN_ROOT}/phase_i_final_pqal_aux/${SAFE_MODEL}/seed_${seed}"

  run_train \
    "phase_i_pqaa" \
    "${DATA_ROOT}/pqaa_only/train.jsonl" \
    "${DATA_ROOT}/pqaa_only/dev.jsonl" \
    "${MODEL_NAME}" \
    "${phase_i_dir}" \
    "${seed}" \
    "${EPOCHS_PHASE_I}" \
    "${LEARNING_RATE_PHASE_I}" \
    "aux"

  run_train \
    "phase_i_final_pqal_aux" \
    "${DATA_ROOT}/pqal_only/train.jsonl" \
    "${DATA_ROOT}/pqal_only/dev.jsonl" \
    "${phase_i_dir}/best" \
    "${final_dir}" \
    "${seed}" \
    "${EPOCHS_FINAL}" \
    "${LEARNING_RATE_FINAL}" \
    "aux"
done

"${PYTHON_BIN}" scripts/classifier/audit_pubmedqa_classifier_data.py \
  --train-jsonl "${DATA_ROOT}/pqal_only/train.jsonl" \
  --dev-jsonl "${DATA_ROOT}/pqal_only/dev.jsonl" \
  --json-out "${RUN_ROOT}/data_audit.json" \
  --md-out "${RUN_ROOT}/data_audit.md"

"${PYTHON_BIN}" scripts/classifier/summarize_pubmedqa_experiments.py \
  --run-root "${RUN_ROOT}" \
  --include-dataset phase_i_final_pqal_aux \
  --json-out "${RUN_ROOT}/experiment_summary.json" \
  --md-out "${RUN_ROOT}/experiment_summary.md"

echo "PubMedQA BioLinkBERT v3 artifacts: ${RUN_ROOT}"
echo "Command log: ${COMMAND_LOG}"
echo "Experiment summary: ${RUN_ROOT}/experiment_summary.md"
