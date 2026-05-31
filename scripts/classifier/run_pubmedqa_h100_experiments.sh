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

RAW_DIR="${RAW_DIR:-data/raw/pubmedqa_official/data}"
RUN_ROOT="${RUN_ROOT:-artifacts/classifier/pubmedqa_research_$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_ROOT="${DATA_ROOT:-data/interim/classifier/pubmedqa_research}"
SEEDS="${SEEDS:-47 123 2026}"
MODEL_NAMES="${MODEL_NAMES:-microsoft/deberta-v3-base microsoft/deberta-v3-large}"
BIOMED_MODEL_NAMES="${BIOMED_MODEL_NAMES:-microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MIN_DEV_PER_LABEL="${MIN_DEV_PER_LABEL:-10}"
DEV_FRACTION="${DEV_FRACTION:-0.20}"
MAX_TRAIN_PER_LABEL="${MAX_TRAIN_PER_LABEL:-20000}"
MAX_DEV_PER_LABEL="${MAX_DEV_PER_LABEL:-500}"
FOCAL_LOSS_GAMMA="${FOCAL_LOSS_GAMMA:-1.5}"
AUX_WEIGHT="${AUX_WEIGHT:-0.10}"

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}"
COMMAND_LOG="${RUN_ROOT}/command_log.jsonl"

echo "==> PubMedQA classifier research run"
echo "    RUN_ROOT=${RUN_ROOT}"
echo "    DATA_ROOT=${DATA_ROOT}"
echo "    MODEL_NAMES=${MODEL_NAMES}"
echo "    BIOMED_MODEL_NAMES=${BIOMED_MODEL_NAMES}"
echo "    NPROC_PER_NODE=${NPROC_PER_NODE} BATCH_SIZE=${BATCH_SIZE} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "    GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION} EPOCHS=${EPOCHS}"
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
  echo "    out_dir=${out_dir}"
  "${cmd[@]}"
}

run_train() {
  local dataset_name="$1"
  local model_name="$2"
  local seed="$3"
  local aux_flag="$4"
  local safe_model
  safe_model="$(echo "${model_name}" | tr '/:' '__')"
  local out_dir="${RUN_ROOT}/${dataset_name}/${safe_model}/seed_${seed}"
  local cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    scripts/classifier/train_deberta_pubmedqa.py
    --train-jsonl "${DATA_ROOT}/${dataset_name}/train.jsonl"
    --dev-jsonl "${DATA_ROOT}/${dataset_name}/dev.jsonl"
    --out-dir "${out_dir}"
    --model-name "${model_name}"
    --max-length "${MAX_LENGTH}"
    --batch-size "${BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --gradient-accumulation "${GRADIENT_ACCUMULATION}"
    --epochs "${EPOCHS}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"
    --early-stopping-patience 3
    --seed "${seed}"
    --selection-metric macro_f1
    --amp bf16
    --gradient-checkpointing
    --class-weighted-loss
    --no-balanced-sampling
    --focal-loss-gamma "${FOCAL_LOSS_GAMMA}"
    --tune-thresholds
    --threshold-metric macro_f1
    --num-workers "${NUM_WORKERS}"
    --log-every 25
  )
  if [[ "${aux_flag}" == "aux" ]]; then
    cmd+=(--aux-long-answer-bow --aux-long-answer-bow-weight "${AUX_WEIGHT}")
  fi
  log_command "train_${dataset_name}_${safe_model}_seed_${seed}" "${cmd[@]}"
  mkdir -p "$(dirname "${out_dir}")"
  echo "==> Training dataset=${dataset_name} model=${model_name} seed=${seed} aux=${aux_flag}"
  echo "    out_dir=${out_dir}"
  echo "    live_log=${out_dir}.log"
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" "${cmd[@]}" 2>&1 | tee "${out_dir}.log"
}

run_prepare "pqal_only" \
  --source-json "${RAW_DIR}/ori_pqal.json" \
  --priority-source-name ori_pqal.json

run_prepare "pqaa_pqal" \
  --source-json "${RAW_DIR}/ori_pqaa.json" \
  --source-json "${RAW_DIR}/ori_pqal.json" \
  --priority-source-name ori_pqal.json

run_prepare "pqaa_pqal_long_answer_aux" \
  --source-json "${RAW_DIR}/ori_pqaa.json" \
  --source-json "${RAW_DIR}/ori_pqal.json" \
  --priority-source-name ori_pqal.json

for model_name in ${MODEL_NAMES}; do
  run_train "pqal_only" "${model_name}" "47" "no_aux"
  run_train "pqaa_pqal" "${model_name}" "47" "no_aux"
  run_train "pqaa_pqal_long_answer_aux" "${model_name}" "47" "aux"
done

if [[ "${RUN_BIOMED_ABLATION:-1}" == "1" ]]; then
  for model_name in ${BIOMED_MODEL_NAMES}; do
    run_train "pqaa_pqal_long_answer_aux" "${model_name}" "47" "aux"
  done
fi

if [[ "${RUN_SEED_SWEEP:-1}" == "1" ]]; then
  BEST_DATASET="${BEST_DATASET:-pqaa_pqal_long_answer_aux}"
  BEST_MODEL="${BEST_MODEL:-microsoft/deberta-v3-large}"
  for seed in ${SEEDS}; do
    run_train "${BEST_DATASET}" "${BEST_MODEL}" "${seed}" "aux"
  done
fi

"${PYTHON_BIN}" scripts/classifier/audit_pubmedqa_classifier_data.py \
  --train-jsonl "${DATA_ROOT}/pqaa_pqal_long_answer_aux/train.jsonl" \
  --dev-jsonl "${DATA_ROOT}/pqaa_pqal_long_answer_aux/dev.jsonl" \
  --json-out "${RUN_ROOT}/data_audit.json" \
  --md-out "${RUN_ROOT}/data_audit.md"

"${PYTHON_BIN}" scripts/classifier/summarize_pubmedqa_experiments.py \
  --run-root "${RUN_ROOT}" \
  --json-out "${RUN_ROOT}/experiment_summary.json" \
  --md-out "${RUN_ROOT}/experiment_summary.md"

echo "H100 experiment artifacts: ${RUN_ROOT}"
echo "Command log: ${COMMAND_LOG}"
echo "Experiment summary: ${RUN_ROOT}/experiment_summary.md"
