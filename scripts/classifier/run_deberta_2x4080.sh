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

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Single A40 (default GPU 1; GPU 2 is often occupied by other users on this host).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/classifier/train_deberta_pubmedqa.py \
  --train-jsonl "${TRAIN_JSONL:-data/interim/classifier/pubmedqa_deberta/train.jsonl}" \
  --dev-jsonl "${DEV_JSONL:-data/interim/classifier/pubmedqa_deberta/dev.jsonl}" \
  --out-dir "${OUT_DIR:-artifacts/classifier/pubmedqa_deberta}" \
  --model-name "${MODEL_NAME:-microsoft/deberta-v3-base}" \
  --max-length "${MAX_LENGTH:-512}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-4}" \
  --epochs "${EPOCHS:-5}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --warmup-ratio "${WARMUP_RATIO:-0.10}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-2}" \
  --amp "${AMP:-bf16}" \
  --gradient-checkpointing \
  --class-weighted-loss \
  --no-balanced-sampling \
  --num-workers "${NUM_WORKERS:-4}" \
  --log-every "${LOG_EVERY:-50}"
