#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SETUP="${RUN_SETUP:-1}"
REQUIRE_H100="${REQUIRE_H100:-1}"
RUN_PUBMEDQA_PIPELINE_EVAL="${RUN_PUBMEDQA_PIPELINE_EVAL:-0}"
MODELS="${MODELS:-}"

export HF_HOME="${HF_HOME:-/content/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/chatbot-med-embedding-benchmark}"
export RUNALL_CUDA_VISIBLE_DEVICES="${RUNALL_CUDA_VISIBLE_DEVICES:-0}"
export RUNALL_DEVICE_0="${RUNALL_DEVICE_0:-cuda:0}"
export RUNALL_DEVICE_1="${RUNALL_DEVICE_1:-cuda:0}"
export RUNALL_EVAL_DEVICE="${RUNALL_EVAL_DEVICE:-cuda:0}"
export RUNALL_CROSS_ENCODER_DEVICE="${RUNALL_CROSS_ENCODER_DEVICE:-cuda:0}"
export RUNALL_AUTOTUNE_SAMPLE_SIZE="${RUNALL_AUTOTUNE_SAMPLE_SIZE:-1024}"
export RUNALL_AUTOTUNE_MAX_BATCH="${RUNALL_AUTOTUNE_MAX_BATCH:-1024}"
export RUNALL_AUTOTUNE_SAFETY_FACTOR="${RUNALL_AUTOTUNE_SAFETY_FACTOR:-0.70}"
export RUNALL_UPSERT_BATCH_SIZE="${RUNALL_UPSERT_BATCH_SIZE:-128}"
export RUNALL_CLEANUP_MODEL_DATA="${RUNALL_CLEANUP_MODEL_DATA:-1}"
export RUNALL_STOP_ON_ERROR="${RUNALL_STOP_ON_ERROR:-1}"
export QDRANT_BIN="${QDRANT_BIN:-${HOME}/bin/qdrant}"
export QDRANT_START_MODE="${QDRANT_START_MODE:-local}"
export QDRANT_ALLOW_DOCKER="${QDRANT_ALLOW_DOCKER:-0}"
export QDRANT_STORAGE_DIR="${QDRANT_STORAGE_DIR:-/content/qdrant_storage}"

if [[ "${RUN_SETUP}" == "1" ]]; then
  REQUIRE_H100="${REQUIRE_H100}" PYTHON_BIN="${PYTHON_BIN}" scripts/colab/setup_embedding_benchmark_h100.sh
else
  if [[ "${REQUIRE_H100}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/colab/check_h100_runtime.py --require-h100
  else
    "${PYTHON_BIN}" scripts/colab/check_h100_runtime.py --no-require-h100
  fi
fi

"${PYTHON_BIN}" scripts/colab/prepare_embedding_shards.py

args=(--sequential-shards --precision bf16)
if [[ "${RUN_PUBMEDQA_PIPELINE_EVAL}" != "1" ]]; then
  args+=(--skip-pubmedqa-pipeline-eval)
fi
if [[ -n "${MODELS}" ]]; then
  args+=(--models)
  for model in ${MODELS}; do
    args+=("${model}")
  done
fi

echo "==> Starting embedding benchmark on one H100"
echo "    OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "    MODELS=${MODELS:-all registry models}"
echo "    RUN_PUBMEDQA_PIPELINE_EVAL=${RUN_PUBMEDQA_PIPELINE_EVAL}"

bash benchmarks/runall.sh "${args[@]}"
