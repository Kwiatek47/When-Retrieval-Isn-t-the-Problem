#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Paper-backed PubMedQA path:
# BioLinkBERT-large + CLS 3-way softmax + LONG_ANSWER BoW auxiliary.
# Main decision path is argmax, so threshold tuning is disabled by default.
export MODEL_NAMES="${MODEL_NAMES:-michiyasunaga/BioLinkBERT-large}"
export RUN_BIOMED_ABLATION="${RUN_BIOMED_ABLATION:-0}"
export RUN_SEED_SWEEP="${RUN_SEED_SWEEP:-1}"
export BEST_MODEL="${BEST_MODEL:-michiyasunaga/BioLinkBERT-large}"
export BEST_DATASET="${BEST_DATASET:-pqaa_pqal_long_answer_aux}"
export SEEDS="${SEEDS:-123 2026}"
export TUNE_THRESHOLDS="${TUNE_THRESHOLDS:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SELECTION_METRIC="${SELECTION_METRIC:-accuracy_macro_f1}"
export CLASS_WEIGHTED_LOSS="${CLASS_WEIGHTED_LOSS:-0}"
export BALANCED_SAMPLING="${BALANCED_SAMPLING:-0}"
export RUN_PQAL_ONLY="${RUN_PQAL_ONLY:-0}"
export RUN_PQAA_PQAL="${RUN_PQAA_PQAL:-1}"
export RUN_PQAA_PQAL_AUX="${RUN_PQAA_PQAL_AUX:-1}"
export EPOCHS="${EPOCHS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export FOCAL_LOSS_GAMMA="${FOCAL_LOSS_GAMMA:-0.0}"
export AUX_WEIGHT="${AUX_WEIGHT:-0.10}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

PYTHON_BIN="${PYTHON_BIN:-${PY:-python3}}" scripts/classifier/run_pubmedqa_research_experiments.sh
