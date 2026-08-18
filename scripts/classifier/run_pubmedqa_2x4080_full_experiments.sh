#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Full research run profile for a single A40 (override with env vars before `make`).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export RUN_ROOT="${RUN_ROOT:-artifacts/classifier/pubmedqa_research_2x4080_$(date -u +%Y%m%dT%H%M%SZ)}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
export EPOCHS="${EPOCHS:-8}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export RUN_BIOMED_ABLATION="${RUN_BIOMED_ABLATION:-1}"
export RUN_SEED_SWEEP="${RUN_SEED_SWEEP:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# The ablation loop already trains the best default variant once with seed 47.
# Sweep only the additional seeds by default, giving final seeds 47/123/2026 without duplicate work.
export SEEDS="${SEEDS:-123 2026}"

exec scripts/classifier/run_pubmedqa_research_experiments.sh
