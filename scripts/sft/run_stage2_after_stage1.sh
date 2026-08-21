#!/usr/bin/env bash
set -euo pipefail
cd /home/s203270/raid/When-Retrieval-Isn-t-the-Problem
LOG=logs/sft_pipeline_watchdog.log
mkdir -p logs
exec >>"$LOG" 2>&1
echo "[$(date -Is)] watchdog start pid=$$"

stage1_running() {
  pgrep -f '[p]ython -u scripts/sft/train_qwen14b_supervisor_unsloth.py --stage director' >/dev/null
}

debates_running() {
  pgrep -f '[p]ython -u scripts/sft/generate_pubmedqa_supervisor_debates.py' >/dev/null
}

while stage1_running; do
  echo "[$(date -Is)] stage1 still running"
  sleep 120
done
echo "[$(date -Is)] stage1 process ended"

if [[ ! -d artifacts/sft/qwen14b-supervisor/stage1-director/lora_adapter ]]; then
  echo "[$(date -Is)] ERROR: stage1 adapter missing"
  exit 1
fi
tail -30 logs/sft_stage1_director.log || true

while debates_running; do
  echo "[$(date -Is)] waiting debates: $(wc -l < data/interim/sft/pubmedqa_supervisor/debates_internal_test.jsonl)/527"
  sleep 120
done
N=$(wc -l < data/interim/sft/pubmedqa_supervisor/debates_internal_test.jsonl)
echo "[$(date -Is)] debates_internal_test lines=$N"
if [[ "$N" -ge 527 ]]; then
  # shellcheck source=/dev/null
  source /raid/s203270/miniconda3/etc/profile.d/conda.sh
  conda activate llm_env
  python scripts/sft/prepare_pubmedqa_supervisor_sft.py \
    --debate-file data/interim/sft/pubmedqa_supervisor/debates_internal_test.jsonl \
    --split-name internal_test \
    --output-dir data/interim/sft/pubmedqa_supervisor
  echo "[$(date -Is)] internal_test SFT ready"
else
  echo "[$(date -Is)] WARN: internal_test incomplete ($N/527); continuing to stage2 anyway"
fi

# shellcheck source=/dev/null
source /raid/s203270/miniconda3/etc/profile.d/conda.sh
conda activate qwen14b-supervisor
export CUDA_VISIBLE_DEVICES=1
echo "[$(date -Is)] starting stage2 multitask"
python -u scripts/sft/train_qwen14b_supervisor_unsloth.py \
  --stage multitask \
  --adapter-path artifacts/sft/qwen14b-supervisor/stage1-director/lora_adapter \
  --train-file data/interim/sft/pubmedqa_supervisor/multitask_train.jsonl \
  --eval-file data/interim/sft/pubmedqa_supervisor/multitask_dev.jsonl \
  --output-dir artifacts/sft/qwen14b-supervisor/stage2-multitask \
  --epochs 1 \
  --learning-rate 5e-5 \
  --save-steps 100 \
  --eval-steps 100 \
  > logs/sft_stage2_multitask.log 2>&1
echo "[$(date -Is)] stage2 finished"
tail -30 logs/sft_stage2_multitask.log || true
echo PIPELINE_TRAINING_DONE
