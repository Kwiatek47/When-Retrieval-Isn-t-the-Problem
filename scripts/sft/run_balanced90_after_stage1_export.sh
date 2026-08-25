#!/usr/bin/env bash
# After Stage1 GGUF export finishes: register Ollama on :11437 (GPU0),
# keep agents on :11434 (GPU2), run balanced90. Never touches GPU1.
set -euo pipefail
cd /home/s203270/raid/When-Retrieval-Isn-t-the-Problem
LOG=logs/sft_balanced90_stage1_pipeline.log
exec >>"$LOG" 2>&1
echo "[$(date -Is)] wait_for_export start"

EXPORT_MARKER=artifacts/sft/qwen14b-supervisor/ollama-stage1-q4_k_m/Modelfile
while pgrep -f '[p]ython -u scripts/sft/export_qwen14b_supervisor_ollama.py' >/dev/null; do
  echo "[$(date -Is)] export still running; gpu0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
  sleep 60
done

if [[ ! -f "$EXPORT_MARKER" ]]; then
  echo "[$(date -Is)] ERROR: Modelfile missing after export"
  exit 1
fi
GGUF=$(find artifacts/sft/qwen14b-supervisor/ollama-stage1-q4_k_m -name '*.gguf' | head -1)
if [[ -z "$GGUF" ]]; then
  echo "[$(date -Is)] ERROR: no GGUF found"
  exit 1
fi
echo "[$(date -Is)] export done gguf=$GGUF"

# Supervisor Ollama on GPU 0 / 11437
if curl -sf http://127.0.0.1:11437/api/tags >/dev/null 2>&1; then
  echo "[$(date -Is)] ollama :11437 already up"
else
  echo "[$(date -Is)] starting ollama supervisor on GPU0 :11437"
  CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11437 OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
    nohup /raid/s203270/bin/ollama serve > logs/ollama_gpu0_11437.log 2>&1 &
  for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:11437/api/tags >/dev/null && break
    sleep 1
  done
fi

echo "[$(date -Is)] registering qwen2.5-supervisor-14b-stage1"
OLLAMA_HOST=http://127.0.0.1:11437 /raid/s203270/bin/ollama create qwen2.5-supervisor-14b-stage1 \
  -f artifacts/sft/qwen14b-supervisor/ollama-stage1-q4_k_m/Modelfile

# Agents Ollama on GPU 2 / 11434
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[$(date -Is)] starting agents ollama on GPU2 :11434"
  CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
    nohup /raid/s203270/bin/ollama serve > logs/ollama_gpu2_11434.log 2>&1 &
  for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
    sleep 1
  done
fi

# shellcheck source=/dev/null
source /raid/s203270/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

export CUDA_VISIBLE_DEVICES=""
export RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best
export RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json
export RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0
export RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1=0
export RAG_EVIDENCE_CLASSIFIER_DEVICE=cpu
export OLLAMA_MODEL=qwen2.5:7b

echo "[$(date -Is)] starting balanced90"
python -u scripts/agents/evaluate_debate_pubmedqa.py \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json \
  --corpus data/benchmarks/pubmedqa/official_pqal_test/corpus.json \
  --backend ollama \
  --ollama-base-urls http://127.0.0.1:11434 \
  --supervisor-model qwen2.5-supervisor-14b-stage1 \
  --supervisor-base-url http://127.0.0.1:11437 \
  --no-supervisor-moderation \
  --aggregate-mode llm_director \
  --hint biolinkbert \
  --rounds 2 \
  --num-predict 900 \
  --agent-concurrency 2 \
  --case-concurrency 2 \
  --label stage1_director_balanced90_peer_llm_director \
  > logs/sft_balanced90_stage1.log 2>&1

echo "[$(date -Is)] BALANCED90_DONE"
tail -40 logs/sft_balanced90_stage1.log || true
