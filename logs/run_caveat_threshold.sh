#!/bin/bash
# Registered caveat-rule / maybe-threshold probe (PLAN, "Reguła zastrzeżenia i próg maybe"):
# 2 models x 2 prompts x 2 inputs on all 1000 PQA-L questions, round 0 only. qwen3:30b runs on the
# GPU0 server (11434), mistral-small3.2:24b on the GPU2 server (11436), in parallel; then both
# models are unloaded and the registered analysis runs. Resumable.
# Start only after the registration is committed:
#   setsid nohup logs/run_caveat_threshold.sh > logs/caveat_threshold.log 2>&1 < /dev/null &
cd /raid/s203270/When-Retrieval-Isn-t-the-Problem || exit 1
PY=/raid/s203270/miniconda3/envs/llm_env/bin/python

chain() {  # model, think, base url
  for prompt in label-probs@1 label-caveat-probs@1; do
    for input in context context+conclusion; do
      $PY scripts/agents/run_label_probe.py --prompt "$prompt" --input "$input" --split all \
        --model "$1" --think "$2" --base-url "$3" --workers 4
    done
  done
}

chain qwen3:30b off http://localhost:11434 > logs/caveat_threshold_qwen3.log 2>&1 &
chain mistral-small3.2:24b default http://localhost:11436 > logs/caveat_threshold_mistral.log 2>&1 &
wait
echo "=== runs finished $(date +%F_%T) ==="
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3:30b","keep_alive":0}' > /dev/null
curl -s http://localhost:11436/api/generate -d '{"model":"mistral-small3.2:24b","keep_alive":0}' > /dev/null
echo "=== models unloaded $(date +%F_%T) ==="
$PY scripts/agents/analyze_caveat_threshold.py
echo "=== finished $(date +%F_%T) ==="
