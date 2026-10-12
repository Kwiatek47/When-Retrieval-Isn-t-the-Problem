#!/bin/bash
# Registered model-dominance replay (PLAN, "Kto wygrywa spór"): every QM and MQ dispute is discussed
# again from its round-0 labels, with the old prompt (rerun) and with quoted evidence. QM on the
# GPU0 server (11434), MQ on the GPU2 server (11436), in parallel; then both models are unloaded
# and the registered analysis runs. Resumable.
# Start only after the registration is committed:
#   setsid nohup logs/run_negotiation_replay.sh > logs/negotiation_replay.log 2>&1 < /dev/null &
cd /raid/s203270/When-Retrieval-Isn-t-the-Problem || exit 1
PY=/raid/s203270/miniconda3/envs/llm_env/bin/python

chain() {  # source run, base url
  for arm in old evidence; do
    $PY scripts/agents/run_negotiation_replay.py --source "$1" --arm "$arm" --base-url "$2" --workers 4
  done
}

chain QM http://localhost:11434 > logs/negotiation_replay_QM.log 2>&1 &
chain MQ http://localhost:11436 > logs/negotiation_replay_MQ.log 2>&1 &
wait
echo "=== replays finished $(date +%F_%T) ==="
for url in http://localhost:11434 http://localhost:11436; do
  for model in qwen3:30b mistral-small3.2:24b; do
    curl -s "$url/api/generate" -d "{\"model\":\"$model\",\"keep_alive\":0}" > /dev/null
  done
done
echo "=== models unloaded $(date +%F_%T) ==="
$PY scripts/agents/analyze_negotiation_dominance.py
echo "=== finished $(date +%F_%T) ==="
