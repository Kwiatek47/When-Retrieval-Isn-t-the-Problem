#!/usr/bin/env bash
# =============================================================================
# ML4H arms — debate / panel / SC / AURC (balanced90 or PQA-L 500)
# =============================================================================
#
# Setup (once per machine)
#   python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
#   cp .env.example .env   # then pin BioLinkBERT seed47 — see TEAM-ML4H-RUNBOOK.md
#   ollama serve && ollama pull qwen2.5:7b
#
# Commands
#   smoke     mock debate + mock SC (no GPU)
#   check     Ollama + checkpoint + dataset
#   debate    4-agent debate + bert_gate (writes mean_llm_calls_per_case)
#   panel     majority vote, NO BioLinkBERT in aggregation (collapse control)
#   sc        compute-matched SC (--match-cost-report from debate JSON)
#   aurc      offline AURC: BERT-confidence vs random vs debate u-score
#   audit     evidence-audit probe (set MODEL=qwen2.5:72b or similar)
#   all90     smoke → debate → sc → aurc   on balanced90
#
# Split
#   SPLIT=90     default — balanced90.json
#   SPLIT=500    official PQA-L eval.json  (slow: ~8 LLM calls × 500 × 2 arms)
#
# Examples
#   scripts/agents/run_ml4h_v1_arms.sh check
#   scripts/agents/run_ml4h_v1_arms.sh debate
#   scripts/agents/run_ml4h_v1_arms.sh sc
#   SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh debate
#   SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh panel
#   SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh sc
#   scripts/agents/run_ml4h_v1_arms.sh aurc
#   MODEL=qwen2.5:72b scripts/agents/run_ml4h_v1_arms.sh audit
#
# Resume is always on. Do not delete *.checkpoint.jsonl.
# Remote Ollama: export OLLAMA_BASE_URL=http://<gpu-host>:11434
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

SPLIT="${SPLIT:-90}"
REPORT_DIR="${REPORT_DIR:-reports/debate}"

case "$SPLIT" in
  90)
    DATASET="${DATASET:-data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json}"
    DEBATE_LABEL="${DEBATE_LABEL:-debate_balanced90_ml4h_v1}"
    PANEL_LABEL="${PANEL_LABEL:-panel_majority_balanced90_ml4h_v1}"
    SC_LABEL="${SC_LABEL:-sc_balanced90_ml4h_v1}"
    ;;
  500)
    DATASET="${DATASET:-data/benchmarks/pubmedqa/official_pqal_test/eval.json}"
    DEBATE_LABEL="${DEBATE_LABEL:-debate_pqal500_ml4h_v1}"
    PANEL_LABEL="${PANEL_LABEL:-panel_majority_pqal500_ml4h_v1}"
    SC_LABEL="${SC_LABEL:-sc_pqal500_ml4h_v1}"
    ;;
  *)
    echo "ERROR: SPLIT must be 90 or 500 (got '$SPLIT')" >&2
    exit 1
    ;;
esac

export RAG_EVIDENCE_CLASSIFIER_MODEL_PATH="${RAG_EVIDENCE_CLASSIFIER_MODEL_PATH:-artifacts/classifier/pubmedqa_biolinkbert_seed47/best}"
export RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH="${RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH:-artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json}"
export RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1="${RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1:-0}"
export RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY="${RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY:-0}"

_load_ollama_env() {
  local envfile="$ROOT/.env"
  [[ -f "$envfile" ]] || return 0
  local key val
  for key in OLLAMA_BASE_URL OLLAMA_HOST OLLAMA_MODEL OLLAMA_TIMEOUT; do
    if [[ -n "${!key:-}" ]]; then
      continue
    fi
    val="$(
      awk -F= -v k="$key" '
        $1==k {
          sub(/^[^=]+=/, "")
          gsub(/^[[:space:]]+|"|'"'"'/, "")
          gsub(/[[:space:]]+$/, "")
          print
          exit
        }
      ' "$envfile"
    )"
    if [[ -n "$val" ]]; then
      export "${key}=${val}"
    fi
  done
}

_resolve_ollama_url() {
  local raw="${OLLAMA_BASE_URL:-${OLLAMA_HOST:-http://localhost:11434}}"
  raw="${raw%/}"
  case "$raw" in
    http://*|https://*) ;;
    *) raw="http://${raw}" ;;
  esac
  export OLLAMA_BASE_URL="$raw"
}

_ollama_recipe() {
  cat <<EOF
Ollama is not reachable at ${OLLAMA_BASE_URL}.
  ollama serve
  ollama pull ${OLLAMA_MODEL:-qwen2.5:7b}
  scripts/agents/run_ml4h_v1_arms.sh check
Remote:
  export OLLAMA_BASE_URL=http://<host>:11434
EOF
}

_require_checkpoint() {
  if [[ ! -d "$RAG_EVIDENCE_CLASSIFIER_MODEL_PATH" ]]; then
    echo "ERROR: BioLinkBERT checkpoint missing:" >&2
    echo "  $RAG_EVIDENCE_CLASSIFIER_MODEL_PATH" >&2
    echo "Copy seed47 from the team drive, or train:" >&2
    echo "  make classifier-train-biolinkbert-h100" >&2
    echo "See docs/research/TEAM-ML4H-RUNBOOK.md" >&2
    return 1
  fi
}

preflight() {
  _load_ollama_env
  _resolve_ollama_url
  local model="${OLLAMA_MODEL:-qwen2.5:7b}"
  echo "Ollama host: ${OLLAMA_BASE_URL}  model: ${model}  split: ${SPLIT}"
  if ! curl -fsS -m 5 "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
    echo "ERROR: $(_ollama_recipe)" >&2
    return 1
  fi
  if ! curl -fsS -m 5 "${OLLAMA_BASE_URL}/api/tags" | grep -q "qwen2.5"; then
    echo "ERROR: Ollama is up but qwen2.5:* is not pulled. Next: ollama pull ${model}" >&2
    return 1
  fi
  echo "Ollama preflight ok."
}

_debate_has_measured_cost() {
  local debate_json="$1"
  [[ -f "$debate_json" ]] || return 1
  "$PY" - "$debate_json" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.agents.evaluate_debate_pubmedqa import report_has_measured_cost
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sys.exit(0 if report_has_measured_cost(report) else 1)
PY
}

usage() {
  sed -n '2,36p' "$0"
  echo
  echo "Usage: SPLIT=90|500 scripts/agents/run_ml4h_v1_arms.sh <smoke|check|debate|panel|sc|aurc|audit|all90>"
}

smoke() {
  echo "== smoke: mock debate =="
  "$PY" scripts/agents/evaluate_debate_pubmedqa.py \
    --backend mock --limit 4 --rounds 2 \
    --report-dir "$REPORT_DIR" --label ml4h_v1_smoke_debate
  echo "== smoke: mock SC =="
  "$PY" scripts/agents/evaluate_self_consistency_pubmedqa.py \
    --backend mock --samples 5 --limit 4 \
    --report-dir "$REPORT_DIR" --label ml4h_v1_smoke_sc
}

debate() {
  preflight
  _require_checkpoint
  echo "== debate SPLIT=${SPLIT} label=${DEBATE_LABEL} (bert_gate, --resume) =="
  "$PY" scripts/agents/evaluate_debate_pubmedqa.py \
    --dataset "$DATASET" \
    --backend ollama --hint biolinkbert --aggregate-with-biolinkbert \
    --aggregate-mode bert_gate --rounds 2 \
    --uncertainty-route --uncertainty-objective macro_f1 \
    --classifier-path "$RAG_EVIDENCE_CLASSIFIER_MODEL_PATH" \
    --classifier-temperature-path "$RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH" \
    --resume --label "$DEBATE_LABEL" --report-dir "$REPORT_DIR"
}

panel() {
  preflight
  echo "== panel majority (NO BERT in aggregation) SPLIT=${SPLIT} =="
  "$PY" scripts/agents/evaluate_debate_pubmedqa.py \
    --dataset "$DATASET" \
    --backend ollama --hint none \
    --aggregate-mode majority --rounds 2 \
    --resume --label "$PANEL_LABEL" --report-dir "$REPORT_DIR"
}

sc() {
  preflight
  debate_json="$REPORT_DIR/${DEBATE_LABEL}.json"
  extra=()
  if _debate_has_measured_cost "$debate_json"; then
    extra+=(--match-cost-report "$debate_json")
    echo "== SC cost-matched to $debate_json =="
  else
    extra+=(--samples "${SAMPLES:-8}")
    echo "== SC with --samples ${SAMPLES:-8} (no measured debate cost — run debate first) =="
  fi
  "$PY" scripts/agents/evaluate_self_consistency_pubmedqa.py \
    --dataset "$DATASET" \
    --backend ollama --temperature 0.7 \
    --resume --label "$SC_LABEL" --report-dir "$REPORT_DIR" \
    "${extra[@]}"
}

aurc() {
  debate_json="$REPORT_DIR/${DEBATE_LABEL}.json"
  if [[ ! -f "$debate_json" ]]; then
    echo "ERROR: need $debate_json (run debate first)" >&2
    exit 1
  fi
  echo "== AURC baselines from $debate_json =="
  "$PY" scripts/agents/compute_aurc_baselines.py --debate-json "$debate_json"
}

audit() {
  preflight
  local model="${MODEL:-${OLLAMA_MODEL:-qwen2.5:14b}}"
  local label="${AUDIT_LABEL:-audit_${model//[:\/]/_}_split${SPLIT}}"
  echo "== evidence audit model=${model} split=${SPLIT} =="
  "$PY" scripts/agents/probe_evidence_audit.py \
    --dataset "$DATASET" \
    --backend ollama --model "$model" \
    --label "$label"
}

check() {
  _load_ollama_env
  _resolve_ollama_url
  echo "split: ${SPLIT}"
  echo "dataset: $DATASET"
  echo "classifier: ${RAG_EVIDENCE_CLASSIFIER_MODEL_PATH}"
  echo "debate: $REPORT_DIR/${DEBATE_LABEL}.json"
  echo "panel:  $REPORT_DIR/${PANEL_LABEL}.json"
  echo "sc:     $REPORT_DIR/${SC_LABEL}.json"
  if [[ -f "$DATASET" ]]; then
    echo "dataset file: ok"
  else
    echo "dataset file: MISSING"
  fi
  if _require_checkpoint; then
    echo "checkpoint: ok"
  fi
  if preflight; then
    :
  else
    echo "Ollama: DOWN"
  fi
  debate_json="$REPORT_DIR/${DEBATE_LABEL}.json"
  if _debate_has_measured_cost "$debate_json"; then
    echo "debate cost: measured (SC can --match-cost-report)"
  elif [[ -f "$debate_json" ]]; then
    echo "debate cost: JSON exists but mean_llm_calls_per_case missing — re-run debate"
  else
    echo "debate cost: no report yet"
  fi
}

cmd="${1:-}"
case "$cmd" in
  smoke) smoke ;;
  debate) debate ;;
  panel) panel ;;
  sc) sc ;;
  aurc) aurc ;;
  audit) audit ;;
  all90) SPLIT=90; debate; sc; aurc ;;
  check|preflight) check ;;
  *) usage; exit 1 ;;
esac
