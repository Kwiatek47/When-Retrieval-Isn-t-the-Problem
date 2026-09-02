#!/usr/bin/env bash
# =============================================================================
# ML4H v1 arms — one-shot runner (balanced90 only; never n=500)
# =============================================================================
#
# What this does
#   smoke   mock debate (limit 4) + mock SC (N=5, limit 4) — no GPU
#   debate  Ollama debate on balanced90; writes summary.cost.mean_llm_calls_per_case
#   sc      compute-matched SC via --match-cost-report (or SAMPLES=8 fallback)
#   all     smoke → debate → sc
#   check   preflight only (Ollama up? model pulled? debate JSON has cost?)
#
# One-shot on a machine with qwen2.5:7b
#   ollama serve                         # leave running
#   ollama pull qwen2.5:7b
#   scripts/agents/run_ml4h_v1_arms.sh check
#   scripts/agents/run_ml4h_v1_arms.sh debate   # then
#   scripts/agents/run_ml4h_v1_arms.sh sc
#
# Remote Ollama (no secrets printed)
#   export OLLAMA_BASE_URL=http://<gpu-host>:11434
#   # or the official alias:  export OLLAMA_HOST=http://<gpu-host>:11434
#   scripts/agents/run_ml4h_v1_arms.sh debate
#
# Env (optional; also read from .env if unset — OLLAMA_* only, never printed)
#   OLLAMA_BASE_URL / OLLAMA_HOST   default http://localhost:11434
#   OLLAMA_MODEL                    default qwen2.5:7b
#   PYTHON / DATASET / REPORT_DIR / DEBATE_LABEL / SC_LABEL / SAMPLES
# Gate is pinned to artifacts/classifier/pubmedqa_biolinkbert_seed47/best
# (overrides a DeBERTa path in .env without writing .env).
#
# After SC finishes
#   .venv/bin/python scripts/agents/insert_sc_row.py
#   # writes the ONE table cell from JSON; refuses if the report is missing
#
# Do not start n=500 from this wrapper. Resume is always on for debate/sc.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

DATASET="${DATASET:-data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json}"
REPORT_DIR="${REPORT_DIR:-reports/debate}"
DEBATE_LABEL="${DEBATE_LABEL:-debate_balanced90_ml4h_v1}"
SC_LABEL="${SC_LABEL:-sc_balanced90_ml4h_v1}"

# Paper-pinned BioLinkBERT gate (seed47). Overrides a DeBERTa path in .env
# if present; does not write .env.
export RAG_EVIDENCE_CLASSIFIER_MODEL_PATH="${RAG_EVIDENCE_CLASSIFIER_MODEL_PATH:-artifacts/classifier/pubmedqa_biolinkbert_seed47/best}"
export RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH="${RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH:-artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json}"
export RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1="${RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1:-0}"
export RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY="${RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY:-0}"

_load_ollama_env() {
  # Export only non-secret Ollama connection keys from .env when unset.
  # Never dump the rest of .env (API keys live there).
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
One-shot:
  ollama serve
  ollama pull ${OLLAMA_MODEL:-qwen2.5:7b}
  ollama list
  scripts/agents/run_ml4h_v1_arms.sh debate
Remote box:
  export OLLAMA_BASE_URL=http://<host>:11434
  # or: export OLLAMA_HOST=http://<host>:11434
  scripts/agents/run_ml4h_v1_arms.sh debate
EOF
}

preflight() {
  _load_ollama_env
  _resolve_ollama_url
  local model="${OLLAMA_MODEL:-qwen2.5:7b}"
  echo "Ollama host: ${OLLAMA_BASE_URL}  model: ${model}"
  if ! curl -fsS -m 5 "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
    echo "ERROR: $(_ollama_recipe)" >&2
    return 1
  fi
  if ! curl -fsS -m 5 "${OLLAMA_BASE_URL}/api/tags" | grep -q "qwen2.5"; then
    echo "ERROR: Ollama is up at ${OLLAMA_BASE_URL} but qwen2.5:* is not in /api/tags." >&2
    echo "  Next: ollama pull ${model}" >&2
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
  echo "Usage: scripts/agents/run_ml4h_v1_arms.sh <smoke|debate|sc|all|check>"
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
  echo "== debate balanced90 (usage-instrumented, --resume) =="
  "$PY" scripts/agents/evaluate_debate_pubmedqa.py \
    --dataset "$DATASET" \
    --backend ollama --hint biolinkbert --aggregate-with-biolinkbert \
    --aggregate-mode bert_gate --rounds 2 \
    --uncertainty-route --uncertainty-objective macro_f1 \
    --classifier-path "$RAG_EVIDENCE_CLASSIFIER_MODEL_PATH" \
    --classifier-temperature-path "$RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH" \
    --resume --label "$DEBATE_LABEL" --report-dir "$REPORT_DIR"
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
    echo "== SC with --samples ${SAMPLES:-8} (no measured debate cost yet) =="
    echo "   Re-run \`scripts/agents/run_ml4h_v1_arms.sh debate\` to write mean_llm_calls_per_case."
  fi
  "$PY" scripts/agents/evaluate_self_consistency_pubmedqa.py \
    --dataset "$DATASET" \
    --backend ollama --temperature 0.7 \
    --resume --label "$SC_LABEL" --report-dir "$REPORT_DIR" \
    "${extra[@]}"
}

check() {
  _load_ollama_env
  _resolve_ollama_url
  echo "dataset: $DATASET"
  echo "classifier: ${RAG_EVIDENCE_CLASSIFIER_MODEL_PATH}"
  echo "debate report: $REPORT_DIR/${DEBATE_LABEL}.json"
  echo "sc report:     $REPORT_DIR/${SC_LABEL}.json"
  if preflight; then
    :
  else
    echo "Ollama: DOWN (blocked). Start it, then re-run this check."
  fi
  debate_json="$REPORT_DIR/${DEBATE_LABEL}.json"
  if _debate_has_measured_cost "$debate_json"; then
    echo "debate cost: measured (ready for --match-cost-report)"
  elif [[ -f "$debate_json" ]]; then
    echo "debate cost: MISSING mean_llm_calls_per_case > 0 — re-run debate"
  else
    echo "debate cost: no report yet"
  fi
  if [[ -f "$REPORT_DIR/${SC_LABEL}.json" ]]; then
    echo "sc report: present — next: .venv/bin/python scripts/agents/insert_sc_row.py"
  else
    echo "sc report: not yet"
  fi
}

cmd="${1:-}"
case "$cmd" in
  smoke) smoke ;;
  debate) debate ;;
  sc) sc ;;
  all) smoke; debate; sc ;;
  check|preflight) check ;;
  *) usage; exit 1 ;;
esac
