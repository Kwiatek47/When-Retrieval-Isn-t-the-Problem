#!/usr/bin/env bash
# Run every debate architecture on the same cases with the same model.
#
# The arms only mean anything next to each other, so this script exists to make
# sure they are produced identically: same dataset, same corpus, same rounds,
# same seed, one model. Do not run the arms by hand with different flags and
# then compare the numbers.
#
#   scripts/agents/run_paper_b_arms.sh --trial          # 50 cases, quick check
#   scripts/agents/run_paper_b_arms.sh --full           # all 500, the real run
#   scripts/agents/run_paper_b_arms.sh --trial --mock   # no LLM, plumbing only
#
# Every arm resumes from its checkpoint, so an interrupted run continues where
# it stopped rather than starting over.
set -euo pipefail

cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
EVAL=scripts/agents/evaluate_debate_pubmedqa.py
REPORT_DIR=${REPORT_DIR:-reports/paper_b}

QUICK_DATASET=data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json
FULL_DATASET=data/benchmarks/pubmedqa/official_pqal_test/eval.json
CORPUS=data/benchmarks/pubmedqa/official_pqal_test/corpus.json

BACKEND=ollama
DATASET=$QUICK_DATASET
LIMIT_ARGS=(--limit 50)
SUFFIX=trial

# Frozen across all arms; changing one of these invalidates the comparison.
ROUNDS=${ROUNDS:-2}
PARTITIONS=${PARTITIONS:-4}
PARTITION_OVERLAP=${PARTITION_OVERLAP:-1}
MAX_INFO_ROUNDS=${MAX_INFO_ROUNDS:-1}
SEED=${SEED:-47}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trial) DATASET=$QUICK_DATASET; LIMIT_ARGS=(--limit 50); SUFFIX=trial ;;
    --full)  DATASET=$FULL_DATASET;  LIMIT_ARGS=();           SUFFIX=full  ;;
    --mock)  BACKEND=mock;                                    SUFFIX=mock  ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$REPORT_DIR"

echo "backend=$BACKEND dataset=$DATASET rounds=$ROUNDS seed=$SEED -> $REPORT_DIR"
echo

for ARCH in round_robin neutral supervisor anonymized asymmetric; do
  LABEL="paper_b_${ARCH}_${SUFFIX}"
  echo "=== $ARCH -> $LABEL ==="
  # shellcheck disable=SC2086
  "$PY" "$EVAL" \
    --dataset "$DATASET" \
    --corpus "$CORPUS" \
    --backend "$BACKEND" \
    --architecture "$ARCH" \
    --rounds "$ROUNDS" \
    --partitions "$PARTITIONS" \
    --partition-overlap "$PARTITION_OVERLAP" \
    --max-info-rounds "$MAX_INFO_ROUNDS" \
    --uncertainty-seed "$SEED" \
    --agent-concurrency 1 \
    --report-dir "$REPORT_DIR" \
    --resume \
    --label "$LABEL" \
    "${LIMIT_ARGS[@]}"
  echo
done

echo "All arms finished. Build the comparison tables with:"
echo "  $PY scripts/agents/build_paper_b_tables.py --report-dir $REPORT_DIR --suffix $SUFFIX"
