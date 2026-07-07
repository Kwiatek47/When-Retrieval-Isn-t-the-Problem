#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

exec scripts/classifier/run_pubmedqa_research_experiments.sh
