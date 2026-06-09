#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRE_H100="${REQUIRE_H100:-1}"
INSTALL_QDRANT="${INSTALL_QDRANT:-1}"
QDRANT_VERSION="${QDRANT_VERSION:-v1.14.1}"
QDRANT_BIN_DIR="${QDRANT_BIN_DIR:-${HOME}/bin}"

echo "==> Installing Python dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r benchmarks/requirements.txt -r requirements.txt

echo "==> Runtime check"
if [[ "${REQUIRE_H100}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/colab/check_h100_runtime.py --require-h100
else
  "${PYTHON_BIN}" scripts/colab/check_h100_runtime.py --no-require-h100
fi

if [[ "${INSTALL_QDRANT}" == "1" ]] && ! command -v qdrant >/dev/null 2>&1 && [[ ! -x "${QDRANT_BIN_DIR}/qdrant" ]]; then
  echo "==> Installing Qdrant ${QDRANT_VERSION} to ${QDRANT_BIN_DIR}/qdrant"
  mkdir -p "${QDRANT_BIN_DIR}"
  tmp_dir="$(mktemp -d)"
  curl -L "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" \
    -o "${tmp_dir}/qdrant.tar.gz"
  tar -xzf "${tmp_dir}/qdrant.tar.gz" -C "${tmp_dir}"
  install -m 0755 "${tmp_dir}/qdrant" "${QDRANT_BIN_DIR}/qdrant"
  rm -rf "${tmp_dir}"
fi

mkdir -p /content/hf-cache /content/chatbot-med-data /content/chatbot-med-runs

cat <<EOF
Setup finished.

Before running the full benchmark, make sure this file exists:
  data/processed/chunks.parquet

If you uploaded it elsewhere, copy it into place:
  mkdir -p data/processed
  cp /content/chatbot-med-data/chunks.parquet data/processed/chunks.parquet
EOF
