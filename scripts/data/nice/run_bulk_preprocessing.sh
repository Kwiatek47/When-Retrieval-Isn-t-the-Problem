#!/usr/bin/env bash
# Full NICE preprocessing: catalog -> all PDFs -> markdown -> documents.parquet
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PIPE="${ROOT}/scripts/data/nice/pipeline"

cd "$ROOT"

echo "=== 1/6 Published guidance catalog (~2500+ entries) ==="
"$PY" "$PIPE/00_fetch_published_catalog.py"

echo "=== 2/6 Download all available PDFs (may take 1-2h; resumable with --skip-existing) ==="
"$PY" "$PIPE/00_download_pdfs.py" --all-published --skip-existing --delay 0.5

echo "=== 3/6 PDF -> Markdown ==="
"$PY" "$PIPE/01_pdf_to_markdown.py" --skip-existing

echo "=== 4/6 Clean boilerplate -> markdown_clean ==="
"$PY" "$PIPE/01b_clean_markdown.py" --skip-existing

echo "=== 5/6 documents.parquet ==="
"$PY" "$PIPE/02_build_documents.py"

echo "=== 6/6 manifest.json ==="
"$PY" "$PIPE/03_write_manifest.py"

echo "Done. See data/interim/nice/pdf_download_report.json for download stats."
