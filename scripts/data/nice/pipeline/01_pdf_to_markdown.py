#!/usr/bin/env python3
"""Parse NICE PDFs to markdown (preprocessing stage — no chunking)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from common import INTERIM_DIR, INTERIM_MARKDOWN_DIR, RAW_PDF_DIR, repo_relative

FAILURES_LOG = INTERIM_DIR / "markdown_failures.jsonl"


def parse_pdf(pdf_path: Path, output_path: Path, parser: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if parser == "pymupdf4llm":
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(str(pdf_path), use_ocr=False)
    else:
        raise ValueError(f"Unsupported parser: {parser}")

    output_path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NICE PDFs to markdown in data/interim/nice/markdown/")
    parser.add_argument("--pdf-dir", type=Path, default=RAW_PDF_DIR, help="Directory with source PDF files")
    parser.add_argument("--out-dir", type=Path, default=INTERIM_MARKDOWN_DIR, help="Output markdown directory")
    parser.add_argument("--parser", default="pymupdf4llm", choices=["pymupdf4llm"])
    parser.add_argument("--document-id", default="", help="Override output stem (default: PDF stem)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if markdown already exists")
    args = parser.parse_args()

    pdf_dir = args.pdf_dir.resolve()
    out_dir = args.out_dir.resolve()
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {pdf_dir}")

    ok = 0
    skipped = 0
    failed = 0
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)

    for i, pdf_path in enumerate(pdfs, start=1):
        stem = args.document_id or pdf_path.stem
        out_path = out_dir / f"{stem}.md"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        try:
            parse_pdf(pdf_path, out_path, args.parser)
            ok += 1
            if i % 50 == 0 or i == len(pdfs):
                print(f"[{i}/{len(pdfs)}] ok={ok} skipped={skipped} failed={failed}", flush=True)
        except Exception as exc:
            failed += 1
            with FAILURES_LOG.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "pdf": repo_relative(pdf_path),
                            "error": str(exc),
                            "traceback": traceback.format_exc()[-2000:],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"FAILED {pdf_path.name}: {exc}", file=sys.stderr, flush=True)

    print(f"Done. ok={ok} skipped={skipped} failed={failed} failures_log={repo_relative(FAILURES_LOG)}")


if __name__ == "__main__":
    main()
