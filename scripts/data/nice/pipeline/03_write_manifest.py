#!/usr/bin/env python3
"""Write NICE corpus manifest.json (preprocessing stage)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from common import INTERIM_DIR, repo_relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--dataset-name", default="nice_guidelines_v1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    interim = args.interim_dir.resolve()
    out = (args.out or interim / "manifest.json").resolve()

    manifest = {
        "dataset_name": args.dataset_name,
        "corpus": "nice",
        "source_type": "guideline",
        "source_name": "NICE",
        "created_at": date.today().isoformat(),
        "stage": "preprocessing",
        "schema_version": "documents_schema_v1",
        "files": {
            "documents": "documents.parquet",
            "chunks": "chunks.parquet",
            "chunks_clinical": "chunks_clinical.parquet",
            "markdown_raw_dir": "markdown/",
            "markdown_clean_dir": "markdown_clean/",
        },
        "paths": {
            "interim_dir": repo_relative(interim),
            "raw_pdf_dir": "data/raw/nice/pdf",
        },
        "parsing": {
            "default_parser": "pymupdf4llm",
            "output_format": "markdown",
        },
        "next_steps": [
            "use_markdown_clean_for_chunking",
            "merge_chunks_parquet_with_pubmed",
            "merge_documents_and_chunks_into_data_processed",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {repo_relative(out)}")


if __name__ == "__main__":
    main()
