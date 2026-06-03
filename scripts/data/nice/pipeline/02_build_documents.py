#!/usr/bin/env python3
"""Build document-level registry (documents.parquet) for NICE preprocessing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    INTERIM_DIR,
    INTERIM_MARKDOWN_CLEAN_DIR,
    INTERIM_MARKDOWN_DIR,
    RAW_PDF_DIR,
    default_metadata_extra,
    dumps_metadata,
    extract_metadata_from_markdown,
    repo_relative,
    slugify_nice_id,
    utc_now_iso,
)

DEFAULT_CATALOG = INTERIM_DIR / "guidance_catalog.jsonl"


def pdf_page_count(pdf_path: Path) -> int:
    import fitz  # pymupdf

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            guidance_id = str(row.get("guidance_id") or row.get("guidance_ref") or "").strip().lower()
            if guidance_id:
                lookup[guidance_id] = row
    return lookup


def normalize_guidance_types(values: list[str]) -> str:
    normalized = []
    for value in values:
        token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        if token and token not in normalized:
            normalized.append(token)
    return ";".join(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register parsed NICE markdown files in documents.parquet")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=None,
        help="Default: markdown_clean/ if present, else markdown/",
    )
    parser.add_argument("--pdf-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--out", type=Path, default=INTERIM_DIR / "documents.parquet")
    parser.add_argument("--parser", default="pymupdf4llm")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--list-documents", action="store_true", help="Print every registered document id")
    args = parser.parse_args()

    markdown_dir = args.markdown_dir
    if markdown_dir is None:
        markdown_dir = (
            INTERIM_MARKDOWN_CLEAN_DIR
            if INTERIM_MARKDOWN_CLEAN_DIR.exists() and any(INTERIM_MARKDOWN_CLEAN_DIR.glob("*.md"))
            else INTERIM_MARKDOWN_DIR
        )
    markdown_dir = markdown_dir.resolve()

    md_files = sorted(markdown_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files in {markdown_dir}")

    try:
        parser_version = importlib.metadata.version("pymupdf4llm")
    except importlib.metadata.PackageNotFoundError:
        parser_version = ""

    ingested_at = utc_now_iso()
    rows: list[dict] = []
    catalog = load_catalog(args.catalog.resolve())

    for md_path in md_files:
        md_text = md_path.read_text(encoding="utf-8")
        meta = extract_metadata_from_markdown(md_text, fallback_stem=md_path.stem)
        meta["external_id"] = md_path.stem.upper()
        meta["document_id"] = slugify_nice_id(meta["external_id"])
        meta["source_url"] = f"https://www.nice.org.uk/guidance/{md_path.stem.lower()}"
        catalog_row = catalog.get(md_path.stem.lower()) or catalog.get(str(meta["external_id"]).lower())
        if catalog_row:
            meta["external_id"] = catalog_row.get("guidance_ref") or meta["external_id"]
            meta["document_id"] = slugify_nice_id(meta["external_id"])
            meta["title"] = catalog_row.get("title") or meta["title"]
            meta["source_url"] = catalog_row.get("url") or meta["source_url"]
            meta["published_at"] = catalog_row.get("publication_date") or meta["published_at"]
            meta["guidance_type"] = normalize_guidance_types(catalog_row.get("guidance_types") or []) or meta["guidance_type"]
        pdf_path = args.pdf_dir / f"{md_path.stem}.pdf"
        if not pdf_path.exists():
            # allow document.pdf paired with ta1158.md via same stem only
            candidates = list(args.pdf_dir.glob("*.pdf"))
            pdf_path = candidates[0] if len(candidates) == 1 else pdf_path

        page_count = pdf_page_count(pdf_path) if pdf_path.exists() else None
        extra = default_metadata_extra(md_text)
        if catalog_row:
            extra["catalog_guidance_types"] = catalog_row.get("guidance_types") or []
            extra["catalog_path_and_query"] = catalog_row.get("path_and_query") or ""

        rows.append(
            {
                "document_id": meta["document_id"],
                "source_type": "guideline",
                "source_name": "NICE",
                "title": meta["title"],
                "external_id": meta["external_id"],
                "guidance_type": meta["guidance_type"],
                "source_url": meta["source_url"],
                "published_at": meta["published_at"],
                "language": meta["language"],
                "pdf_relpath": repo_relative(pdf_path) if pdf_path.exists() else "",
                "markdown_relpath": repo_relative(md_path),
                "parser_name": args.parser,
                "parser_version": parser_version,
                "page_count": page_count,
                "markdown_chars": len(md_text),
                "ingested_at": ingested_at,
                "metadata": dumps_metadata(extra),
            }
        )

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"Registered {len(df)} document(s) -> {repo_relative(out)}")
    if args.list_documents:
        for _, row in df.iterrows():
            print(f"  {row['document_id']} ({row['external_id']})")


if __name__ == "__main__":
    main()
