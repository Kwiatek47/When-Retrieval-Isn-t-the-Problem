#!/usr/bin/env python3
"""Merge local corpus chunk parquet files into one processed chunks.parquet."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.json"

COMMON_CHUNK_COLUMNS = [
    "chunk_id",
    "doc_id",
    "document_id",
    "source",
    "source_type",
    "source_name",
    "title",
    "text",
    "url",
    "source_url",
    "external_id",
    "pmid",
    "doi",
    "journal",
    "year",
    "publication_date",
    "publication_types",
    "guidance_type",
    "section",
    "section_name",
    "header_path",
    "is_review",
    "is_systematic_review",
    "word_count",
    "parent_chunk_id",
    "parent_word_count",
    "text_hash",
    "chunk_index",
    "metadata",
]


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.chunks)
    if not inputs:
        raise RuntimeError("Provide at least one --chunks corpus=path entry.")

    tables: list[pa.Table] = []
    manifest_inputs: list[dict[str, Any]] = []

    for corpus, path in inputs:
        table = read_chunk_table(path, corpus)
        validate_chunk_table(table, corpus, path)
        tables.append(table)
        manifest_inputs.append(
            {
                "corpus": corpus,
                "path": repo_relative_or_abs(path),
                "rows": table.num_rows,
                "columns": table.column_names,
            }
        )

    merged = pa.concat_tables(tables, promote_options="default")
    merged = order_columns(merged)
    validate_unique_chunk_ids(merged)

    args.out_chunks.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, args.out_chunks, compression="zstd")
    write_manifest(args.manifest_out, args.out_chunks, merged, manifest_inputs)

    print(f"Merged chunks: {merged.num_rows}")
    print(f"Saved chunks: {repo_relative_or_abs(args.out_chunks)}")
    print(f"Saved manifest: {repo_relative_or_abs(args.manifest_out)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge corpus chunks parquet files.")
    parser.add_argument(
        "--chunks",
        nargs="+",
        required=True,
        help="Corpus chunk parquet entries as corpus=path, e.g. pubmed=/mnt/pubmed/chunks.parquet nice=data/interim/nice/chunks.parquet",
    )
    parser.add_argument("--out-chunks", type=Path, default=DEFAULT_OUT_CHUNKS)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected corpus=path, got: {value}")
        corpus, raw_path = value.split("=", 1)
        corpus = corpus.strip().lower()
        path = Path(raw_path).expanduser().resolve()
        if not corpus:
            raise ValueError(f"Empty corpus name in: {value}")
        if not path.exists():
            raise FileNotFoundError(path)
        inputs.append((corpus, path))
    return inputs


def read_chunk_table(path: Path, corpus: str) -> pa.Table:
    table = pq.read_table(path)
    if "source" not in table.column_names:
        table = table.append_column("source", pa.array([corpus] * table.num_rows, pa.string()))
    return table


def validate_chunk_table(table: pa.Table, corpus: str, path: Path) -> None:
    missing = {"chunk_id", "text"} - set(table.column_names)
    if missing:
        raise RuntimeError(f"{path} ({corpus}) is missing required columns: {sorted(missing)}")

    chunk_ids = table.column("chunk_id").to_pylist()
    texts = table.column("text").to_pylist()
    empty_ids = sum(1 for value in chunk_ids if not str(value or "").strip())
    empty_texts = sum(1 for value in texts if not str(value or "").strip())
    duplicate_ids = len(chunk_ids) - len(set(chunk_ids))
    if empty_ids or empty_texts or duplicate_ids:
        raise RuntimeError(
            f"{path} ({corpus}) failed validation: "
            f"empty_chunk_id={empty_ids}, empty_text={empty_texts}, duplicate_chunk_id={duplicate_ids}"
        )


def order_columns(table: pa.Table) -> pa.Table:
    ordered = [column for column in COMMON_CHUNK_COLUMNS if column in table.column_names]
    remaining = [column for column in table.column_names if column not in ordered]
    return table.select(ordered + remaining)


def validate_unique_chunk_ids(table: pa.Table) -> None:
    chunk_ids = table.column("chunk_id").to_pylist()
    duplicate_ids = len(chunk_ids) - len(set(chunk_ids))
    if duplicate_ids:
        raise RuntimeError(f"Merged chunks contain {duplicate_ids} duplicate chunk_id values.")


def write_manifest(
    path: Path,
    out_chunks: Path,
    merged: pa.Table,
    inputs: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": "medical_knowledge_chunks",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "processed_merge",
        "chunks": repo_relative_or_abs(out_chunks),
        "chunk_rows": merged.num_rows,
        "chunk_columns": merged.column_names,
        "inputs": inputs,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def repo_relative_or_abs(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
