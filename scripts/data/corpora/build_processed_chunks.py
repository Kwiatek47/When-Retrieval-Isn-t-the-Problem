#!/usr/bin/env python3
"""Build the unified ``data/processed/chunks.parquet`` from registered corpora.

The script drives the whole merge:

1. read ``scripts/data/corpora/registry.json`` (or the file passed via
   ``--registry``);
2. select the requested corpus ids (default: every entry marked ``active`` or
   ``pilot`` with an existing ``local_chunks_path``);
3. run each adapter, get an Arrow table conforming to the canonical schema;
4. concatenate and run cross-corpus deduplication (pmid, doi, optionally
   title_hash) honouring per-entry ``dedupe_priority``;
5. validate uniqueness/non-empty invariants and write the merged parquet plus
   ``manifest.json`` next to it.

Unlike the previous ``scripts/data/merge_corpora.py`` / ``merge_parquet.py``
this script does not rely on ``concat_tables(promote_options="default")`` -
every adapter emits the same fixed PyArrow schema so concatenation is
deterministic and columns keep stable types.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.corpora.registry import (
    CorpusEntry,
    DEFAULT_REGISTRY_PATH,
    filter_registry,
    load_adapter,
    load_registry,
)
from scripts.data.corpora.schema import (
    SCHEMA_VERSION,
    normalize_title,
    to_pyarrow_schema,
)


DEFAULT_OUT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.json"


def main() -> int:
    args = _parse_args()

    registry = load_registry(args.registry)
    selected = _select_corpora(registry, args)
    if not selected:
        print("ERROR: no corpora selected (registry empty or all inputs missing).", file=sys.stderr)
        return 2

    tables: list[pa.Table] = []
    per_corpus_stats: list[dict[str, Any]] = []

    for entry in selected:
        print(f"[{entry.corpus_id}] reading {entry.local_chunks_path}", flush=True)
        adapter = load_adapter(entry)
        table = adapter.to_pyarrow_table()
        print(f"[{entry.corpus_id}] canonical rows: {table.num_rows}", flush=True)
        tables.append(table)
        per_corpus_stats.append(
            {
                "corpus_id": entry.corpus_id,
                "corpus_version": entry.corpus_version,
                "source": entry.source,
                "dedupe_priority": entry.dedupe_priority,
                "input_path": _rel(entry.local_chunks_path),
                "rows": table.num_rows,
            }
        )

    merged = pa.concat_tables(tables)
    print(f"pre-dedupe merged rows: {merged.num_rows}", flush=True)

    dedupe_stats: dict[str, int] = {"by_pmid": 0, "by_doi": 0, "by_title": 0}
    merged = _cross_corpus_dedupe(
        merged,
        selected,
        dedupe_titles=args.dedupe_titles,
        stats=dedupe_stats,
    )
    print(
        "post-dedupe merged rows: "
        f"{merged.num_rows} (removed pmid={dedupe_stats['by_pmid']}, "
        f"doi={dedupe_stats['by_doi']}, title={dedupe_stats['by_title']})",
        flush=True,
    )

    _validate(merged)

    args.out_chunks.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, args.out_chunks, compression="zstd")
    print(f"wrote {args.out_chunks} ({merged.num_rows} rows)", flush=True)

    _write_manifest(
        args.manifest_out,
        args.out_chunks,
        merged,
        per_corpus_stats,
        dedupe_stats,
        registry_path=args.registry,
    )
    print(f"wrote {args.manifest_out}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified data/processed/chunks.parquet from registered corpora.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to registry.json (default: scripts/data/corpora/registry.json).",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=None,
        help="Corpus ids to include (default: every registry entry whose local_chunks_path exists).",
    )
    parser.add_argument("--out-chunks", type=Path, default=DEFAULT_OUT_CHUNKS)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dedupe-titles",
        action="store_true",
        help="Enable optional cross-corpus dedupe on normalized title hash (in addition to pmid/doi).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="When --corpora is not given, silently skip registry entries whose local_chunks_path does not exist.",
    )
    return parser.parse_args()


def _select_corpora(registry: list[CorpusEntry], args: argparse.Namespace) -> list[CorpusEntry]:
    if args.corpora:
        return filter_registry(registry, args.corpora)

    selected: list[CorpusEntry] = []
    for entry in registry:
        if not entry.default_include:
            print(f"[skip] {entry.corpus_id}: default_include=false (pass --corpora to force)", flush=True)
            continue
        if entry.local_chunks_path.exists():
            selected.append(entry)
        else:
            msg = f"[skip] {entry.corpus_id}: local_chunks_path missing ({entry.local_chunks_path})"
            print(msg, flush=True)
    return selected


def _cross_corpus_dedupe(
    table: pa.Table,
    entries: list[CorpusEntry],
    *,
    dedupe_titles: bool,
    stats: dict[str, int],
) -> pa.Table:
    """Deduplicate rows across corpora.

    Priority: rows from corpora with higher ``dedupe_priority`` win.  Ties
    keep the first occurrence.  Duplicates that would fall inside a single
    corpus are left untouched (the merger already validates per-corpus chunk
    uniqueness); this pass only removes rows that clash across corpora.
    """

    if table.num_rows == 0:
        return table

    priority_by_source = {entry.source: entry.dedupe_priority for entry in entries}
    sources = table.column("source").to_pylist()
    pmids = table.column("pmid").to_pylist() if "pmid" in table.column_names else [None] * table.num_rows
    dois = table.column("doi").to_pylist() if "doi" in table.column_names else [None] * table.num_rows
    titles = table.column("title").to_pylist() if "title" in table.column_names else [None] * table.num_rows

    winner_for_pmid: dict[str, int] = {}
    winner_for_doi: dict[str, int] = {}
    winner_for_title: dict[str, int] = {}
    keep = [True] * table.num_rows

    def _prefer(current_index: int, new_index: int) -> int:
        current_priority = priority_by_source.get(sources[current_index], 0)
        new_priority = priority_by_source.get(sources[new_index], 0)
        return new_index if new_priority > current_priority else current_index

    for index in range(table.num_rows):
        pmid = _norm_key(pmids[index])
        if pmid:
            existing = winner_for_pmid.get(pmid)
            if existing is None:
                winner_for_pmid[pmid] = index
            else:
                winner = _prefer(existing, index)
                loser = existing if winner == index else index
                if sources[existing] != sources[index]:
                    keep[loser] = False
                    stats["by_pmid"] += 1
                winner_for_pmid[pmid] = winner

        doi = _norm_key(dois[index])
        if keep[index] and doi:
            existing = winner_for_doi.get(doi)
            if existing is None:
                winner_for_doi[doi] = index
            else:
                winner = _prefer(existing, index)
                loser = existing if winner == index else index
                if sources[existing] != sources[index]:
                    keep[loser] = False
                    stats["by_doi"] += 1
                winner_for_doi[doi] = winner

    if dedupe_titles:
        for index in range(table.num_rows):
            if not keep[index]:
                continue
            title_key = normalize_title(str(titles[index] or ""))
            if not title_key:
                continue
            existing = winner_for_title.get(title_key)
            if existing is None:
                winner_for_title[title_key] = index
            else:
                if sources[existing] == sources[index]:
                    continue
                winner = _prefer(existing, index)
                loser = existing if winner == index else index
                keep[loser] = False
                stats["by_title"] += 1
                winner_for_title[title_key] = winner

    if all(keep):
        return table

    mask = pa.array(keep, type=pa.bool_())
    return table.filter(mask)


def _norm_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _validate(table: pa.Table) -> None:
    expected_schema = to_pyarrow_schema()
    if set(table.column_names) != set(expected_schema.names):
        missing = set(expected_schema.names) - set(table.column_names)
        extra = set(table.column_names) - set(expected_schema.names)
        raise RuntimeError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    chunk_ids = table.column("chunk_id").to_pylist()
    empty = sum(1 for value in chunk_ids if not str(value or "").strip())
    duplicates = len(chunk_ids) - len(set(chunk_ids))
    texts = table.column("text").to_pylist()
    empty_texts = sum(1 for value in texts if not str(value or "").strip())
    if empty or duplicates or empty_texts:
        # Provide a compact preview of up to 5 duplicate ids for debugging.
        seen: dict[str, int] = defaultdict(int)
        for value in chunk_ids:
            seen[value] += 1
        dup_preview = [cid for cid, count in seen.items() if count > 1][:5]
        raise RuntimeError(
            "merged chunks failed validation: "
            f"empty_chunk_id={empty}, duplicate_chunk_id={duplicates} "
            f"(preview={dup_preview}), empty_text={empty_texts}"
        )


def _write_manifest(
    path: Path,
    out_chunks: Path,
    merged: pa.Table,
    inputs: list[dict[str, Any]],
    dedupe_stats: dict[str, int],
    *,
    registry_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts_by_source: dict[str, int] = defaultdict(int)
    for value in merged.column("source").to_pylist():
        counts_by_source[value] += 1
    counts_by_version: dict[str, int] = defaultdict(int)
    for value in merged.column("corpus_version").to_pylist():
        counts_by_version[value] += 1

    manifest = {
        "dataset_name": "medical_knowledge_chunks",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "chunks": _rel(out_chunks),
        "chunk_rows": merged.num_rows,
        "chunk_columns": merged.column_names,
        "registry": _rel(registry_path),
        "inputs": inputs,
        "dedupe_stats": dedupe_stats,
        "counts_by_source": dict(counts_by_source),
        "counts_by_corpus_version": dict(counts_by_version),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
