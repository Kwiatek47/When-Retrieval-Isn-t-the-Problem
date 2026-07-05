"""Adapter for the NICE guidelines corpus.

Input: parquet produced by ``scripts/embeddings/nice_markdown_to_parquet.py``
either in the broad or clinical variant, with columns roughly
``chunk_id, doc_id, document_id, source, source_type, source_name, title,
external_id, url, source_url, publication_date, guidance_type, section,
section_name, header_path, text, word_count, chunk_index, text_hash, metadata``.
"""

from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa

from scripts.data.corpora.adapters.base import BaseAdapter


class NiceAdapter(BaseAdapter):
    def iter_rows(self, table: pa.Table) -> Iterator[dict[str, Any]]:
        pylist = table.to_pylist()
        for native in pylist:
            chunk_id = _clean(native.get("chunk_id"))
            text = _clean(native.get("text"))
            if not chunk_id or not text:
                continue

            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": _clean(native.get("doc_id"))
                or _clean(native.get("document_id"))
                or chunk_id.split(":", 1)[0],
                "title": _clean(native.get("title")) or "Untitled NICE guideline",
                "text": text,
                "url": _clean(native.get("url")) or _clean(native.get("source_url")),
                "section": _clean(native.get("section")) or _clean(native.get("section_name")),
                "header_path": _clean(native.get("header_path")),
                "chunk_index": _to_int(native.get("chunk_index")) or 0,
                "word_count": _to_int(native.get("word_count")),
                "text_hash": _clean(native.get("text_hash")),
                "external_id": _clean(native.get("external_id")),
                "guidance_type": _clean(native.get("guidance_type")),
                "publication_date": _clean(native.get("publication_date")),
                "publication_types": _publication_types(native.get("guidance_type")),
                "metadata": _clean(native.get("metadata")),
            }
            yield self._finalize_row(row)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _publication_types(guidance_type: Any) -> list[str]:
    """Map NICE guidance_type into RAG publication_types shared with PubMed policy.

    NICE guidelines are Practice Guidelines / Guidelines regardless of the
    per-document guidance_type sub-classification.
    """

    if not guidance_type:
        return ["Practice Guideline", "Guideline"]
    return ["Practice Guideline", "Guideline"]
