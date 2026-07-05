"""Adapter for the StatPearls (NCBI Bookshelf) corpus.

Input: parquet produced by ``scripts/data/statpearls/build_chunks.py`` with
columns ``chunk_id, doc_id, nbk_id, title, section, section_index,
chunk_index, parent_chunk_id, text, url, year, word_count, text_hash``.
"""

from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa

from scripts.data.corpora.adapters.base import BaseAdapter


class StatPearlsAdapter(BaseAdapter):
    def iter_rows(self, table: pa.Table) -> Iterator[dict[str, Any]]:
        pylist = table.to_pylist()
        for native in pylist:
            chunk_id = _clean(native.get("chunk_id"))
            text = _clean(native.get("text"))
            if not chunk_id or not text:
                continue

            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": _clean(native.get("doc_id")) or chunk_id.rsplit(":", 1)[0],
                "title": _clean(native.get("title")) or "Untitled StatPearls chapter",
                "text": text,
                "url": _clean(native.get("url")),
                "section": _clean(native.get("section")),
                "chunk_index": _to_int(native.get("chunk_index")) or 0,
                "parent_chunk_id": _clean(native.get("parent_chunk_id")),
                "word_count": _to_int(native.get("word_count")),
                "text_hash": _clean(native.get("text_hash")),
                "journal": "StatPearls",
                "year": _to_int(native.get("year")),
                "external_id": _clean(native.get("nbk_id")),
                "publication_types": ["Clinical Overview"],
                "is_review": True,
                "is_systematic_review": False,
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
