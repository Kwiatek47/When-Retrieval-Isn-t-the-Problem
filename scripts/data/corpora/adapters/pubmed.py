"""Adapter for the PubMed reviews corpus.

Input: parquet produced by ``scripts/data/pubmed/pipeline/04_clean_dedupe_chunk.py``
with columns ``chunk_id, pmid, title, abstract, text, doi, journal, year,
publication_types, is_review, is_systematic``.  Each row is one PubMed
abstract-as-chunk.
"""

from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa

from scripts.data.corpora.adapters.base import BaseAdapter


class PubmedAdapter(BaseAdapter):
    def iter_rows(self, table: pa.Table) -> Iterator[dict[str, Any]]:
        pylist = table.to_pylist()
        for index, native in enumerate(pylist):
            pmid = _clean(native.get("pmid"))
            if not pmid:
                continue
            chunk_id = _clean(native.get("chunk_id")) or f"pubmed:{pmid}:abstract"
            text = _clean(native.get("text")) or _build_text(
                _clean(native.get("title")), _clean(native.get("abstract"))
            )
            if not text:
                continue
            doi = _clean(native.get("doi"))
            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": f"pubmed:{pmid}",
                "title": _clean(native.get("title")) or "Untitled PubMed abstract",
                "text": text,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "section": "abstract",
                "chunk_index": 0,
                "pmid": pmid,
                "doi": doi.lower() if doi else None,
                "journal": _clean(native.get("journal")),
                "year": _to_int(native.get("year")),
                "publication_types": _to_list(native.get("publication_types")),
                "is_review": _to_bool(native.get("is_review")),
                "is_systematic_review": _to_bool(native.get("is_systematic")),
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


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)]


def _build_text(title: str | None, abstract: str | None) -> str:
    if not title and not abstract:
        return ""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n\n".join(parts)
