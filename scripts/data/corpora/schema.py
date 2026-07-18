"""Canonical chunk schema shared by all corpus adapters.

Every adapter produces rows conforming to ``CanonicalChunk`` and every merged
``data/processed/chunks.parquet`` writer uses ``to_pyarrow_schema()`` so that
concatenation is order-preserving and does not rely on
``pa.concat_tables(promote_options="default")``.

Required fields are minimal but stable across corpora; optional fields are
nullable and typed once here so that per-corpus adapters only need to know
which subset applies to their source.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pyarrow as pa


SCHEMA_VERSION = "chunks_schema_v2"


@dataclass(frozen=True)
class CanonicalChunk:
    """A single chunk row after normalization by a corpus adapter."""

    chunk_id: str
    doc_id: str
    source: str
    source_type: str
    source_name: str
    title: str
    text: str
    text_hash: str
    word_count: int
    chunk_index: int
    corpus_version: str

    url: str | None = None
    section: str | None = None
    header_path: str | None = None
    parent_chunk_id: str | None = None
    parent_word_count: int | None = None

    pmid: str | None = None
    doi: str | None = None
    journal: str | None = None
    year: int | None = None
    publication_date: str | None = None
    publication_types: list[str] = field(default_factory=list)
    is_review: bool | None = None
    is_systematic_review: bool | None = None

    external_id: str | None = None
    guidance_type: str | None = None

    metadata: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COLUMN_ORDER: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "source",
    "source_type",
    "source_name",
    "title",
    "text",
    "text_hash",
    "word_count",
    "chunk_index",
    "corpus_version",
    "url",
    "section",
    "header_path",
    "parent_chunk_id",
    "parent_word_count",
    "pmid",
    "doi",
    "journal",
    "year",
    "publication_date",
    "publication_types",
    "is_review",
    "is_systematic_review",
    "external_id",
    "guidance_type",
    "metadata",
)


REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "doc_id",
        "source",
        "source_type",
        "source_name",
        "title",
        "text",
        "text_hash",
        "word_count",
        "chunk_index",
        "corpus_version",
    }
)


def to_pyarrow_schema() -> pa.Schema:
    """Fixed PyArrow schema used by all adapters and the merger."""

    fields: list[pa.Field] = [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_type", pa.string(), nullable=False),
        pa.field("source_name", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("text_hash", pa.string(), nullable=False),
        pa.field("word_count", pa.int64(), nullable=False),
        pa.field("chunk_index", pa.int64(), nullable=False),
        pa.field("corpus_version", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=True),
        pa.field("section", pa.string(), nullable=True),
        pa.field("header_path", pa.string(), nullable=True),
        pa.field("parent_chunk_id", pa.string(), nullable=True),
        pa.field("parent_word_count", pa.int64(), nullable=True),
        pa.field("pmid", pa.string(), nullable=True),
        pa.field("doi", pa.string(), nullable=True),
        pa.field("journal", pa.string(), nullable=True),
        pa.field("year", pa.int64(), nullable=True),
        pa.field("publication_date", pa.string(), nullable=True),
        pa.field("publication_types", pa.list_(pa.string()), nullable=True),
        pa.field("is_review", pa.bool_(), nullable=True),
        pa.field("is_systematic_review", pa.bool_(), nullable=True),
        pa.field("external_id", pa.string(), nullable=True),
        pa.field("guidance_type", pa.string(), nullable=True),
        pa.field("metadata", pa.string(), nullable=True),
    ]
    return pa.schema(fields)


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation used for cross-corpus title dedup."""

    if not title:
        return ""
    lowered = title.lower()
    lowered = re.sub(r"[^\w\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def rows_to_arrow_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a PyArrow table from canonical row dicts using the fixed schema.

    Missing optional keys are filled with ``None`` (or ``[]`` for
    ``publication_types``) before conversion so that the resulting table
    strictly matches ``to_pyarrow_schema()``.
    """

    schema = to_pyarrow_schema()
    if not rows:
        return schema.empty_table()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for name in COLUMN_ORDER:
            if name == "publication_types":
                value = row.get(name)
                item[name] = list(value) if value else None
            else:
                item[name] = row.get(name)
        normalized.append(item)

    return pa.Table.from_pylist(normalized, schema=schema)


def validate_row(row: dict[str, Any]) -> list[str]:
    """Return list of validation error strings (empty when the row is OK)."""

    errors: list[str] = []
    for column in REQUIRED_COLUMNS:
        value = row.get(column)
        if value is None:
            errors.append(f"missing required column: {column}")
            continue
        if column in {"chunk_id", "text", "title", "source", "source_type", "source_name", "text_hash", "corpus_version", "doc_id"}:
            if not isinstance(value, str) or not value.strip():
                errors.append(f"column {column} must be a non-empty string")
        elif column in {"word_count", "chunk_index"}:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"column {column} must be an int")
    return errors
