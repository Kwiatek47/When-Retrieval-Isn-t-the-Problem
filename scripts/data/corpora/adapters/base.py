"""Base adapter shared by all corpus-specific implementations.

Subclasses implement ``iter_rows`` (yielding canonical row dicts) and inherit
``to_pyarrow_table`` which reads the parquet input (single file or directory
of shards) and builds an Arrow table conforming to the canonical schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts.data.corpora.registry import CorpusEntry
from scripts.data.corpora.schema import rows_to_arrow_table, text_hash, word_count


class BaseAdapter(ABC):
    """Adapter contract: read a corpus-native parquet and emit canonical rows."""

    def __init__(self, entry: CorpusEntry) -> None:
        self.entry = entry

    @abstractmethod
    def iter_rows(self, table: pa.Table) -> Iterator[dict[str, Any]]:
        """Yield canonical row dicts from a corpus-native input table."""

    def load_table(self, path: Path | None = None) -> pa.Table:
        target = Path(path) if path else self.entry.local_chunks_path
        if not target.exists():
            raise FileNotFoundError(f"Corpus chunks path not found: {target}")
        if target.is_dir():
            dataset = ds.dataset(target, format="parquet")
            return dataset.to_table()
        return pq.read_table(target)

    def to_pyarrow_table(self, path: Path | None = None) -> pa.Table:
        native = self.load_table(path)
        rows = list(self.iter_rows(native))
        return rows_to_arrow_table(rows)

    def _finalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Fill common derived fields when the adapter did not compute them."""

        text = row.get("text") or ""
        if not row.get("text_hash"):
            row["text_hash"] = text_hash(text)
        if row.get("word_count") is None:
            row["word_count"] = word_count(text)
        row.setdefault("corpus_version", self.entry.corpus_version)
        row.setdefault("source", self.entry.source)
        row.setdefault("source_type", self.entry.source_type)
        row.setdefault("source_name", self.entry.source_name)
        return row


def _first_string(table: pa.Table, name: str, index: int) -> str | None:
    if name not in table.column_names:
        return None
    value = table.column(name)[index].as_py()
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_int(table: pa.Table, name: str, index: int) -> int | None:
    if name not in table.column_names:
        return None
    value = table.column(name)[index].as_py()
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_bool(table: pa.Table, name: str, index: int) -> bool | None:
    if name not in table.column_names:
        return None
    value = table.column(name)[index].as_py()
    if value is None:
        return None
    return bool(value)


def _first_list(table: pa.Table, name: str, index: int) -> list[str] | None:
    if name not in table.column_names:
        return None
    value = table.column(name)[index].as_py()
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]
