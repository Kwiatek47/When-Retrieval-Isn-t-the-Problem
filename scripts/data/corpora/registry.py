"""Read-only accessor for ``scripts/data/corpora/registry.json``.

The registry is the single source of truth for which corpora exist, where their
local chunk parquet files live and which adapter builds canonical rows for
them.  The merger consumes this file rather than a bespoke list of arguments.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "scripts" / "data" / "corpora" / "registry.json"


@dataclass(frozen=True)
class CorpusEntry:
    corpus_id: str
    corpus_version: str
    source: str
    source_type: str
    source_name: str
    adapter_class: str
    priority: str
    status: str
    dedupe_priority: int
    default_include: bool
    local_chunks_path: Path
    documents_path: Path | None
    manifest_path: Path | None
    publication_types: list[str]
    notes: str
    raw: dict[str, Any]


def load_registry(path: Path | None = None) -> list[CorpusEntry]:
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    entries: list[CorpusEntry] = []
    for row in data.get("corpora", []):
        entries.append(
            CorpusEntry(
                corpus_id=row["corpus_id"],
                corpus_version=row["corpus_version"],
                source=row["source"],
                source_type=row["source_type"],
                source_name=row["source_name"],
                adapter_class=row["adapter_class"],
                priority=row.get("priority", ""),
                status=row.get("status", ""),
                dedupe_priority=int(row.get("dedupe_priority", 0)),
                default_include=bool(row.get("default_include", True)),
                local_chunks_path=_resolve(row.get("local_chunks_path")),
                documents_path=_optional_resolve(row.get("documents_path")),
                manifest_path=_optional_resolve(row.get("manifest_path")),
                publication_types=list(row.get("publication_types", [])),
                notes=row.get("notes", ""),
                raw=row,
            )
        )
    return entries


def filter_registry(entries: list[CorpusEntry], corpus_ids: list[str] | None) -> list[CorpusEntry]:
    if not corpus_ids:
        return list(entries)
    known = {entry.corpus_id: entry for entry in entries}
    missing = [cid for cid in corpus_ids if cid not in known]
    if missing:
        raise KeyError(f"Unknown corpus_id(s) in registry: {missing}")
    return [known[cid] for cid in corpus_ids]


def load_adapter(entry: CorpusEntry) -> Any:
    """Import ``adapter_class`` and instantiate it with the entry."""

    module_name, class_name = entry.adapter_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(entry)


def _resolve(value: str | None) -> Path:
    if not value:
        raise ValueError("local_chunks_path is required in registry.json entries")
    return _resolve_path(value)


def _optional_resolve(value: str | None) -> Path | None:
    if not value:
        return None
    return _resolve_path(value)


def _resolve_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
