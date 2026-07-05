"""Unit tests for the merger's cross-corpus dedupe logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.corpora.build_processed_chunks import _cross_corpus_dedupe, _validate
from scripts.data.corpora.registry import CorpusEntry
from scripts.data.corpora.schema import to_pyarrow_schema


def _entry(corpus_id: str, source: str, priority: int) -> CorpusEntry:
    return CorpusEntry(
        corpus_id=corpus_id,
        corpus_version=f"{corpus_id}-v1",
        source=source,
        source_type="literature",
        source_name=source.upper(),
        adapter_class="",
        priority="P0",
        status="active",
        dedupe_priority=priority,
        default_include=True,
        local_chunks_path=Path("/tmp/none"),
        documents_path=None,
        manifest_path=None,
        publication_types=[],
        notes="",
        raw={},
    )


def _make_table(rows: list[dict]) -> pa.Table:
    schema = to_pyarrow_schema()
    normalized = []
    for row in rows:
        item = {name: row.get(name) for name in schema.names}
        item.setdefault("word_count", 1)
        item.setdefault("chunk_index", 0)
        item.setdefault("text_hash", "hash")
        item.setdefault("title", "title")
        item.setdefault("text", "text")
        item.setdefault("source", "pubmed")
        item.setdefault("source_type", "literature")
        item.setdefault("source_name", "PubMed")
        item.setdefault("corpus_version", "v1")
        item.setdefault("doc_id", "doc")
        if item.get("publication_types") is None:
            item["publication_types"] = []
        normalized.append(item)
    return pa.Table.from_pylist(normalized, schema=schema)


class CrossCorpusDedupeTests(unittest.TestCase):
    def test_higher_priority_wins_on_pmid_collision(self) -> None:
        entries = [_entry("pubmed_reviews_v1", "pubmed", 100), _entry("preprints_v1", "preprints", 50)]
        table = _make_table(
            [
                {"chunk_id": "a", "pmid": "1", "source": "preprints"},
                {"chunk_id": "b", "pmid": "1", "source": "pubmed"},
            ]
        )
        stats: dict[str, int] = {"by_pmid": 0, "by_doi": 0, "by_title": 0}
        result = _cross_corpus_dedupe(table, entries, dedupe_titles=False, stats=stats)
        self.assertEqual(result.num_rows, 1)
        self.assertEqual(result.column("chunk_id")[0].as_py(), "b")
        self.assertEqual(stats["by_pmid"], 1)

    def test_no_dedupe_within_same_corpus(self) -> None:
        entries = [_entry("pubmed_reviews_v1", "pubmed", 100)]
        table = _make_table(
            [
                {"chunk_id": "a", "pmid": "1", "source": "pubmed"},
                {"chunk_id": "b", "pmid": "1", "source": "pubmed"},
            ]
        )
        stats: dict[str, int] = {"by_pmid": 0, "by_doi": 0, "by_title": 0}
        result = _cross_corpus_dedupe(table, entries, dedupe_titles=False, stats=stats)
        self.assertEqual(result.num_rows, 2)
        self.assertEqual(stats["by_pmid"], 0)

    def test_doi_is_normalized_lowercase(self) -> None:
        entries = [_entry("pubmed_reviews_v1", "pubmed", 100), _entry("nice_v1", "nice", 90)]
        table = _make_table(
            [
                {"chunk_id": "a", "doi": "10.0000/EXAMPLE", "source": "nice"},
                {"chunk_id": "b", "doi": "10.0000/example", "source": "pubmed"},
            ]
        )
        stats: dict[str, int] = {"by_pmid": 0, "by_doi": 0, "by_title": 0}
        result = _cross_corpus_dedupe(table, entries, dedupe_titles=False, stats=stats)
        self.assertEqual(result.num_rows, 1)
        self.assertEqual(result.column("chunk_id")[0].as_py(), "b")
        self.assertEqual(stats["by_doi"], 1)

    def test_title_dedupe_only_when_flag_set(self) -> None:
        entries = [_entry("pubmed_reviews_v1", "pubmed", 100), _entry("statpearls_v1", "statpearls", 80)]
        table = _make_table(
            [
                {"chunk_id": "a", "title": "Acetaminophen Overview", "source": "statpearls"},
                {"chunk_id": "b", "title": "acetaminophen overview.", "source": "pubmed"},
            ]
        )
        stats: dict[str, int] = {"by_pmid": 0, "by_doi": 0, "by_title": 0}

        result = _cross_corpus_dedupe(table, entries, dedupe_titles=False, stats=stats)
        self.assertEqual(result.num_rows, 2)

        stats = {"by_pmid": 0, "by_doi": 0, "by_title": 0}
        result = _cross_corpus_dedupe(table, entries, dedupe_titles=True, stats=stats)
        self.assertEqual(result.num_rows, 1)
        self.assertEqual(result.column("chunk_id")[0].as_py(), "b")
        self.assertEqual(stats["by_title"], 1)


class ValidateTests(unittest.TestCase):
    def test_raises_on_duplicate_chunk_id(self) -> None:
        table = _make_table(
            [
                {"chunk_id": "same", "source": "pubmed"},
                {"chunk_id": "same", "source": "pubmed"},
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate_chunk_id"):
            _validate(table)

    def test_raises_on_empty_text(self) -> None:
        table = _make_table(
            [
                {"chunk_id": "a", "text": " ", "source": "pubmed"},
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "empty_text"):
            _validate(table)


if __name__ == "__main__":
    unittest.main()
