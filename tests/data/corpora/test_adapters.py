"""Unit tests for per-corpus canonical-schema adapters.

Each adapter test constructs a small PyArrow table shaped like the native
per-corpus parquet, feeds it through the adapter and asserts that:

- required canonical fields are present and non-empty,
- corpus-specific mapping (chunk_id, source, publication_types, etc.) works,
- the final PyArrow table conforms to ``to_pyarrow_schema()``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.corpora.registry import CorpusEntry
from scripts.data.corpora.schema import to_pyarrow_schema


def _entry(**overrides):
    defaults = dict(
        corpus_id="test_corpus",
        corpus_version="test-v1",
        source="pubmed",
        source_type="literature",
        source_name="PubMed",
        adapter_class="",
        priority="P0",
        status="active",
        dedupe_priority=100,
        default_include=True,
        local_chunks_path=Path("/tmp/does-not-exist.parquet"),
        documents_path=None,
        manifest_path=None,
        publication_types=[],
        notes="",
        raw={},
    )
    defaults.update(overrides)
    return CorpusEntry(**defaults)


class PubmedAdapterTests(unittest.TestCase):
    def test_maps_native_pubmed_row_to_canonical(self) -> None:
        from scripts.data.corpora.adapters.pubmed import PubmedAdapter

        native = pa.table(
            {
                "chunk_id": ["pubmed:12345:abstract"],
                "pmid": ["12345"],
                "title": ["Example review"],
                "abstract": ["Example abstract with enough words."],
                "text": [
                    "Title: Example review\n\nAbstract: Example abstract with enough words."
                ],
                "doi": ["10.0000/Example"],
                "journal": ["Example Journal"],
                "year": [2024],
                "publication_types": [["Review"]],
                "is_review": [True],
                "is_systematic": [False],
            }
        )
        adapter = PubmedAdapter(_entry(source="pubmed", source_type="literature", source_name="PubMed"))
        table = pa.Table.from_pylist(list(adapter.iter_rows(native)), schema=to_pyarrow_schema())

        self.assertEqual(table.num_rows, 1)
        row = table.to_pylist()[0]
        self.assertEqual(row["chunk_id"], "pubmed:12345:abstract")
        self.assertEqual(row["doc_id"], "pubmed:12345")
        self.assertEqual(row["source"], "pubmed")
        self.assertEqual(row["corpus_version"], "test-v1")
        self.assertEqual(row["url"], "https://pubmed.ncbi.nlm.nih.gov/12345/")
        self.assertEqual(row["doi"], "10.0000/example")
        self.assertEqual(row["publication_types"], ["Review"])
        self.assertTrue(row["is_review"])
        self.assertGreater(row["word_count"], 0)
        self.assertEqual(len(row["text_hash"]), 64)

    def test_skips_rows_without_pmid(self) -> None:
        from scripts.data.corpora.adapters.pubmed import PubmedAdapter

        native = pa.table(
            {
                "pmid": [None, "9"],
                "title": ["x", "y"],
                "abstract": ["a", "b"],
                "text": ["Title: x\n\nAbstract: a", "Title: y\n\nAbstract: b"],
                "doi": [None, None],
                "journal": [None, None],
                "year": [None, None],
                "publication_types": [[], []],
                "is_review": [False, False],
                "is_systematic": [False, False],
            }
        )
        adapter = PubmedAdapter(_entry())
        rows = list(adapter.iter_rows(native))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pmid"], "9")


class NiceAdapterTests(unittest.TestCase):
    def test_maps_native_nice_row_to_canonical(self) -> None:
        from scripts.data.corpora.adapters.nice import NiceAdapter

        native = pa.table(
            {
                "chunk_id": ["nice-amr1:chunk:0000"],
                "doc_id": ["nice-amr1"],
                "document_id": ["nice-amr1"],
                "title": ["Ceftazidime with avibactam"],
                "external_id": ["AMR1"],
                "url": ["https://www.nice.org.uk/guidance/amr1"],
                "publication_date": ["2022-08-17"],
                "guidance_type": ["antimicrobial_prescribing_guidelines"],
                "section": ["1.1"],
                "header_path": ["Guidance > 1.1"],
                "text": ["Title: X\n\n1.1 Recommendation body of the guideline."],
                "word_count": [42],
                "chunk_index": [0],
                "text_hash": ["deadbeef"],
                "metadata": ['{"external_id": "AMR1"}'],
            }
        )
        adapter = NiceAdapter(
            _entry(source="nice", source_type="guideline", source_name="NICE")
        )
        table = pa.Table.from_pylist(list(adapter.iter_rows(native)), schema=to_pyarrow_schema())

        self.assertEqual(table.num_rows, 1)
        row = table.to_pylist()[0]
        self.assertEqual(row["chunk_id"], "nice-amr1:chunk:0000")
        self.assertEqual(row["source"], "nice")
        self.assertEqual(row["external_id"], "AMR1")
        self.assertEqual(row["publication_types"], ["Practice Guideline", "Guideline"])
        self.assertEqual(row["guidance_type"], "antimicrobial_prescribing_guidelines")
        self.assertEqual(row["header_path"], "Guidance > 1.1")


class StatPearlsAdapterTests(unittest.TestCase):
    def test_maps_native_statpearls_row_to_canonical(self) -> None:
        from scripts.data.corpora.adapters.statpearls import StatPearlsAdapter

        native = pa.table(
            {
                "chunk_id": ["statpearls:nbk430685:s0:introduction:0"],
                "doc_id": ["statpearls:nbk430685"],
                "nbk_id": ["NBK430685"],
                "title": ["Acetaminophen"],
                "section": ["Introduction"],
                "section_index": [0],
                "chunk_index": [0],
                "parent_chunk_id": [None],
                "text": [
                    "Title: Acetaminophen\n\nSection: Introduction\n\nBody paragraph."
                ],
                "url": ["https://www.ncbi.nlm.nih.gov/books/NBK430685/"],
                "year": [2024],
                "word_count": [12],
                "text_hash": ["cafef00d"],
            }
        )
        adapter = StatPearlsAdapter(
            _entry(
                source="statpearls",
                source_type="clinical_overview",
                source_name="StatPearls",
                corpus_version="statpearls-v1",
            )
        )
        table = pa.Table.from_pylist(list(adapter.iter_rows(native)), schema=to_pyarrow_schema())

        self.assertEqual(table.num_rows, 1)
        row = table.to_pylist()[0]
        self.assertEqual(row["chunk_id"], "statpearls:nbk430685:s0:introduction:0")
        self.assertEqual(row["source"], "statpearls")
        self.assertEqual(row["source_type"], "clinical_overview")
        self.assertEqual(row["publication_types"], ["Clinical Overview"])
        self.assertEqual(row["journal"], "StatPearls")
        self.assertEqual(row["external_id"], "NBK430685")
        self.assertEqual(row["year"], 2024)
        self.assertTrue(row["is_review"])
        self.assertFalse(row["is_systematic_review"])


if __name__ == "__main__":
    unittest.main()
