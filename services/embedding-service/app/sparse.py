from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Literal, TypedDict


class SparseVectorPayload(TypedDict):
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class BM25Stats:
    document_count: int
    average_document_length: float
    document_frequency: dict[str, int]
    vocabulary: dict[str, int]
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25Stats":
        if data.get("encoder") != "bm25":
            raise ValueError("BM25 stats file must contain encoder='bm25'.")
        return cls(
            document_count=int(data["document_count"]),
            average_document_length=float(data["average_document_length"]),
            document_frequency={str(term): int(count) for term, count in data["document_frequency"].items()},
            vocabulary={str(term): int(index) for term, index in data["vocabulary"].items()},
            k1=float(data.get("k1", 1.2)),
            b=float(data.get("b", 0.75)),
        )


class BM25SparseEncoder:
    """Corpus-aware BM25 sparse encoder for Qdrant sparse vectors."""

    def __init__(self, stats: BM25Stats) -> None:
        if stats.document_count <= 0:
            raise ValueError("BM25 document_count must be positive.")
        if stats.average_document_length <= 0:
            raise ValueError("BM25 average_document_length must be positive.")
        self.stats = stats

    @classmethod
    def from_file(cls, path: Path) -> "BM25SparseEncoder":
        with path.open(encoding="utf-8") as file:
            return cls(BM25Stats.from_dict(json.load(file)))

    def encode_query(self, text: str) -> SparseVectorPayload:
        return self._encode(text, mode="query")

    def _encode(self, text: str, *, mode: Literal["query"]) -> SparseVectorPayload:
        tokens = _tokenize(text)
        term_counts = Counter(token for token in tokens if token in self.stats.vocabulary)
        if not term_counts:
            return {"indices": [], "values": []}

        indices: list[int] = []
        values: list[float] = []

        for term in sorted(term_counts):
            indices.append(self.stats.vocabulary[term])
            values.append(self._query_term_weight(term_counts[term]))

        return {"indices": indices, "values": values}

    def _query_term_weight(self, frequency: int) -> float:
        return 1.0 + math.log(float(frequency))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())
