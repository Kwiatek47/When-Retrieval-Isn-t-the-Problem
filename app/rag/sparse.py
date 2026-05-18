from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder": "bm25",
            "document_count": self.document_count,
            "average_document_length": self.average_document_length,
            "document_frequency": self.document_frequency,
            "vocabulary": self.vocabulary,
            "k1": self.k1,
            "b": self.b,
        }

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
    def from_corpus(
        cls,
        documents: Iterable[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "BM25SparseEncoder":
        document_frequency: Counter[str] = Counter()
        document_count = 0
        total_document_length = 0
        for document in documents:
            tokens = _tokenize(document)
            document_count += 1
            total_document_length += len(tokens)
            document_frequency.update(set(tokens))

        if document_count == 0:
            raise ValueError("Cannot build BM25 stats from an empty corpus.")

        vocabulary = {term: index for index, term in enumerate(sorted(document_frequency))}
        stats = BM25Stats(
            document_count=document_count,
            average_document_length=total_document_length / document_count,
            document_frequency=dict(document_frequency),
            vocabulary=vocabulary,
            k1=k1,
            b=b,
        )
        return cls(stats)

    @classmethod
    def from_file(cls, path: Path) -> "BM25SparseEncoder":
        with path.open(encoding="utf-8") as file:
            return cls(BM25Stats.from_dict(json.load(file)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.stats.to_dict(), file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")

    def encode_document(self, text: str) -> SparseVectorPayload:
        return self._encode(text, mode="document")

    def encode_query(self, text: str) -> SparseVectorPayload:
        return self._encode(text, mode="query")

    def _encode(self, text: str, *, mode: Literal["document", "query"]) -> SparseVectorPayload:
        tokens = _tokenize(text)
        term_counts = Counter(token for token in tokens if token in self.stats.vocabulary)
        if not term_counts:
            return {"indices": [], "values": []}

        document_length = max(len(tokens), 1)
        indices: list[int] = []
        values: list[float] = []

        for term in sorted(term_counts):
            indices.append(self.stats.vocabulary[term])
            if mode == "document":
                values.append(self._document_term_weight(term, term_counts[term], document_length))
            else:
                values.append(self._query_term_weight(term_counts[term]))

        return {"indices": indices, "values": values}

    def _document_term_weight(self, term: str, frequency: int, document_length: int) -> float:
        numerator = frequency * (self.stats.k1 + 1.0)
        length_normalizer = 1.0 - self.stats.b + self.stats.b * (
            document_length / self.stats.average_document_length
        )
        denominator = frequency + self.stats.k1 * length_normalizer
        return self._idf(term) * (numerator / denominator)

    def _query_term_weight(self, frequency: int) -> float:
        return 1.0 + math.log(float(frequency))

    def _idf(self, term: str) -> float:
        document_frequency = self.stats.document_frequency.get(term, 0)
        numerator = self.stats.document_count - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return math.log(1.0 + numerator / denominator)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())
