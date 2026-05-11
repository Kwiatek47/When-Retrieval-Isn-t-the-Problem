from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.schemas import HybridQueryDocument
from app.sparse import BM25SparseEncoder


class QdrantMedicalRetriever:
    """Retrieve medical chunks from Qdrant using dense+sparse RRF fusion."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout: float,
        collection_name: str,
        vector_name: str,
        sparse_vector_name: str,
        bm25_stats_path: Path,
        dense_prefetch_limit: int = 50,
        sparse_prefetch_limit: int = 50,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.sparse_vector_name = sparse_vector_name
        self.bm25_stats_path = bm25_stats_path
        self.dense_prefetch_limit = dense_prefetch_limit
        self.sparse_prefetch_limit = sparse_prefetch_limit
        self._bm25_encoder: BM25SparseEncoder | None = None

    def search(self, query_vector: list[float], *, query_text: str, limit: int) -> list[HybridQueryDocument]:
        sparse_payload = self._get_bm25_encoder().encode_query(query_text)
        if not sparse_payload["indices"]:
            return self._search_dense(query_vector, limit=limit)

        sparse_vector = models.SparseVector(
            indices=sparse_payload["indices"],
            values=sparse_payload["values"],
        )
        client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)
        try:
            result = client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=self.vector_name,
                        limit=max(limit, self.dense_prefetch_limit),
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=self.sparse_vector_name,
                        limit=max(limit, self.sparse_prefetch_limit),
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [self._map_point(point) for point in result.points]
        finally:
            client.close()

    def _search_dense(self, query_vector: list[float], *, limit: int) -> list[HybridQueryDocument]:
        client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)
        try:
            result = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.vector_name,
                limit=limit,
                with_payload=True,
            )
            return [self._map_point(point) for point in result.points]
        finally:
            client.close()

    def _get_bm25_encoder(self) -> BM25SparseEncoder:
        if self._bm25_encoder is None:
            self._bm25_encoder = BM25SparseEncoder.from_file(self.bm25_stats_path)
        return self._bm25_encoder

    def _map_point(self, point: Any) -> HybridQueryDocument:
        payload = point.payload or {}
        title = str(payload.get("title") or "Untitled medical chunk")
        content = str(payload.get("text") or "")
        source = self._source_from_payload(payload)
        metadata = self._metadata_from_payload(payload)

        return HybridQueryDocument(
            id=str(point.id),
            title=title,
            content=content,
            source=source,
            score=float(point.score or 0.0),
            metadata=metadata,
        )

    def _source_from_payload(self, payload: dict[str, Any]) -> str:
        source = payload.get("source")
        pmid = payload.get("pmid")
        if source and pmid:
            return f"{source}: PMID {pmid}"
        if pmid:
            return f"PMID {pmid}"
        return str(source or "qdrant")

    def _metadata_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata_keys = (
            "chunkId",
            "pmid",
            "doi",
            "journal",
            "year",
            "authors",
            "meshTerms",
            "section",
            "chunkIndex",
            "documentId",
            "url",
            "publicationDate",
            "publicationTypes",
            "isReview",
            "isSystematicReview",
            "wordCount",
            "textHash",
            "embeddingModel",
            "corpusVersion",
        )
        return {key: payload[key] for key in metadata_keys if key in payload}
