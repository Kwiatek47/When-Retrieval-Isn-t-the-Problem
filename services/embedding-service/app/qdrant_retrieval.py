from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from app.schemas import HybridQueryDocument


class QdrantMedicalRetriever:
    """Retrieve medical chunks from Qdrant using externally computed query embeddings."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout: float,
        collection_name: str,
        vector_name: str,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.collection_name = collection_name
        self.vector_name = vector_name

    def search(self, query_vector: list[float], *, limit: int) -> list[HybridQueryDocument]:
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
            "pmid",
            "journal",
            "year",
            "authors",
            "meshTerms",
            "section",
            "chunkIndex",
            "documentId",
            "embeddingModel",
            "corpusVersion",
        )
        return {key: payload[key] for key in metadata_keys if key in payload}
