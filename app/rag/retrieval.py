from __future__ import annotations

from typing import Any, Protocol

import httpx

from app.rag.models import PreRetrievalResult, RetrievedDocument, RetrievalResult


class MedicalKnowledgeRetriever(Protocol):
    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        ...


class EmptyMedicalKnowledgeRetriever:
    """Temporary retriever used until the VectorDB implementation is connected."""

    provider_name = "empty"

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        return RetrievalResult(query=query, documents=[], provider=self.provider_name)


class EmbeddingServiceHybridRetriever:
    """Retrieve medical chunks through the embedding service hybrid query endpoint."""

    provider_name = "qdrant"

    def __init__(
        self,
        *,
        embedding_service_url: str,
        embedding_timeout: float,
        embedding_dimension: int,
    ) -> None:
        self.embedding_service_url = embedding_service_url.rstrip("/")
        self.embedding_timeout = embedding_timeout
        self.embedding_dimension = embedding_dimension

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        query_text = self._embedding_query_text(query)
        documents = await self._hybrid_query(query_text, limit)
        return RetrievalResult(query=query, documents=documents, provider=self.provider_name)

    def _embedding_query_text(self, query: PreRetrievalResult) -> str:
        search_queries = [item.strip() for item in query.search_queries if item.strip()]
        if search_queries:
            return search_queries[-1]
        return query.normalized_query

    async def _hybrid_query(self, query_text: str, limit: int) -> list[RetrievedDocument]:
        payload = {"text": query_text, "limit": limit}
        async with httpx.AsyncClient(
            base_url=self.embedding_service_url,
            timeout=self.embedding_timeout,
        ) as client:
            response = await client.post("/embed/hybrid/query", json=payload)
            response.raise_for_status()

        data = response.json()
        dimension = data.get("dimension")
        if dimension != self.embedding_dimension:
            raise RuntimeError(
                f"Expected embedding dimension {self.embedding_dimension}, got {dimension}."
            )

        return [self._map_document(item) for item in data.get("documents", [])]

    def _map_document(self, item: dict[str, Any]) -> RetrievedDocument:
        return RetrievedDocument(
            id=str(item.get("id") or ""),
            title=str(item.get("title") or "Untitled medical chunk"),
            content=str(item.get("content") or ""),
            source=str(item.get("source") or "qdrant"),
            score=float(item.get("score") or 0.0),
            metadata=item.get("metadata") or {},
        )
