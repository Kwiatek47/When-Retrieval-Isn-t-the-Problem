from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx
import weaviate
from weaviate.classes.query import MetadataQuery

from app.rag.models import PreRetrievalResult, RetrievedDocument, RetrievalResult


class MedicalKnowledgeRetriever(Protocol):
    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        ...


class EmptyMedicalKnowledgeRetriever:
    """Temporary retriever used until the VectorDB implementation is connected."""

    provider_name = "empty"

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        return RetrievalResult(query=query, documents=[], provider=self.provider_name)


class WeaviateMedicalKnowledgeRetriever:
    """Retrieve medical chunks from Weaviate using externally computed query embeddings."""

    provider_name = "weaviate"

    def __init__(
        self,
        *,
        embedding_service_url: str,
        embedding_timeout: float,
        embedding_dimension: int,
        weaviate_http_host: str,
        weaviate_http_port: int,
        weaviate_grpc_host: str,
        weaviate_grpc_port: int,
        collection_name: str,
    ) -> None:
        self.embedding_service_url = embedding_service_url.rstrip("/")
        self.embedding_timeout = embedding_timeout
        self.embedding_dimension = embedding_dimension
        self.weaviate_http_host = weaviate_http_host
        self.weaviate_http_port = weaviate_http_port
        self.weaviate_grpc_host = weaviate_grpc_host
        self.weaviate_grpc_port = weaviate_grpc_port
        self.collection_name = collection_name

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        query_text = self._embedding_query_text(query)
        query_vector = await self._embed_query(query_text)
        documents = await asyncio.to_thread(self._search_weaviate, query_vector, limit)
        return RetrievalResult(query=query, documents=documents, provider=self.provider_name)

    def _embedding_query_text(self, query: PreRetrievalResult) -> str:
        search_queries = [item.strip() for item in query.search_queries if item.strip()]
        if search_queries:
            return search_queries[-1]
        return query.normalized_query

    async def _embed_query(self, query_text: str) -> list[float]:
        payload = {"text": query_text}
        async with httpx.AsyncClient(
            base_url=self.embedding_service_url,
            timeout=self.embedding_timeout,
        ) as client:
            response = await client.post("/embed/query", json=payload)
            response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Embedding service returned no query embedding.")

        query_vector = embeddings[0]
        if len(query_vector) != self.embedding_dimension:
            raise RuntimeError(
                f"Expected embedding dimension {self.embedding_dimension}, got {len(query_vector)}."
            )

        return [float(value) for value in query_vector]

    def _search_weaviate(self, query_vector: list[float], limit: int) -> list[RetrievedDocument]:
        client = weaviate.connect_to_custom(
            http_host=self.weaviate_http_host,
            http_port=self.weaviate_http_port,
            http_secure=False,
            grpc_host=self.weaviate_grpc_host,
            grpc_port=self.weaviate_grpc_port,
            grpc_secure=False,
        )
        try:
            chunks = client.collections.get(self.collection_name)
            result = chunks.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            return [self._map_object(item) for item in result.objects]
        finally:
            client.close()

    def _map_object(self, item: Any) -> RetrievedDocument:
        properties = item.properties or {}
        distance = getattr(item.metadata, "distance", None)
        score = self._score_from_distance(distance)

        title = str(properties.get("title") or "Untitled medical chunk")
        content = str(properties.get("text") or "")
        source = self._source_from_properties(properties)
        metadata = self._metadata_from_properties(properties, distance)

        return RetrievedDocument(
            id=str(
                getattr(item, "uuid", "")
                or properties.get("documentId")
                or properties.get("pmid")
                or title
            ),
            title=title,
            content=content,
            source=source,
            score=score,
            metadata=metadata,
        )

    def _score_from_distance(self, distance: float | None) -> float:
        if distance is None:
            return 0.0
        return 1.0 - float(distance)

    def _source_from_properties(self, properties: dict[str, Any]) -> str:
        source = properties.get("source")
        pmid = properties.get("pmid")
        if source and pmid:
            return f"{source}: PMID {pmid}"
        if pmid:
            return f"PMID {pmid}"
        return str(source or "weaviate")

    def _metadata_from_properties(
        self,
        properties: dict[str, Any],
        distance: float | None,
    ) -> dict[str, Any]:
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
        metadata = {key: properties[key] for key in metadata_keys if key in properties}
        if distance is not None:
            metadata["distance"] = float(distance)
        return metadata
