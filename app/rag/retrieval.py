from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.rag.models import PreRetrievalResult, RetrievedDocument, RetrievalResult
from app.rag.sparse import BM25SparseEncoder


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


class QdrantHybridKnowledgeRetriever:
    """Hybrid dense+sparse medical retriever using Qdrant RRF fusion."""

    provider_name = "qdrant_hybrid"

    def __init__(
        self,
        *,
        qdrant_client: Any,
        embedding_http_client: httpx.AsyncClient,
        collection_name: str,
        dense_vector_name: str = "medcpt_dense",
        sparse_vector_name: str = "bm25_sparse",
        embedding_dimension: int = 768,
        dense_prefetch_limit: int = 50,
        sparse_prefetch_limit: int = 50,
        embedding_query_path: str = "/embed/query",
        bm25_stats_path: Path | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.embedding_http_client = embedding_http_client
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.embedding_dimension = embedding_dimension
        self.dense_prefetch_limit = dense_prefetch_limit
        self.sparse_prefetch_limit = sparse_prefetch_limit
        self.embedding_query_path = embedding_query_path
        self.bm25_stats_path = bm25_stats_path
        self._bm25_encoder: BM25SparseEncoder | None = None

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        query_text = self._embedding_query_text(query)
        dense_vector = await self._embed_query(query_text)
        sparse_vector = self._generate_sparse_vector(query_text)
        points = await self._query_qdrant(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )
        documents = [self._map_point(point) for point in points]
        return RetrievalResult(query=query, documents=documents, provider=self.provider_name)

    def _embedding_query_text(self, query: PreRetrievalResult) -> str:
        search_queries = [item.strip() for item in query.search_queries if item.strip()]
        if search_queries:
            return search_queries[-1]
        return query.normalized_query

    async def _embed_query(self, query_text: str) -> list[float]:
        response = await self.embedding_http_client.post(
            self.embedding_query_path,
            json={"text": query_text},
        )
        response.raise_for_status()
        data = response.json()

        dimension = data.get("dimension")
        if dimension != self.embedding_dimension:
            raise RuntimeError(
                f"Expected embedding dimension {self.embedding_dimension}, got {dimension}."
            )

        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Embedding service returned no query embedding.")
        return [float(value) for value in embeddings[0]]

    async def _query_qdrant(
        self,
        *,
        dense_vector: list[float],
        sparse_vector: Any,
        limit: int,
    ) -> list[Any]:
        models = self._qdrant_models()
        call = self.qdrant_client.query_points
        kwargs = {
            "collection_name": self.collection_name,
            "prefetch": [
                models.Prefetch(
                    query=dense_vector,
                    using=self.dense_vector_name,
                    limit=max(limit, self.dense_prefetch_limit),
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=self.sparse_vector_name,
                    limit=max(limit, self.sparse_prefetch_limit),
                ),
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": limit,
            "with_payload": True,
        }

        if inspect.iscoroutinefunction(call):
            result = await call(**kwargs)
        else:
            result = await asyncio.to_thread(call, **kwargs)
            if inspect.isawaitable(result):
                result = await result

        return list(getattr(result, "points", result))

    def _generate_sparse_vector(self, query: str) -> Any:
        models = self._qdrant_models()
        sparse_payload = self._get_bm25_encoder().encode_query(query)
        return models.SparseVector(
            indices=sparse_payload["indices"],
            values=sparse_payload["values"],
        )

    def _get_bm25_encoder(self) -> BM25SparseEncoder:
        if self.bm25_stats_path is None:
            raise RuntimeError("BM25 stats path is required for hybrid sparse retrieval.")
        if self._bm25_encoder is None:
            self._bm25_encoder = BM25SparseEncoder.from_file(self.bm25_stats_path)
        return self._bm25_encoder

    def _map_point(self, point: Any) -> RetrievedDocument:
        payload = getattr(point, "payload", None) or {}
        title = str(payload.get("title") or "Untitled medical chunk")
        content = str(payload.get("text") or payload.get("content") or "")

        return RetrievedDocument(
            id=str(getattr(point, "id", "") or ""),
            title=title,
            content=content,
            source=self._source_from_payload(payload),
            score=float(getattr(point, "score", 0.0) or 0.0),
            metadata=self._metadata_from_payload(payload),
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

    def _qdrant_models(self) -> Any:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "QdrantHybridKnowledgeRetriever requires qdrant-client. "
                "Install it before enabling this retriever."
            ) from exc
        return models
