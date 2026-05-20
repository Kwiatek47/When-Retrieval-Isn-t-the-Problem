from __future__ import annotations

import asyncio
from dataclasses import replace
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
        expansion_multiplier: int = 2,
        expanded_limit_max: int = 100,
    ) -> None:
        self.embedding_service_url = embedding_service_url.rstrip("/")
        self.embedding_timeout = embedding_timeout
        self.embedding_dimension = embedding_dimension
        self.expansion_multiplier = expansion_multiplier
        self.expanded_limit_max = expanded_limit_max

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        query_texts = _query_texts(query)
        metadata_filter = _metadata_filter_payload(query)
        expanded_limit = _expanded_candidate_limit(
            limit,
            query,
            multiplier=self.expansion_multiplier,
            max_limit=self.expanded_limit_max,
        )
        results = [
            await self._hybrid_query(query_text, expanded_limit, metadata_filter=metadata_filter)
            for query_text in query_texts
        ]
        merged_documents = _weighted_rrf_merge(results, query_texts=query_texts, limit=expanded_limit)
        boosted_documents = _metadata_boost_documents(merged_documents, query=query, limit=0)
        documents = boosted_documents if limit <= 0 else boosted_documents[:limit]
        return RetrievalResult(
            query=query,
            documents=documents,
            provider=self.provider_name,
            debug={
                "rrf_documents": merged_documents,
                "metadata_boosted_documents": boosted_documents,
                "expanded_limit": expanded_limit,
            },
        )

    def _embedding_query_text(self, query: PreRetrievalResult) -> str:
        search_queries = [item.strip() for item in query.search_queries if item.strip()]
        if search_queries:
            return search_queries[-1]
        return query.normalized_query

    async def _hybrid_query(
        self,
        query_text: str,
        limit: int,
        *,
        metadata_filter: dict[str, Any],
    ) -> list[RetrievedDocument]:
        payload = {"text": query_text, "limit": limit}
        if metadata_filter:
            payload["metadata_filter"] = metadata_filter
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
        expansion_multiplier: int = 2,
        expanded_limit_max: int = 100,
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
        self.expansion_multiplier = expansion_multiplier
        self.expanded_limit_max = expanded_limit_max
        self._bm25_encoder: BM25SparseEncoder | None = None

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        query_texts = _query_texts(query)
        query_filter = self._qdrant_filter(query)
        expanded_limit = _expanded_candidate_limit(
            limit,
            query,
            multiplier=self.expansion_multiplier,
            max_limit=self.expanded_limit_max,
        )
        results = []
        for query_text in query_texts:
            dense_vector = await self._embed_query(query_text)
            sparse_vector = self._generate_sparse_vector(query_text)
            points = await self._query_qdrant(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                query_filter=query_filter,
                limit=expanded_limit,
            )
            results.append([self._map_point(point) for point in points])
        merged_documents = _weighted_rrf_merge(results, query_texts=query_texts, limit=expanded_limit)
        boosted_documents = _metadata_boost_documents(merged_documents, query=query, limit=0)
        documents = boosted_documents if limit <= 0 else boosted_documents[:limit]
        return RetrievalResult(
            query=query,
            documents=documents,
            provider=self.provider_name,
            debug={
                "rrf_documents": merged_documents,
                "metadata_boosted_documents": boosted_documents,
                "expanded_limit": expanded_limit,
            },
        )

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
        query_filter: Any | None,
        limit: int,
    ) -> list[Any]:
        models = self._qdrant_models()
        call = self.qdrant_client.query_points
        if not getattr(sparse_vector, "indices", None):
            kwargs = {
                "collection_name": self.collection_name,
                "query": dense_vector,
                "using": self.dense_vector_name,
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            return await self._call_qdrant(call, kwargs)

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
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        return await self._call_qdrant(call, kwargs)

    async def _call_qdrant(self, call: Any, kwargs: dict[str, Any]) -> list[Any]:
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

    def _qdrant_filter(self, query: PreRetrievalResult) -> Any | None:
        models = self._qdrant_models()
        must = []
        corpus_version = query.filters.get("corpusVersion")
        if corpus_version:
            must.append(
                models.FieldCondition(
                    key="corpusVersion",
                    match=models.MatchValue(value=corpus_version),
                )
            )
        if query.min_year is not None:
            must.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=float(query.min_year)),
                )
            )
        if not must:
            return None
        return models.Filter(must=must)

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
            "parentChunkId",
            "parentWordCount",
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
            "corpusType",
            "sourceAuthority",
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


def _query_texts(query: PreRetrievalResult) -> list[str]:
    unique = []
    seen = set()
    for value in [*query.search_queries, query.normalized_query]:
        query_text = value.strip()
        key = query_text.lower()
        if query_text and key not in seen:
            unique.append(query_text)
            seen.add(key)
    return unique or [query.normalized_query]


def _expanded_candidate_limit(
    limit: int,
    query: PreRetrievalResult,
    *,
    multiplier: int = 2,
    max_limit: int = 100,
) -> int:
    if limit <= 0:
        return limit
    if not (query.preferred_publication_types or query.requires_recent_evidence or query.min_year is not None):
        return limit
    return max(limit, min(limit * max(multiplier, 1), max_limit))


def _metadata_filter_payload(query: PreRetrievalResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"intent": query.intent}
    corpus_version = query.filters.get("corpusVersion")
    if corpus_version:
        payload["corpusVersion"] = corpus_version
    if query.min_year is not None:
        payload["min_year"] = query.min_year
    if query.preferred_publication_types:
        payload["preferred_publication_types"] = query.preferred_publication_types
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def _weighted_rrf_merge(
    results_by_query: list[list[RetrievedDocument]],
    *,
    query_texts: list[str],
    limit: int,
    rrf_k: int = 60,
) -> list[RetrievedDocument]:
    merged: dict[str, tuple[RetrievedDocument, float, list[str], list[float]]] = {}
    for query_index, documents in enumerate(results_by_query):
        query_weight = _query_weight(query_index)
        query_text = query_texts[query_index] if query_index < len(query_texts) else ""
        for rank, document in enumerate(documents, start=1):
            if not document.content.strip():
                continue
            key = _document_merge_key(document)
            rrf_score = query_weight / float(rrf_k + rank)
            raw_score = float(document.score or 0.0)
            if key not in merged:
                merged[key] = (document, rrf_score, [query_text], [raw_score])
                continue

            current_document, current_score, matched_queries, raw_scores = merged[key]
            best_document = document if raw_score > max(raw_scores or [current_document.score]) else current_document
            merged[key] = (
                best_document,
                current_score + rrf_score,
                [*matched_queries, query_text],
                [*raw_scores, raw_score],
            )

    reranked = []
    for document, rrf_score, matched_queries, raw_scores in merged.values():
        metadata = {
            **document.metadata,
            "rrfScore": rrf_score,
            "matchedQueryCount": len(set(matched_queries)),
            "matchedQueries": matched_queries,
            "rawRetrievalScores": raw_scores,
        }
        reranked.append(replace(document, score=rrf_score, metadata=metadata))

    return sorted(reranked, key=lambda item: item.score, reverse=True)[:limit]


def _metadata_boost_documents(
    documents: list[RetrievedDocument],
    *,
    query: PreRetrievalResult,
    limit: int,
) -> list[RetrievedDocument]:
    boosted_documents = []
    for document in documents:
        boost, reasons = _metadata_boost(document, query)
        boosted_score = float(document.score or 0.0) * boost
        metadata = {
            **document.metadata,
            "metadataBoost": round(boost, 6),
            "metadataBoostReasons": reasons,
            "preMetadataBoostScore": document.score,
        }
        boosted_documents.append(replace(document, score=boosted_score, metadata=metadata))

    ranked = sorted(boosted_documents, key=lambda item: item.score, reverse=True)
    return ranked if limit <= 0 else ranked[:limit]


def _metadata_boost(document: RetrievedDocument, query: PreRetrievalResult) -> tuple[float, list[str]]:
    publication_types = _publication_types(document.metadata)
    preferred = {item.lower() for item in query.preferred_publication_types}
    reasons: list[str] = []
    boost = 1.0

    if publication_types & preferred:
        boost += 0.10
        reasons.append("preferred_publication_type")
    if _truthy(document.metadata.get("isSystematicReview")) or "systematic review" in publication_types:
        boost += 0.08
        reasons.append("systematic_review")
    if publication_types & {"practice guideline", "guideline"}:
        boost += 0.08
        reasons.append("guideline")
    if query.intent in {"treatment", "diagnosis", "adverse_effects"} and publication_types & {
        "meta-analysis",
        "randomized controlled trial",
        "clinical trial",
    }:
        boost += 0.05
        reasons.append("clinical_evidence_type")

    year = _optional_int(document.metadata.get("year"))
    if query.min_year is not None and year is not None and year >= query.min_year:
        boost += 0.03
        reasons.append("recent_evidence")

    if publication_types & {"case reports", "letter", "editorial", "comment"}:
        boost -= 0.12
        reasons.append("weak_publication_type")

    return max(boost, 0.75), reasons


def _publication_types(metadata: dict[str, Any]) -> set[str]:
    value = metadata.get("publicationTypes") or metadata.get("publication_types") or []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped.strip("[]")
        return {
            item.strip().strip("'\"").lower()
            for item in stripped.replace(",", ";").split(";")
            if item.strip().strip("'\"")
        }
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _query_weight(index: int) -> float:
    if index == 0:
        return 1.0
    if index == 1:
        return 0.8
    return 0.6


def _document_merge_key(document: RetrievedDocument) -> str:
    metadata = document.metadata
    for key in ("parentChunkId", "parent_chunk_id", "chunkId", "chunk_id", "documentId", "document_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return document.id or f"{document.source}:{document.title}"
