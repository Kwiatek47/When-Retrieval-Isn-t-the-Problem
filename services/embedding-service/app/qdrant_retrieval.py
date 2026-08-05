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

    def search(
        self,
        query_vector: list[float],
        *,
        query_text: str,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[HybridQueryDocument]:
        sparse_payload = self._get_bm25_encoder().encode_query(query_text)
        if not sparse_payload["indices"]:
            return self._search_dense(query_vector, limit=limit, metadata_filter=metadata_filter)

        sparse_vector = models.SparseVector(
            indices=sparse_payload["indices"],
            values=sparse_payload["values"],
        )
        query_filter = self._build_filter(metadata_filter)
        client = QdrantClient(path="/raid/s203270/When-Retrieval-Isn-t-the-Problem/local_qdrant_db")
        try:
            kwargs = {
                "collection_name": self.collection_name,
                "prefetch": [
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
                "query": models.FusionQuery(fusion=models.Fusion.RRF),
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            result = client.query_points(**kwargs)
            return [self._map_point(point) for point in result.points]
        finally:
            client.close()

    def _search_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[HybridQueryDocument]:
        query_filter = self._build_filter(metadata_filter)
        client = QdrantClient(path="/raid/s203270/When-Retrieval-Isn-t-the-Problem/local_qdrant_db")
        try:
            kwargs = {
                "collection_name": self.collection_name,
                "query": query_vector,
                "using": self.vector_name,
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            result = client.query_points(**kwargs)
            return [self._map_point(point) for point in result.points]
        finally:
            client.close()

    def _build_filter(self, metadata_filter: dict[str, Any] | None) -> models.Filter | None:
        if not metadata_filter:
            return None

        must = []
        corpus_version = str(metadata_filter.get("corpusVersion") or "").strip()
        if corpus_version:
            must.append(
                models.FieldCondition(
                    key="corpusVersion",
                    match=models.MatchValue(value=corpus_version),
                )
            )
        source = str(metadata_filter.get("source") or "").strip()
        if source:
            must.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source),
                )
            )
        external_ids = _csv_values(metadata_filter.get("externalId"))
        if external_ids:
            must.append(
                models.FieldCondition(
                    key="externalId",
                    match=models.MatchAny(any=external_ids),
                )
            )
        min_year = _optional_int(metadata_filter.get("min_year"))
        if min_year is not None:
            must.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=float(min_year)),
                )
            )

        if not must:
            return None
        return models.Filter(must=must)

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
        external_id = payload.get("externalId")
        source_name = payload.get("sourceName")
        if source_name and external_id:
            return f"{source_name} {external_id}"
        if str(source or "").lower() == "nice" and external_id:
            return f"NICE {external_id}"
        if source and pmid:
            return f"{source}: PMID {pmid}"
        if pmid:
            return f"PMID {pmid}"
        return str(source or "qdrant")

    def _metadata_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata_keys = (
            "chunkId",
            "pmid",
            "externalId",
            "doi",
            "journal",
            "year",
            "authors",
            "meshTerms",
            "section",
            "sectionName",
            "headerPath",
            "guidanceType",
            "chunkIndex",
            "parentChunkId",
            "parentWordCount",
            "documentId",
            "url",
            "sourceUrl",
            "sourceType",
            "sourceName",
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


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _csv_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]
