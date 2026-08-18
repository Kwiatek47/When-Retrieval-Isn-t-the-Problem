from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.sparse import BM25SparseEncoder

PUBMED_SAMPLE_PATH = Path(os.getenv("PUBMED_SAMPLE_PATH", PROJECT_ROOT / "data" / "sample" / "pubmed_sample.json"))
BM25_STATS_PATH = Path(os.getenv("BM25_STATS_PATH", PROJECT_ROOT / "data" / "bm25_stats.json"))
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8081").rstrip("/")
LOCAL_QDRANT_PATH = PROJECT_ROOT / "local_qdrant_db"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "MedicalChunk_pubmed_reviews_v1_medcpt_20260518")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25_sparse")
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "pubmed-sample-v1")


def main() -> None:
    articles = _load_articles(PUBMED_SAMPLE_PATH)
    if not articles:
        raise RuntimeError(f"No articles found in {PUBMED_SAMPLE_PATH}.")

    _ensure_qdrant_collection_exists()
    embedding_response = _embed_articles(articles)
    embeddings = embedding_response["embeddings"]
    if len(embeddings) != len(articles):
        raise RuntimeError(f"Expected {len(articles)} embeddings, got {len(embeddings)}.")

    bm25_encoder = BM25SparseEncoder.from_corpus(_article_sparse_text(article) for article in articles)
    bm25_encoder.save(BM25_STATS_PATH)

    points = [
        _build_point(
            article=article,
            embedding=embedding,
            embedding_model=str(embedding_response["model"]),
            chunk_index=index,
            bm25_encoder=bm25_encoder,
        )
        for index, (article, embedding) in enumerate(zip(articles, embeddings))
    ]
    _upsert_points(points)
    print(f"Upserted {len(points)} PubMed sample chunks into Qdrant collection {QDRANT_COLLECTION} at {LOCAL_QDRANT_PATH}.")


def _load_articles(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of articles in {path}.")
    return [item for item in data if str(item.get("content") or "").strip()]


def _ensure_qdrant_collection_exists() -> None:
    client = QdrantClient(path=str(LOCAL_QDRANT_PATH))
    collections = [col.name for col in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        print(f"Creating local Qdrant collection: {QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                QDRANT_VECTOR_NAME: models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                QDRANT_SPARSE_VECTOR_NAME: models.SparseVectorParams()
            },
        )


def _embed_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "texts": [_article_dense_text(article) for article in articles],
        "corpus_version": CORPUS_VERSION,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{EMBEDDING_SERVICE_URL}/embed/documents",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch embeddings from {EMBEDDING_SERVICE_URL}: {exc}") from exc


def _article_dense_text(article: dict[str, Any]) -> str:
    title = str(article.get("title") or "").strip()
    abstract = str(article.get("content") or "").strip()
    if title and abstract:
        return f"{title} [SEP] {abstract}"
    return title or abstract


def _article_sparse_text(article: dict[str, Any]) -> str:
    title = str(article.get("title") or "").strip()
    abstract = str(article.get("content") or "").strip()
    return f"{title}\n{abstract}".strip()


def _build_point(
    article: dict[str, Any],
    embedding: list[float],
    embedding_model: str,
    chunk_index: int,
    bm25_encoder: BM25SparseEncoder,
) -> models.PointStruct:
    article_id = str(article.get("id") or f"sample_{chunk_index}")
    point_uuid = str(uuid5(NAMESPACE_URL, f"pubmed_sample:{article_id}:{chunk_index}"))
    
    sparse_vector = bm25_encoder.encode_document(_article_sparse_text(article))
    
    payload = {
        "article_id": article_id,
        "title": article.get("title"),
        "content": article.get("content"),
        "corpus_version": CORPUS_VERSION,
        "embedding_model": embedding_model,
        "chunk_index": chunk_index,
    }

    return models.PointStruct(
        id=point_uuid,
        vector={
            QDRANT_VECTOR_NAME: embedding,
            QDRANT_SPARSE_VECTOR_NAME: models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
        },
        payload=payload,
    )


def _upsert_points(points: list[models.PointStruct]) -> None:
    client = QdrantClient(path=str(LOCAL_QDRANT_PATH))
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )


if __name__ == "__main__":
    main()
