from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.sparse import BM25SparseEncoder

PUBMED_SAMPLE_PATH = Path(os.getenv("PUBMED_SAMPLE_PATH", PROJECT_ROOT / "data" / "pubmed_sample.json"))
BM25_STATS_PATH = Path(os.getenv("BM25_STATS_PATH", PROJECT_ROOT / "data" / "bm25_stats.json"))
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8081").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
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
    print(f"Upserted {len(points)} PubMed sample chunks into Qdrant collection {QDRANT_COLLECTION}.")


def _load_articles(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of articles in {path}.")
    return [item for item in data if str(item.get("content") or "").strip()]


def _ensure_qdrant_collection_exists() -> None:
    try:
        response = _request_json("GET", f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}")
    except RuntimeError as exc:
        raise RuntimeError(
            f"Qdrant collection {QDRANT_COLLECTION} is unavailable at {QDRANT_URL}. "
            "Start qdrant and qdrant-init before ingestion."
        ) from exc
    sparse_vectors = (
        response.get("result", {})
        .get("config", {})
        .get("params", {})
        .get("sparse_vectors", {})
    )
    if QDRANT_SPARSE_VECTOR_NAME not in sparse_vectors:
        raise RuntimeError(
            f"Qdrant collection {QDRANT_COLLECTION} does not expose sparse vector "
            f"{QDRANT_SPARSE_VECTOR_NAME}. Recreate the collection with the updated qdrant schema."
        )


def _embed_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    documents = [
        {
            "title": str(article.get("title") or ""),
            "text": str(article.get("content") or ""),
        }
        for article in articles
    ]
    return _request_json("POST", f"{EMBEDDING_SERVICE_URL}/embed/documents", {"documents": documents})


def _build_point(
    *,
    article: dict[str, Any],
    embedding: list[float],
    embedding_model: str,
    chunk_index: int,
    bm25_encoder: BM25SparseEncoder,
) -> dict[str, Any]:
    metadata = article.get("metadata") or {}
    document_id = str(article.get("id") or metadata.get("pmid") or f"pubmed-sample-{chunk_index}")
    pmid = str(metadata.get("pmid") or document_id.removeprefix("pmid-"))
    title = str(article.get("title") or "Untitled PubMed sample")
    source = str(article.get("source") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
    topic = str(metadata.get("topic") or "")
    content = str(article.get("content") or "")

    return {
        "id": str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_index}")),
        "vector": {
            QDRANT_VECTOR_NAME: embedding,
            QDRANT_SPARSE_VECTOR_NAME: bm25_encoder.encode_document(_article_sparse_text(article)),
        },
        "payload": {
            "text": content,
            "pmid": pmid,
            "title": title,
            "doi": str(metadata.get("doi") or ""),
            "journal": str(metadata.get("journal") or ""),
            "year": _to_int(metadata.get("year")),
            "authors": [],
            "meshTerms": [term for term in topic.replace(",", " ").split() if term],
            "section": "abstract",
            "source": source,
            "chunkIndex": chunk_index,
            "documentId": document_id,
            "publicationTypes": _string_list(metadata.get("publicationTypes")),
            "isReview": bool(metadata.get("isReview", False)),
            "isSystematicReview": bool(metadata.get("isSystematicReview", False)),
            "corpusType": str(metadata.get("corpusType") or ""),
            "sourceAuthority": str(metadata.get("sourceAuthority") or ""),
            "embeddingModel": embedding_model,
            "corpusVersion": CORPUS_VERSION,
        },
    }


def _article_sparse_text(article: dict[str, Any]) -> str:
    metadata = article.get("metadata") or {}
    return " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("content") or ""),
            str(metadata.get("topic") or ""),
            str(metadata.get("journal") or ""),
            str(metadata.get("pmid") or ""),
        ]
    )


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _upsert_points(points: list[dict[str, Any]]) -> None:
    response = _request_json(
        "PUT",
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
        {"points": points},
    )
    status = str(response.get("status") or "").lower()
    if status not in {"ok", "accepted"}:
        raise RuntimeError(f"Unexpected Qdrant upsert response: {response}")


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


if __name__ == "__main__":
    main()
