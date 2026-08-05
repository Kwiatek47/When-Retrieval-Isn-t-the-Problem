from __future__ import annotations

import os
import time

from qdrant_client import QdrantClient, models


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "MedicalChunk_pubmed_reviews_v1_medcpt_20260518")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25_sparse")
RECREATE_COLLECTION = os.getenv("QDRANT_RECREATE_COLLECTION", "").lower() in {"1", "true", "yes"}


PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "chunkId": models.PayloadSchemaType.KEYWORD,
    "documentId": models.PayloadSchemaType.KEYWORD,
    "pmid": models.PayloadSchemaType.KEYWORD,
    "doi": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
    "journal": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "authors": models.PayloadSchemaType.KEYWORD,
    "meshTerms": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "url": models.PayloadSchemaType.KEYWORD,
    "chunkIndex": models.PayloadSchemaType.INTEGER,
    "publicationDate": models.PayloadSchemaType.KEYWORD,
    "publicationTypes": models.PayloadSchemaType.KEYWORD,
    "isReview": models.PayloadSchemaType.BOOL,
    "isSystematicReview": models.PayloadSchemaType.BOOL,
    "wordCount": models.PayloadSchemaType.INTEGER,
    "parentChunkId": models.PayloadSchemaType.KEYWORD,
    "parentWordCount": models.PayloadSchemaType.INTEGER,
    "embeddingModel": models.PayloadSchemaType.KEYWORD,
    "corpusVersion": models.PayloadSchemaType.KEYWORD,
    "textHash": models.PayloadSchemaType.KEYWORD,
}


def connect_with_retry() -> QdrantClient:
    last_error: Exception | None = None
    for _ in range(30):
        client = QdrantClient(path="/raid/s203270/When-Retrieval-Isn-t-the-Problem/local_qdrant_db")
        try:
            client.get_collections()
            return client
        except Exception as exc:  # pragma: no cover - startup guard for docker compose
            last_error = exc
            client.close()
            time.sleep(2)

    raise RuntimeError(f"Qdrant is not ready: {last_error}")


def ensure_medical_chunk_collection(client: QdrantClient) -> None:
    if RECREATE_COLLECTION and client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )
    else:
        collection = client.get_collection(collection_name=COLLECTION_NAME)
        sparse_vectors = getattr(collection.config.params, "sparse_vectors", None) or {}
        if SPARSE_VECTOR_NAME not in sparse_vectors:
            raise RuntimeError(
                f"Collection {COLLECTION_NAME} exists without sparse vector {SPARSE_VECTOR_NAME}. "
                "Set QDRANT_RECREATE_COLLECTION=true to recreate it with hybrid search support."
            )

    for field_name, field_schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise


def main() -> None:
    client = connect_with_retry()
    try:
        ensure_medical_chunk_collection(client)
        print(f"Collection {COLLECTION_NAME} is ready.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
