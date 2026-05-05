from __future__ import annotations

import os
import time

from qdrant_client import QdrantClient, models


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "MedicalChunk")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))


PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "pmid": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
    "journal": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "authors": models.PayloadSchemaType.KEYWORD,
    "meshTerms": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "chunkIndex": models.PayloadSchemaType.INTEGER,
    "documentId": models.PayloadSchemaType.KEYWORD,
    "embeddingModel": models.PayloadSchemaType.KEYWORD,
    "corpusVersion": models.PayloadSchemaType.KEYWORD,
}


def connect_with_retry() -> QdrantClient:
    last_error: Exception | None = None
    for _ in range(30):
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
        try:
            client.get_collections()
            return client
        except Exception as exc:  # pragma: no cover - startup guard for docker compose
            last_error = exc
            client.close()
            time.sleep(2)

    raise RuntimeError(f"Qdrant is not ready: {last_error}")


def ensure_medical_chunk_collection(client: QdrantClient) -> None:
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
            hnsw_config=models.HnswConfigDiff(on_disk=True),
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
