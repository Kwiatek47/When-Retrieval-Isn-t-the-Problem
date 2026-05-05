from __future__ import annotations

import os
import time

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances


COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "MedicalChunk")
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", WEAVIATE_HTTP_HOST)
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))


def connect_with_retry() -> weaviate.WeaviateClient:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            client = weaviate.connect_to_custom(
                http_host=WEAVIATE_HTTP_HOST,
                http_port=WEAVIATE_HTTP_PORT,
                http_secure=False,
                grpc_host=WEAVIATE_GRPC_HOST,
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=False,
            )
            if client.is_ready():
                return client
            client.close()
        except Exception as exc:  # pragma: no cover - startup guard for docker compose
            last_error = exc
        time.sleep(2)

    raise RuntimeError(f"Weaviate is not ready: {last_error}")


def ensure_medical_chunk_collection(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(COLLECTION_NAME):
        return

    client.collections.create(
        COLLECTION_NAME,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="pmid", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="journal", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.INT),
            Property(name="authors", data_type=DataType.TEXT_ARRAY),
            Property(name="meshTerms", data_type=DataType.TEXT_ARRAY),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunkIndex", data_type=DataType.INT),
            Property(name="documentId", data_type=DataType.TEXT),
            Property(name="embeddingModel", data_type=DataType.TEXT),
            Property(name="corpusVersion", data_type=DataType.TEXT),
        ],
    )


def main() -> None:
    client = connect_with_retry()
    try:
        ensure_medical_chunk_collection(client)
        print(f"Collection {COLLECTION_NAME} is ready.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
