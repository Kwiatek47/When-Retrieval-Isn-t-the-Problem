from functools import lru_cache
import os

from pydantic import BaseModel


class Settings(BaseModel):
    app_title: str = "Embedding Service"
    app_version: str = "0.1.0"
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "medcpt-ncbi-v1")
    query_model_name: str = os.getenv("MEDCPT_QUERY_MODEL", "ncbi/MedCPT-Query-Encoder")
    document_model_name: str = os.getenv("MEDCPT_DOCUMENT_MODEL", "ncbi/MedCPT-Article-Encoder")
    embedding_dimension: int = 768
    query_max_length: int = int(os.getenv("MEDCPT_QUERY_MAX_LENGTH", "64"))
    document_max_length: int = int(os.getenv("MEDCPT_DOCUMENT_MAX_LENGTH", "512"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    qdrant_host: str = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_timeout: float = float(os.getenv("QDRANT_TIMEOUT", "10"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "MedicalChunk")
    qdrant_vector_name: str = os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense")


@lru_cache
def get_settings() -> Settings:
    return Settings()
