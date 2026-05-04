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


@lru_cache
def get_settings() -> Settings:
    return Settings()
