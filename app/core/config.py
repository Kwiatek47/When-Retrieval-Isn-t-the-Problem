from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_version: str
    default_model: str
    ollama_base_url: str
    ollama_timeout: float
    embedding_service_url: str
    embedding_timeout: float
    embedding_dimension: int
    weaviate_http_host: str
    weaviate_http_port: int
    weaviate_grpc_host: str
    weaviate_grpc_port: int
    weaviate_collection: str
    rag_top_k: int
    rag_max_context_chars: int
    static_dir: Path
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_title="MedChat",
        app_version="0.1.0",
        default_model=os.getenv("OLLAMA_MODEL", "medgemma"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "60")),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8080"),
        embedding_timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        weaviate_http_host=os.getenv("WEAVIATE_HTTP_HOST", "weaviate"),
        weaviate_http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST", os.getenv("WEAVIATE_HTTP_HOST", "weaviate")),
        weaviate_grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        weaviate_collection=os.getenv("WEAVIATE_COLLECTION", "MedicalChunk"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000")),
        static_dir=PROJECT_ROOT / "static",
        system_prompt=(
            "You are a professional medical assistant. Provide concise, helpful information. "
            "Always include a disclaimer that this is not medical advice."
        ),
    )
