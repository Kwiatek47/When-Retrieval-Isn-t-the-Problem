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
    ollama_keep_alive: str
    ollama_num_predict: int
    ollama_num_ctx: int
    query_rewrite_model: str
    query_rewrite_timeout: float
    embedding_service_url: str
    embedding_timeout: float
    embedding_dimension: int
    qdrant_host: str
    qdrant_port: int
    qdrant_timeout: float
    qdrant_collection: str
    qdrant_vector_name: str
    qdrant_sparse_vector_name: str
    bm25_stats_path: Path
    rag_retriever: str
    cross_encoder_model_name: str | None
    rag_candidate_k: int
    rag_top_k: int
    rag_max_context_chars: int
    answer_quality_method: str
    answer_quality_model_name: str
    answer_quality_similarity_threshold: float
    static_dir: Path
    telemetry_path: Path
    active_prompt_version: str
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_title="MedChat",
        app_version="0.1.0",
        default_model=os.getenv("OLLAMA_MODEL", "medgemma"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "400")),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")),
        query_rewrite_model=os.getenv("QUERY_REWRITE_MODEL", "llama3.2:3b"),
        query_rewrite_timeout=float(os.getenv("QUERY_REWRITE_TIMEOUT", "15")),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8080"),
        embedding_timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "10")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "MedicalChunk"),
        qdrant_vector_name=os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense"),
        qdrant_sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25_sparse"),
        bm25_stats_path=Path(os.getenv("BM25_STATS_PATH", PROJECT_ROOT / "data" / "bm25_stats.json")),
        rag_retriever=os.getenv("RAG_RETRIEVER", "embedding_service"),
        cross_encoder_model_name=os.getenv("CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder") or None,
        rag_candidate_k=int(os.getenv("RAG_CANDIDATE_K", "50")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000")),
        answer_quality_method=os.getenv("ANSWER_QUALITY_METHOD", "semantic_similarity"),
        answer_quality_model_name=os.getenv(
            "ANSWER_QUALITY_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        answer_quality_similarity_threshold=float(os.getenv("ANSWER_QUALITY_SIMILARITY_THRESHOLD", "0.45")),
        static_dir=PROJECT_ROOT / "static",
        telemetry_path=Path(os.getenv("TELEMETRY_PATH", str(PROJECT_ROOT / "data" / "telemetry" / "events.jsonl"))),
        active_prompt_version=os.getenv("PROMPT_VERSION", "v4"),
        system_prompt=(
            "You are a neurology clinical decision-support assistant for physicians. "
            "Provide differential diagnosis from patient history and symptom chronology. "
            "Highlight red flags, urgent exclusions, and recommended next diagnostics. "
            "Include a disclaimer that this is decision support and not a final diagnosis."
        ),
    )
