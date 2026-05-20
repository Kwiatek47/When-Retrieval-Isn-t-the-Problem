from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT_COLLECTION = "MedicalChunk_pubmed_reviews_v1_medcpt_20260518"


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
    rag_corpus_version: str
    bm25_stats_path: Path
    rag_retriever: str
    cross_encoder_model_name: str | None
    cross_encoder_max_length: int
    cross_encoder_batch_size: int
    cross_encoder_device: str | None
    rag_candidate_k: int
    rag_top_k: int
    rag_max_context_chars: int
    rag_max_excerpt_chars: int
    rag_adaptive_retrieval_enabled: bool
    rag_adaptive_max_rounds: int
    rag_retrieval_expansion_multiplier: int
    rag_retrieval_expanded_limit_max: int
    rag_evidence_filter_enabled: bool
    rag_evidence_judge_enabled: bool
    rag_evidence_judge_method: str
    rag_evidence_judge_model: str
    rag_evidence_judge_max_sources: int
    rag_evidence_judge_voting_enabled: bool
    rag_evidence_judge_votes: int
    rag_citation_repair_enabled: bool
    rag_answer_quality_gate_enabled: bool
    rag_answer_quality_max_hallucination_rate: float
    rag_extractive_fallback_enabled: bool
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
        default_model=os.getenv("OLLAMA_MODEL", "gemma4:26b"),
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
        qdrant_collection=os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        qdrant_vector_name=os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense"),
        qdrant_sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25_sparse"),
        rag_corpus_version=os.getenv("RAG_CORPUS_VERSION", ""),
        bm25_stats_path=Path(os.getenv("BM25_STATS_PATH", PROJECT_ROOT / "data" / "bm25_stats.json")),
        rag_retriever=os.getenv("RAG_RETRIEVER", "embedding_service"),
        cross_encoder_model_name=os.getenv("CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder") or None,
        cross_encoder_max_length=int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512")),
        cross_encoder_batch_size=int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "8")),
        cross_encoder_device=os.getenv("CROSS_ENCODER_DEVICE") or None,
        rag_candidate_k=int(os.getenv("RAG_CANDIDATE_K", "50")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000")),
        rag_max_excerpt_chars=int(os.getenv("RAG_MAX_EXCERPT_CHARS", "1600")),
        rag_adaptive_retrieval_enabled=_bool_env("RAG_ADAPTIVE_RETRIEVAL_ENABLED", True),
        rag_adaptive_max_rounds=int(os.getenv("RAG_ADAPTIVE_MAX_ROUNDS", "1")),
        rag_retrieval_expansion_multiplier=int(os.getenv("RAG_RETRIEVAL_EXPANSION_MULTIPLIER", "2")),
        rag_retrieval_expanded_limit_max=int(os.getenv("RAG_RETRIEVAL_EXPANDED_LIMIT_MAX", "100")),
        rag_evidence_filter_enabled=_bool_env("RAG_EVIDENCE_FILTER_ENABLED", True),
        rag_evidence_judge_enabled=_bool_env("RAG_EVIDENCE_JUDGE_ENABLED", True),
        rag_evidence_judge_method=os.getenv("RAG_EVIDENCE_JUDGE_METHOD", "llm"),
        rag_evidence_judge_model=os.getenv("RAG_EVIDENCE_JUDGE_MODEL", ""),
        rag_evidence_judge_max_sources=int(os.getenv("RAG_EVIDENCE_JUDGE_MAX_SOURCES", "3")),
        rag_evidence_judge_voting_enabled=_bool_env("RAG_EVIDENCE_JUDGE_VOTING_ENABLED", False),
        rag_evidence_judge_votes=int(os.getenv("RAG_EVIDENCE_JUDGE_VOTES", "3")),
        rag_citation_repair_enabled=_bool_env("RAG_CITATION_REPAIR_ENABLED", True),
        rag_answer_quality_gate_enabled=_bool_env("RAG_ANSWER_QUALITY_GATE_ENABLED", True),
        rag_answer_quality_max_hallucination_rate=float(
            os.getenv("RAG_ANSWER_QUALITY_MAX_HALLUCINATION_RATE", "0.25")
        ),
        rag_extractive_fallback_enabled=_bool_env("RAG_EXTRACTIVE_FALLBACK_ENABLED", True),
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


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
