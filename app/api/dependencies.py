from functools import lru_cache

import httpx

from app.core.config import get_settings
from app.providers.base import LLMProvider
from app.providers.ollama import OllamaProvider
from app.rag.evidence_judge import EvidenceJudge
from app.rag.pipeline import RagPipeline
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import (
    EmbeddingServiceHybridRetriever,
    MedicalKnowledgeRetriever,
    QdrantHybridKnowledgeRetriever,
)


@lru_cache
def get_ollama_provider() -> OllamaProvider:
    settings = get_settings()
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
    )


def get_llm_provider() -> LLMProvider:
    return get_ollama_provider()


@lru_cache
def get_pre_retriever() -> PreRetriever:
    settings = get_settings()
    return PreRetriever(
        ollama_base_url=settings.ollama_base_url,
        rewrite_model=settings.query_rewrite_model,
        rewrite_timeout=settings.query_rewrite_timeout,
        active_corpus_version=settings.rag_corpus_version,
    )


@lru_cache
def get_medical_knowledge_retriever() -> MedicalKnowledgeRetriever:
    settings = get_settings()
    if settings.rag_retriever == "qdrant_hybrid":
        from qdrant_client import QdrantClient

        return QdrantHybridKnowledgeRetriever(
            qdrant_client=QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=settings.qdrant_timeout,
            ),
            embedding_http_client=httpx.AsyncClient(
                base_url=settings.embedding_service_url,
                timeout=settings.embedding_timeout,
            ),
            collection_name=settings.qdrant_collection,
            dense_vector_name=settings.qdrant_vector_name,
            sparse_vector_name=settings.qdrant_sparse_vector_name,
            embedding_dimension=settings.embedding_dimension,
            bm25_stats_path=settings.bm25_stats_path,
            expansion_multiplier=settings.rag_retrieval_expansion_multiplier,
            expanded_limit_max=settings.rag_retrieval_expanded_limit_max,
        )

    return EmbeddingServiceHybridRetriever(
        embedding_service_url=settings.embedding_service_url,
        embedding_timeout=settings.embedding_timeout,
        embedding_dimension=settings.embedding_dimension,
        expansion_multiplier=settings.rag_retrieval_expansion_multiplier,
        expanded_limit_max=settings.rag_retrieval_expanded_limit_max,
    )


@lru_cache
def get_post_retriever() -> PostRetriever:
    settings = get_settings()
    return PostRetriever(
        max_context_chars=settings.rag_max_context_chars,
        final_documents_limit=settings.rag_top_k,
        max_excerpt_chars=settings.rag_max_excerpt_chars,
        cross_encoder_model_name=settings.cross_encoder_model_name,
        cross_encoder_max_length=settings.cross_encoder_max_length,
        cross_encoder_batch_size=settings.cross_encoder_batch_size,
        cross_encoder_device=settings.cross_encoder_device,
        evidence_filter_enabled=settings.rag_evidence_filter_enabled,
    )


@lru_cache
def get_evidence_judge() -> EvidenceJudge:
    settings = get_settings()
    return EvidenceJudge(
        enabled=settings.rag_evidence_judge_enabled,
        method=settings.rag_evidence_judge_method,
        max_sources=settings.rag_evidence_judge_max_sources,
        voting_enabled=settings.rag_evidence_judge_voting_enabled,
        voting_rounds=settings.rag_evidence_judge_votes,
    )


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    settings = get_settings()
    return RagPipeline(
        pre_retriever=get_pre_retriever(),
        retriever=get_medical_knowledge_retriever(),
        post_retriever=get_post_retriever(),
        retrieval_candidate_limit=max(settings.rag_candidate_k, settings.rag_top_k),
        adaptive_retrieval_enabled=settings.rag_adaptive_retrieval_enabled,
        adaptive_max_rounds=settings.rag_adaptive_max_rounds,
    )
