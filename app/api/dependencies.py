from functools import lru_cache

from app.core.config import get_settings
from app.providers.base import LLMProvider
from app.providers.ollama import OllamaProvider
from app.rag.pipeline import RagPipeline
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import MedicalKnowledgeRetriever, WeaviateMedicalKnowledgeRetriever


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
    return PreRetriever()


@lru_cache
def get_medical_knowledge_retriever() -> MedicalKnowledgeRetriever:
    settings = get_settings()
    return WeaviateMedicalKnowledgeRetriever(
        embedding_service_url=settings.embedding_service_url,
        embedding_timeout=settings.embedding_timeout,
        embedding_dimension=settings.embedding_dimension,
        weaviate_http_host=settings.weaviate_http_host,
        weaviate_http_port=settings.weaviate_http_port,
        weaviate_grpc_host=settings.weaviate_grpc_host,
        weaviate_grpc_port=settings.weaviate_grpc_port,
        collection_name=settings.weaviate_collection,
    )


@lru_cache
def get_post_retriever() -> PostRetriever:
    settings = get_settings()
    return PostRetriever(max_context_chars=settings.rag_max_context_chars)


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    settings = get_settings()
    return RagPipeline(
        pre_retriever=get_pre_retriever(),
        retriever=get_medical_knowledge_retriever(),
        post_retriever=get_post_retriever(),
        retrieval_limit=settings.rag_top_k,
    )
