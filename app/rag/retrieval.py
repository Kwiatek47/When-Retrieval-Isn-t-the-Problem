from typing import Protocol

from app.rag.models import PreRetrievalResult, RetrievalResult


class MedicalKnowledgeRetriever(Protocol):
    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        ...


class EmptyMedicalKnowledgeRetriever:
    """Temporary retriever used until the VectorDB implementation is connected."""

    provider_name = "empty"

    async def retrieve(self, query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
        return RetrievalResult(query=query, documents=[], provider=self.provider_name)

