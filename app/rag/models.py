from dataclasses import dataclass, field
from typing import Any

from app.schemas import ChatMessage, Citation, RetrievalInfo


@dataclass(frozen=True)
class PreRetrievalResult:
    original_query: str
    normalized_query: str
    search_queries: list[str]
    requires_retrieval: bool
    filters: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    title: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    query: PreRetrievalResult
    documents: list[RetrievedDocument]
    provider: str


@dataclass(frozen=True)
class PostRetrievalResult:
    messages: list[ChatMessage]
    citations: list[Citation]
    retrieval: RetrievalInfo

