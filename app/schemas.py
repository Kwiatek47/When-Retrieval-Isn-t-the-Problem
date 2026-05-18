from typing import Literal
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default=get_settings().default_model, min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class Citation(BaseModel):
    id: str
    title: str
    source: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalInfo(BaseModel):
    enabled: bool
    status: Literal["skipped", "grounded", "no_sources", "low_evidence"]
    provider: str
    query: str
    documents_count: int


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    pmid: Optional[str] = None
    title: str
    text: str
    doi: Optional[str] = None
    year: Optional[int] = None
    source: str
    url: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    provider: str
    results: list[SearchResult] = Field(default_factory=list)


class CitationValidation(BaseModel):
    passed: bool
    cited_ids: list[str] = Field(default_factory=list)
    missing_citation_ids: list[str] = Field(default_factory=list)
    unused_citation_ids: list[str] = Field(default_factory=list)
    has_required_citation: bool = True
    issues: list[str] = Field(default_factory=list)


class EvidenceConflictPair(BaseModel):
    source_ids: list[str]
    shared_terms: list[str] = Field(default_factory=list)
    reason: str
    newer_source_id: Optional[str] = None


class EvidenceConflictInfo(BaseModel):
    detected: bool
    strategy: Literal["none", "conflicting_evidence_flag"] = "none"
    conflict_count: int = 0
    pairs: list[EvidenceConflictPair] = Field(default_factory=list)
    instruction: str = ""


class AnswerQuality(BaseModel):
    groundedness: Optional[float] = None
    hallucination_rate: Optional[float] = None
    unsupported_statements: list[str] = Field(default_factory=list)
    evaluated_statements_count: int = 0
    average_similarity: Optional[float] = None
    method: str = "semantic_similarity"


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool
    citations: list[Citation] = Field(default_factory=list)
    retrieval: Optional[RetrievalInfo] = None
    citation_validation: Optional[CitationValidation] = None
    evidence_conflicts: Optional[EvidenceConflictInfo] = None
    answer_quality: Optional[AnswerQuality] = None
