from typing import Any
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
    prompt_version: str | None = Field(default=None, min_length=1)


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


class RagTraceRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    candidate_k: Optional[int] = Field(default=None, ge=1, le=100)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class RagTraceDocument(BaseModel):
    rank: int
    id: str
    title: str
    source: str
    score: float
    content_preview: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceDecisionInfo(BaseModel):
    enabled: bool = True
    status: Literal["skipped", "supported", "refuted", "uncertain", "insufficient"]
    method: str = "rules"
    answer_label: Optional[Literal["yes", "no", "maybe"]] = None
    confidence: Optional[float] = None
    rationale: str = ""
    citations: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RagTraceResponse(BaseModel):
    original_query: str
    normalized_query: str
    search_queries: list[str]
    intent: str
    filters: dict[str, str] = Field(default_factory=dict)
    preferred_publication_types: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provider: str
    rrf_candidates: list[RagTraceDocument] = Field(default_factory=list)
    metadata_boosted_candidates: list[RagTraceDocument] = Field(default_factory=list)
    final_documents: list[RagTraceDocument] = Field(default_factory=list)
    retrieval: RetrievalInfo
    evidence_decision: Optional[EvidenceDecisionInfo] = None
    context_preview: str


class CitationValidation(BaseModel):
    passed: bool
    cited_ids: list[str] = Field(default_factory=list)
    missing_citation_ids: list[str] = Field(default_factory=list)
    unused_citation_ids: list[str] = Field(default_factory=list)
    has_required_citation: bool = True
    claim_count: int = 0
    cited_claims_count: int = 0
    citation_recall: Optional[float] = None
    citation_precision: Optional[float] = None
    uncited_claims: list[str] = Field(default_factory=list)
    shotgun_citation_claims: list[str] = Field(default_factory=list)
    orphan_citations: list[str] = Field(default_factory=list)
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
    request_id: str = ""
    prompt_version: str = ""
    timestamp: str = ""
    latency_ms: int = Field(default=0, ge=0)
    citations: list[Citation] = Field(default_factory=list)
    retrieval: Optional[RetrievalInfo] = None
    evidence_decision: Optional[EvidenceDecisionInfo] = None
    citation_validation: Optional[CitationValidation] = None
    evidence_conflicts: Optional[EvidenceConflictInfo] = None
    answer_quality: Optional[AnswerQuality] = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)
    model: str | None = Field(default=None, min_length=1)
    prompt_version: str | None = Field(default=None, min_length=1)


class FeedbackResponse(BaseModel):
    ok: bool = True
