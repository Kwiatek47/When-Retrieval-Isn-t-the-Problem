import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_llm_provider, get_medical_knowledge_retriever, get_rag_pipeline
from app.core.config import Settings, get_settings
from app.providers.base import LLMProvider, ProviderError, ProviderUnavailableError
from app.rag.answer_extraction import extract_answer_content
from app.rag.answer_guardrails import apply_answer_guardrails
from app.rag.answer_quality import evaluate_answer_quality
from app.rag.citation_validation import normalize_citation_format, validate_citations
from app.rag.models import PreRetrievalResult, RetrievedDocument
from app.rag.pipeline import RagPipeline
from app.rag.retrieval import MedicalKnowledgeRetriever
from app.schemas import ChatMessage, ChatRequest, ChatResponse, SearchResponse, SearchResult


router = APIRouter(prefix="/api", tags=["chat"])
search_router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    rag_pipeline: Annotated[RagPipeline, Depends(get_rag_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    request_started_at = perf_counter()
    try:
        rag_result = await rag_pipeline.run(messages=request.messages, system_prompt=settings.system_prompt)
        rag_done_at = perf_counter()
        if rag_result.retrieval and rag_result.retrieval.status in {"no_sources", "low_evidence"}:
            refusal_content = (
                "Baza wiedzy nie zwróciła wystarczająco mocnych źródeł dla tego pytania, "
                "więc nie mogę udzielić odpowiedzi opartej na wiarygodnych cytowanych danych. "
                "Skonsultuj decyzje medyczne z wykwalifikowanym lekarzem."
            )
            if rag_result.citations:
                citation_labels = " ".join(f"[{citation.id}]" for citation in rag_result.citations)
                refusal_content = (
                    f"{refusal_content}\n\n"
                    f"Znalezione źródła oznaczono jako niewystarczające dla bezpiecznej odpowiedzi: {citation_labels}"
                )
            logger.info(
                "chat_request timing rag_total=%.3fs llm_total=0.000s total=%.3fs model=%s "
                "retrieval_status=%s documents=%d",
                rag_done_at - request_started_at,
                perf_counter() - request_started_at,
                request.model,
                rag_result.retrieval.status,
                rag_result.retrieval.documents_count,
            )
            return ChatResponse(
                model=request.model,
                message=ChatMessage(
                    role="assistant",
                    content=refusal_content,
                ),
                done=True,
                citations=rag_result.citations,
                retrieval=rag_result.retrieval,
                citation_validation=validate_citations(refusal_content, rag_result.citations),
                evidence_conflicts=rag_result.evidence_conflicts,
                answer_quality=evaluate_answer_quality(
                    "",
                    rag_result.source_documents,
                    method=settings.answer_quality_method,
                    model_name=settings.answer_quality_model_name,
                    similarity_threshold=settings.answer_quality_similarity_threshold,
                ),
            )

        llm_response = await llm_provider.chat(
            model=request.model,
            messages=rag_result.messages,
            temperature=request.temperature,
        )
        llm_done_at = perf_counter()
        answer_content = extract_answer_content(llm_response.message.content)
        if not answer_content:
            logger.warning("LLM response had no extractable answer content; using fallback response.")
            citation_labels = " ".join(f"[{citation.id}]" for citation in rag_result.citations[:1])
            citation_suffix = f" {citation_labels}" if citation_labels else ""
            answer_content = (
                "Nie udało się wyodrębnić poprawnej odpowiedzi z modelu. "
                "Znaleziono źródła w bazie wiedzy, ale model nie wygenerował użytecznej odpowiedzi. "
                f"Spróbuj ponowić pytanie lub zawęzić je do konkretnego problemu medycznego.{citation_suffix}"
            )
        answer_content = apply_answer_guardrails(
            normalize_citation_format(answer_content),
            rag_result.source_documents,
        )
        answer_message = ChatMessage(role=llm_response.message.role, content=answer_content)
        citation_validation = validate_citations(answer_content, rag_result.citations)
        answer_quality = evaluate_answer_quality(
            answer_content,
            rag_result.source_documents,
            method=settings.answer_quality_method,
            model_name=settings.answer_quality_model_name,
            similarity_threshold=settings.answer_quality_similarity_threshold,
        )
        logger.info(
            "chat_request timing rag_total=%.3fs llm_total=%.3fs total=%.3fs model=%s "
            "retrieval_status=%s documents=%d citation_validation=%s groundedness=%s hallucination_rate=%s",
            rag_done_at - request_started_at,
            llm_done_at - rag_done_at,
            llm_done_at - request_started_at,
            llm_response.model,
            rag_result.retrieval.status if rag_result.retrieval else "none",
            rag_result.retrieval.documents_count if rag_result.retrieval else 0,
            citation_validation.passed,
            answer_quality.groundedness,
            answer_quality.hallucination_rate,
        )
        return ChatResponse(
            model=llm_response.model,
            message=answer_message,
            done=llm_response.done,
            citations=rag_result.citations,
            retrieval=rag_result.retrieval,
            citation_validation=citation_validation,
            evidence_conflicts=rag_result.evidence_conflicts,
            answer_quality=answer_quality,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.get("/search", response_model=SearchResponse)
async def api_search(
    q: Annotated[str, Query(min_length=1)],
    retriever: Annotated[MedicalKnowledgeRetriever, Depends(get_medical_knowledge_retriever)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SearchResponse:
    return await _search(q=q, top_k=top_k, retriever=retriever)


@search_router.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1)],
    retriever: Annotated[MedicalKnowledgeRetriever, Depends(get_medical_knowledge_retriever)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SearchResponse:
    return await _search(q=q, top_k=top_k, retriever=retriever)


async def _search(
    *,
    q: str,
    top_k: int,
    retriever: MedicalKnowledgeRetriever,
) -> SearchResponse:
    normalized_query = " ".join(q.split())
    if not normalized_query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search query cannot be blank.")
    retrieval = await retriever.retrieve(
        PreRetrievalResult(
            original_query=q,
            normalized_query=normalized_query,
            search_queries=[normalized_query],
            requires_retrieval=True,
        ),
        limit=top_k,
    )
    return SearchResponse(
        query=normalized_query,
        top_k=top_k,
        provider=retrieval.provider,
        results=[_search_result(document) for document in retrieval.documents],
    )


def _search_result(document: RetrievedDocument) -> SearchResult:
    metadata = {str(key): str(value) for key, value in document.metadata.items()}
    return SearchResult(
        chunk_id=metadata.get("chunkId") or metadata.get("chunk_id") or metadata.get("documentId") or document.id,
        score=document.score,
        pmid=metadata.get("pmid"),
        title=document.title,
        text=document.content,
        doi=metadata.get("doi"),
        year=_optional_int(metadata.get("year")),
        source=document.source,
        url=metadata.get("url"),
        metadata=metadata,
    )


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None
