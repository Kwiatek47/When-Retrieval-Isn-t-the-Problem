import logging
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import (
    get_evidence_judge,
    get_llm_provider,
    get_medical_knowledge_retriever,
    get_ollama_provider,
    get_post_retriever,
    get_pre_retriever,
    get_rag_pipeline,
    get_telemetry_logger,
)
from app.core.config import Settings, get_settings
from app.core.prompt_registry import resolve_prompt
from app.providers.base import LLMProvider, ProviderError, ProviderUnavailableError
from app.providers.ollama import OllamaProvider
from app.rag.answer_contract import (
    enforce_yes_no_maybe_contract,
    is_yes_no_maybe_task,
)
from app.rag.answer_extraction import extract_answer_content
from app.rag.answer_guardrails import apply_answer_guardrails, repair_missing_citations
from app.rag.answer_quality import evaluate_answer_quality
from app.rag.citation_validation import normalize_citation_format, validate_citations
from app.rag.evidence_judge import (
    EvidenceJudge,
    answer_from_evidence_decision,
    build_evidence_decision_block,
)
from app.rag.models import PostRetrievalResult, RetrievedDocument, RetrievalResult
from app.rag.pipeline import RagPipeline
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import MedicalKnowledgeRetriever
from app.services.chat_service import (
    build_refusal_response,
    extractive_fallback_answer,
    low_evidence_warning,
    quality_gate_failed,
    repair_yes_no_maybe_answer,
)
from app.services.telemetry_service import TelemetryLogger
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EvidenceDecisionInfo,
    RagTraceDocument,
    RagTraceRequest,
    RagTraceResponse,
    SearchResponse,
    SearchResult,
)


router = APIRouter(prefix="/api", tags=["chat"])
search_router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health(
    ollama: Annotated[OllamaProvider, Depends(get_ollama_provider)],
) -> dict:
    try:
        tags = await ollama.ping()
        return {"status": "ok", "ollama": tags}
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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    rag_pipeline: Annotated[RagPipeline, Depends(get_rag_pipeline)],
    evidence_judge: Annotated[EvidenceJudge, Depends(get_evidence_judge)],
    settings: Annotated[Settings, Depends(get_settings)],
    telemetry_logger: Annotated[TelemetryLogger, Depends(get_telemetry_logger)],
) -> ChatResponse:
    request_id = uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    prompt_version, system_prompt = resolve_prompt(
        requested_version=request.prompt_version,
        active_version=settings.active_prompt_version,
        fallback_prompt=settings.system_prompt,
    )
    request_started_at = perf_counter()

    try:
        rag_result = await rag_pipeline.run(messages=request.messages, system_prompt=system_prompt)
        rag_done_at = perf_counter()
        yes_no_maybe_task = is_yes_no_maybe_task(request.messages)
        low_evidence_non_blocking = (
            bool(rag_result.retrieval)
            and rag_result.retrieval.status == "low_evidence"
            and not settings.rag_refuse_on_low_evidence
        )
        if rag_result.pre_retrieval is not None:
            evidence_decision = await evidence_judge.judge(
                messages=request.messages,
                pre_retrieval=rag_result.pre_retrieval,
                source_documents=rag_result.source_documents,
                retrieval_status=rag_result.retrieval.status if rag_result.retrieval else "skipped",
                llm_provider=llm_provider,
                model=settings.rag_evidence_judge_model or request.model,
            )
            rag_result = _with_evidence_decision(rag_result, evidence_decision)
        should_refuse = (
            bool(rag_result.retrieval)
            and (
                rag_result.retrieval.status == "no_sources"
                or (rag_result.retrieval.status == "low_evidence" and settings.rag_refuse_on_low_evidence)
            )
        )
        if should_refuse:
            logger.info(
                "chat_request timing rag_total=%.3fs llm_total=0.000s total=%.3fs model=%s "
                "retrieval_status=%s documents=%d",
                rag_done_at - request_started_at,
                perf_counter() - request_started_at,
                request.model,
                rag_result.retrieval.status,
                rag_result.retrieval.documents_count,
            )
            return build_refusal_response(
                request=request,
                request_id=request_id,
                prompt_version=prompt_version,
                timestamp=timestamp,
                request_started_at=request_started_at,
                rag_result=rag_result,
                yes_no_maybe_task=yes_no_maybe_task,
                settings=settings,
                telemetry_logger=telemetry_logger,
            )

        if yes_no_maybe_task:
            judged_answer = answer_from_evidence_decision(rag_result.evidence_decision, rag_result.source_documents)
            if judged_answer:
                answer_content = enforce_yes_no_maybe_contract(judged_answer, rag_result.source_documents)
                answer_content = apply_answer_guardrails(
                    normalize_citation_format(answer_content),
                    rag_result.source_documents,
                )
                if settings.rag_citation_repair_enabled:
                    answer_content = repair_missing_citations(answer_content, rag_result.source_documents)
                answer_content = enforce_yes_no_maybe_contract(answer_content, rag_result.source_documents)
                if low_evidence_non_blocking and not yes_no_maybe_task:
                    answer_content = f"{answer_content}\n\n{low_evidence_warning(request.messages, rag_result.citations)}"
                citation_validation = validate_citations(answer_content, rag_result.citations)
                answer_quality = evaluate_answer_quality(
                    answer_content,
                    rag_result.source_documents,
                    method=settings.answer_quality_method,
                    model_name=settings.answer_quality_model_name,
                    similarity_threshold=settings.answer_quality_similarity_threshold,
                )
                logger.info(
                    "chat_request timing rag_total=%.3fs llm_total=0.000s total=%.3fs model=%s "
                    "retrieval_status=%s documents=%d evidence_judge=%s citation_validation=%s",
                    rag_done_at - request_started_at,
                    perf_counter() - request_started_at,
                    request.model,
                    rag_result.retrieval.status if rag_result.retrieval else "none",
                    rag_result.retrieval.documents_count if rag_result.retrieval else 0,
                    rag_result.evidence_decision.status if rag_result.evidence_decision else "none",
                    citation_validation.passed,
                )
                return ChatResponse(
                    model=request.model,
                    message=ChatMessage(role="assistant", content=answer_content),
                    done=True,
                    citations=rag_result.citations,
                    retrieval=rag_result.retrieval,
                    evidence_decision=rag_result.evidence_decision,
                    citation_validation=citation_validation,
                    evidence_conflicts=rag_result.evidence_conflicts,
                    answer_quality=answer_quality,
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
        if settings.rag_citation_repair_enabled:
            answer_content = repair_missing_citations(answer_content, rag_result.source_documents)
        if yes_no_maybe_task:
            answer_content = await repair_yes_no_maybe_answer(
                llm_provider=llm_provider,
                model=request.model,
                messages=request.messages,
                draft_answer=answer_content,
                source_documents=rag_result.source_documents,
            )
        if low_evidence_non_blocking and not yes_no_maybe_task:
            answer_content = f"{answer_content}\n\n{low_evidence_warning(request.messages, rag_result.citations)}"
        citation_validation = validate_citations(answer_content, rag_result.citations)
        answer_quality = evaluate_answer_quality(
            answer_content,
            rag_result.source_documents,
            method=settings.answer_quality_method,
            model_name=settings.answer_quality_model_name,
            similarity_threshold=settings.answer_quality_similarity_threshold,
        )
        if quality_gate_failed(citation_validation, answer_quality, settings):
            fallback_content = extractive_fallback_answer(request.messages, rag_result.source_documents)
            if fallback_content:
                if yes_no_maybe_task:
                    fallback_content = enforce_yes_no_maybe_contract(fallback_content, rag_result.source_documents)
                answer_content = fallback_content
                citation_validation = validate_citations(answer_content, rag_result.citations)
                answer_quality = evaluate_answer_quality(
                    answer_content,
                    rag_result.source_documents,
                    method=settings.answer_quality_method,
                    model_name=settings.answer_quality_model_name,
                    similarity_threshold=settings.answer_quality_similarity_threshold,
                )
        answer_message = ChatMessage(role=llm_response.message.role, content=answer_content)
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
        latency_ms = int((perf_counter() - request_started_at) * 1000)
        response = ChatResponse(
            model=llm_response.model,
            message=answer_message,
            done=llm_response.done,
            request_id=request_id,
            prompt_version=prompt_version,
            timestamp=timestamp,
            latency_ms=latency_ms,
            citations=rag_result.citations,
            retrieval=rag_result.retrieval,
            evidence_decision=rag_result.evidence_decision,
            citation_validation=citation_validation,
            evidence_conflicts=rag_result.evidence_conflicts,
            answer_quality=answer_quality,
        )
        telemetry_logger.log_chat_event(
            {
                "request_id": request_id,
                "status": "ok",
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": prompt_version,
                "messages": [message.model_dump() for message in request.messages],
                "response": response.model_dump(),
            }
        )
        return response
    except ProviderUnavailableError as exc:
        telemetry_logger.log_chat_event(
            {
                "request_id": request_id,
                "status": "provider_unavailable",
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": prompt_version,
                "messages": [message.model_dump() for message in request.messages],
                "error": str(exc),
            }
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        telemetry_logger.log_chat_event(
            {
                "request_id": request_id,
                "status": "provider_error",
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": prompt_version,
                "messages": [message.model_dump() for message in request.messages],
                "error": str(exc),
            }
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/rag/trace", response_model=RagTraceResponse)
async def rag_trace(
    request: RagTraceRequest,
    pre_retriever: Annotated[PreRetriever, Depends(get_pre_retriever)],
    retriever: Annotated[MedicalKnowledgeRetriever, Depends(get_medical_knowledge_retriever)],
    post_retriever: Annotated[PostRetriever, Depends(get_post_retriever)],
    evidence_judge: Annotated[EvidenceJudge, Depends(get_evidence_judge)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagTraceResponse:
    pre_retrieval = await pre_retriever.prepare(request.messages)
    candidate_k = request.candidate_k or max(settings.rag_candidate_k, settings.rag_top_k)
    top_k = request.top_k or settings.rag_top_k
    if pre_retrieval.requires_retrieval:
        retrieval = await retriever.retrieve(pre_retrieval, limit=candidate_k)
    else:
        retrieval = RetrievalResult(query=pre_retrieval, documents=[], provider="skipped")

    post_result = post_retriever.assemble(
        messages=request.messages,
        system_prompt=settings.system_prompt,
        pre_retrieval=pre_retrieval,
        retrieval=retrieval,
        final_documents_limit=top_k,
    )
    evidence_decision = await evidence_judge.judge(
        messages=request.messages,
        pre_retrieval=pre_retrieval,
        source_documents=post_result.source_documents,
        retrieval_status=post_result.retrieval.status,
        llm_provider=None,
        model="",
    )

    rrf_documents = _debug_documents(retrieval, "rrf_documents") or sorted(
        retrieval.documents,
        key=lambda document: _metadata_float(document, "preMetadataBoostScore", "rrfScore", fallback=document.score),
        reverse=True,
    )
    metadata_boosted_documents = _debug_documents(retrieval, "metadata_boosted_documents") or retrieval.documents
    return RagTraceResponse(
        original_query=pre_retrieval.original_query,
        normalized_query=pre_retrieval.normalized_query,
        search_queries=pre_retrieval.search_queries,
        intent=pre_retrieval.intent,
        filters=pre_retrieval.filters,
        preferred_publication_types=pre_retrieval.preferred_publication_types,
        notes=pre_retrieval.notes,
        provider=retrieval.provider,
        rrf_candidates=_trace_documents(rrf_documents),
        metadata_boosted_candidates=_trace_documents(metadata_boosted_documents),
        final_documents=_trace_documents(post_result.source_documents),
        retrieval=post_result.retrieval,
        evidence_decision=evidence_decision,
        context_preview=_context_preview(post_result.messages[0].content),
    )


@router.get("/search", response_model=SearchResponse)
async def api_search(
    q: Annotated[str, Query(min_length=1)],
    retriever: Annotated[MedicalKnowledgeRetriever, Depends(get_medical_knowledge_retriever)],
    pre_retriever: Annotated[PreRetriever, Depends(get_pre_retriever)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SearchResponse:
    return await _search(q=q, top_k=top_k, retriever=retriever, pre_retriever=pre_retriever)


@search_router.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1)],
    retriever: Annotated[MedicalKnowledgeRetriever, Depends(get_medical_knowledge_retriever)],
    pre_retriever: Annotated[PreRetriever, Depends(get_pre_retriever)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SearchResponse:
    return await _search(q=q, top_k=top_k, retriever=retriever, pre_retriever=pre_retriever)


async def _search(
    *,
    q: str,
    top_k: int,
    retriever: MedicalKnowledgeRetriever,
    pre_retriever: PreRetriever,
) -> SearchResponse:
    normalized_query = " ".join(q.split())
    if not normalized_query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search query cannot be blank.")
    pre_retrieval = await pre_retriever.prepare(
        [ChatMessage(role="user", content=q)],
        allow_rewrite=False,
    )
    if not pre_retrieval.requires_retrieval:
        pre_retrieval = replace(
            pre_retrieval,
            requires_retrieval=True,
            notes=[*pre_retrieval.notes, "Retrieval forced for search endpoint."],
        )
    retrieval = await retriever.retrieve(
        pre_retrieval,
        limit=top_k,
    )
    return SearchResponse(
        query=pre_retrieval.normalized_query,
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


def _with_evidence_decision(
    rag_result: PostRetrievalResult,
    evidence_decision: EvidenceDecisionInfo,
) -> PostRetrievalResult:
    decision_block = build_evidence_decision_block(evidence_decision)
    if not decision_block or not rag_result.messages:
        return replace(rag_result, evidence_decision=evidence_decision)

    messages = list(rag_result.messages)
    system_message = messages[0]
    messages[0] = ChatMessage(
        role=system_message.role,
        content=f"{system_message.content}\n\n{decision_block}",
    )
    return replace(rag_result, messages=messages, evidence_decision=evidence_decision)


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _trace_documents(documents: list[RetrievedDocument]) -> list[RagTraceDocument]:
    return [
        RagTraceDocument(
            rank=index,
            id=document.id,
            title=document.title,
            source=document.source,
            score=document.score,
            content_preview=document.content[:500],
            metadata={str(key): value for key, value in document.metadata.items()},
        )
        for index, document in enumerate(documents, start=1)
    ]


def _debug_documents(retrieval: RetrievalResult, key: str) -> list[RetrievedDocument]:
    value = retrieval.debug.get(key)
    if isinstance(value, list) and all(isinstance(document, RetrievedDocument) for document in value):
        return value
    return []


def _metadata_float(document: RetrievedDocument, *keys: str, fallback: float = 0.0) -> float:
    for key in keys:
        value = document.metadata.get(key)
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(fallback or 0.0)


def _context_preview(system_prompt: str) -> str:
    marker = "MEDICAL_KNOWLEDGE_BASE:"
    if marker not in system_prompt:
        return system_prompt[:4000]
    return system_prompt.split(marker, 1)[1].strip()[:4000]

