import json
import logging
from dataclasses import replace
import re
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import (
    get_llm_provider,
    get_medical_knowledge_retriever,
    get_post_retriever,
    get_pre_retriever,
    get_rag_pipeline,
)
from app.core.config import Settings, get_settings
from app.providers.base import LLMProvider, ProviderError, ProviderUnavailableError
from app.rag.answer_contract import (
    enforce_yes_no_maybe_contract,
    extract_yes_no_maybe_label,
    is_yes_no_maybe_task,
)
from app.rag.answer_extraction import extract_answer_content
from app.rag.answer_guardrails import apply_answer_guardrails, repair_missing_citations
from app.rag.answer_quality import evaluate_answer_quality
from app.rag.citation_validation import normalize_citation_format, validate_citations
from app.rag.models import RetrievedDocument, RetrievalResult
from app.rag.pipeline import RagPipeline
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import MedicalKnowledgeRetriever
from app.schemas import (
    AnswerQuality,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CitationValidation,
    RagTraceDocument,
    RagTraceRequest,
    RagTraceResponse,
    SearchResponse,
    SearchResult,
)


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
        yes_no_maybe_task = is_yes_no_maybe_task(request.messages)
        if rag_result.retrieval and rag_result.retrieval.status in {"no_sources", "low_evidence"}:
            refusal_content = _low_evidence_refusal(
                request.messages,
                rag_result.citations,
                yes_no_maybe_task=yes_no_maybe_task,
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
        if settings.rag_citation_repair_enabled:
            answer_content = repair_missing_citations(answer_content, rag_result.source_documents)
        if yes_no_maybe_task:
            answer_content = await _repair_yes_no_maybe_answer(
                llm_provider=llm_provider,
                model=request.model,
                messages=request.messages,
                draft_answer=answer_content,
                source_documents=rag_result.source_documents,
            )
        citation_validation = validate_citations(answer_content, rag_result.citations)
        answer_quality = evaluate_answer_quality(
            answer_content,
            rag_result.source_documents,
            method=settings.answer_quality_method,
            model_name=settings.answer_quality_model_name,
            similarity_threshold=settings.answer_quality_similarity_threshold,
        )
        if _quality_gate_failed(citation_validation, answer_quality, settings):
            fallback_content = _extractive_fallback_answer(request.messages, rag_result.source_documents)
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


@router.post("/rag/trace", response_model=RagTraceResponse)
async def rag_trace(
    request: RagTraceRequest,
    pre_retriever: Annotated[PreRetriever, Depends(get_pre_retriever)],
    retriever: Annotated[MedicalKnowledgeRetriever, Depends(get_medical_knowledge_retriever)],
    post_retriever: Annotated[PostRetriever, Depends(get_post_retriever)],
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


def _low_evidence_refusal(
    messages: list[ChatMessage],
    citations: list,
    *,
    yes_no_maybe_task: bool = False,
) -> str:
    citation_labels = " ".join(f"[{citation.id}]" for citation in citations)
    if yes_no_maybe_task:
        cited_suffix = f" {citation_labels}" if citation_labels else ""
        return (
            "Answer: maybe\n"
            "Evidence: The retrieved evidence was not strong enough to support a reliable yes or no answer"
            f"{cited_suffix}. Consult a qualified clinician for medical decisions."
        )
    if _prefer_english(messages):
        content = (
            "The knowledge base did not return sources strong enough for this question, "
            "so I cannot provide a reliable cited medical answer. "
            "Consult a qualified clinician for medical decisions."
        )
        if citation_labels:
            content = (
                f"{content}\n\n"
                f"The retrieved sources were marked as insufficient for a safe answer: {citation_labels}"
            )
        return content

    content = (
        "Baza wiedzy nie zwróciła wystarczająco mocnych źródeł dla tego pytania, "
        "więc nie mogę udzielić odpowiedzi opartej na wiarygodnych cytowanych danych. "
        "Skonsultuj decyzje medyczne z wykwalifikowanym lekarzem."
    )
    if citation_labels:
        content = (
            f"{content}\n\n"
            f"Znalezione źródła oznaczono jako niewystarczające dla bezpiecznej odpowiedzi: {citation_labels}"
        )
    return content


def _prefer_english(messages: list[ChatMessage]) -> bool:
    user_text = " ".join(message.content for message in messages if message.role == "user").lower()
    if not user_text:
        return False
    english_markers = {"answer", "what", "which", "how", "does", "do", "is", "are", "known", "compare", "risk", "used"}
    polish_markers = {"jak", "jakie", "czy", "jest", "stosuje", "leki", "chorobie", "ryzyko"}
    words = set(user_text.split())
    return bool(words & english_markers) and not bool(words & polish_markers)


def _quality_gate_failed(
    citation_validation: CitationValidation,
    answer_quality: AnswerQuality,
    settings: Settings,
) -> bool:
    if not settings.rag_answer_quality_gate_enabled:
        return False
    if not settings.rag_extractive_fallback_enabled:
        return False
    if not citation_validation.passed:
        return True
    if answer_quality.hallucination_rate is None:
        return False
    return answer_quality.hallucination_rate > settings.rag_answer_quality_max_hallucination_rate


async def _repair_yes_no_maybe_answer(
    *,
    llm_provider: LLMProvider,
    model: str,
    messages: list[ChatMessage],
    draft_answer: str,
    source_documents: list[RetrievedDocument],
) -> str:
    if not source_documents:
        return enforce_yes_no_maybe_contract(draft_answer, source_documents)

    user_question = "\n".join(message.content for message in messages if message.role == "user").strip()
    decision_messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a biomedical evidence classification layer for a PubMedQA-style task. "
                "Use only the supplied retrieved source excerpts. Treat the draft answer as non-binding and ignore it "
                "when it conflicts with the source excerpts. "
                "Output only compact JSON with keys `answer` and `evidence`. "
                "`answer` must be exactly one of `yes`, `no`, or `maybe`. "
                "This is evidence classification, not patient-specific clinical advice; do not default to `maybe` just "
                "because there is one abstract, a small study, or cautious scientific wording. "
                "Choose `yes` when the abstract/results directionally support the proposition in the question: effect, "
                "association, diagnostic utility, prognostic value, usefulness, feasibility, or superiority. "
                "Choose `no` when the abstract/results directly refute the proposition or report no meaningful effect, "
                "no association, no diagnostic/prognostic value, not enough accuracy, not useful, not reliable, or no "
                "advantage. "
                "Choose `maybe` only when the abstract/results are explicitly inconclusive, mixed, conflicting, indirect, "
                "or do not address the proposition. "
                "The `evidence` value must be one short sentence with source citations like [S1]. "
                "Do not include hidden reasoning or any text outside JSON."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Question:\n{user_question}\n\n"
                f"Retrieved source excerpts:\n{_decision_source_block(source_documents)}\n\n"
                f"Draft answer for reference only:\n{draft_answer}\n\n"
                "Return JSON now."
            ),
        ),
    ]
    try:
        response = await llm_provider.chat(model=model, messages=decision_messages, temperature=0.0)
    except ProviderError:
        logger.warning("yes_no_maybe_decision_repair provider failed; using deterministic contract.", exc_info=True)
        return enforce_yes_no_maybe_contract(draft_answer, source_documents)

    decision = _parse_yes_no_maybe_decision(response.message.content)
    if decision is None:
        return enforce_yes_no_maybe_contract(draft_answer, source_documents)

    label, evidence = decision
    return enforce_yes_no_maybe_contract(
        f"Answer: {label}\nEvidence: {normalize_citation_format(evidence)}",
        source_documents,
    )


def _decision_source_block(source_documents: list[RetrievedDocument]) -> str:
    parts = []
    for index, document in enumerate(source_documents[:3], start=1):
        excerpt = re.sub(r"\s+", " ", document.content).strip()
        if len(excerpt) > 1800:
            excerpt = excerpt[:1800].rsplit(" ", 1)[0].strip()
        parts.append(
            "\n".join(
                [
                    f"[S{index}] {document.title}",
                    f"Source: {document.source}",
                    "Excerpt:",
                    excerpt,
                ]
            )
        )
    return "\n\n".join(parts)


def _parse_yes_no_maybe_decision(content: str) -> tuple[str, str] | None:
    normalized = extract_answer_content(content)
    json_match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            label = str(parsed.get("answer") or "").strip().lower()
            evidence = str(parsed.get("evidence") or "").strip()
            if label in {"yes", "no", "maybe"} and evidence:
                return label, evidence

    label = extract_yes_no_maybe_label(normalized)
    if not label:
        return None
    evidence = re.sub(r"^\s*(?:answer\s*[:\-]\s*)?(?:yes|no|maybe)\b\s*[,.;:\-]*", "", normalized, flags=re.IGNORECASE)
    evidence = re.sub(r"^\s*evidence\s*[:\-]\s*", "", evidence.strip(), flags=re.IGNORECASE)
    return label, evidence.strip() or "The retrieved evidence supports this classification."


def _extractive_fallback_answer(
    messages: list[ChatMessage],
    source_documents: list[RetrievedDocument],
) -> str:
    if not source_documents:
        return ""

    query_text = " ".join(message.content for message in messages if message.role == "user")
    query_terms = _content_tokens(query_text)
    sentences = []
    for index, document in enumerate(source_documents[:2], start=1):
        sentence_limit = 2 if index == 1 else 1
        for selected_sentence in _best_source_sentences(document.content, query_terms, limit=sentence_limit):
            sentences.append(_append_source_citation(selected_sentence, f"S{index}"))

    if not sentences:
        return ""

    if _prefer_english(messages):
        return (
            "Retrieved evidence summary: "
            + " ".join(sentences)
            + " Consult a qualified clinician for medical decisions."
        )
    return (
        "Podsumowanie danych ze źródeł: "
        + " ".join(sentences)
        + " Skonsultuj decyzje medyczne z wykwalifikowanym lekarzem."
    )


def _best_source_sentences(content: str, query_terms: set[str], *, limit: int) -> list[str]:
    candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", content).strip())
        if sentence.strip()
    ]
    candidates = [
        sentence
        for sentence in candidates
        if not sentence.lower().startswith(("real source", "relevant concepts"))
    ]
    if not candidates:
        return []
    if not query_terms:
        return candidates[:limit]
    ranked = sorted(
        candidates,
        key=lambda sentence: len(_content_tokens(sentence) & query_terms) / max(len(query_terms), 1),
        reverse=True,
    )
    return ranked[:limit]


def _append_source_citation(sentence: str, citation_id: str) -> str:
    match = re.search(r"([.!?])$", sentence)
    if match:
        return f"{sentence[: match.start()].rstrip()} [{citation_id}]{match.group(1)}"
    return f"{sentence} [{citation_id}]."


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "about",
        "and",
        "are",
        "can",
        "compare",
        "does",
        "for",
        "from",
        "how",
        "into",
        "known",
        "the",
        "used",
        "what",
        "when",
        "with",
    }
    return {
        token
        for token in re.findall(r"[\w]+", text.lower())
        if len(token) >= 4 and token not in stopwords
    }
