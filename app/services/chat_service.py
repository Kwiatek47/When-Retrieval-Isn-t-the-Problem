from __future__ import annotations

import json
import re
from time import perf_counter

from app.core.config import Settings
from app.providers.base import LLMProvider, ProviderError
from app.rag.answer_contract import enforce_yes_no_maybe_contract, extract_yes_no_maybe_label
from app.rag.answer_extraction import extract_answer_content
from app.rag.answer_quality import evaluate_answer_quality
from app.rag.citation_validation import normalize_citation_format, validate_citations
from app.rag.models import PostRetrievalResult, RetrievedDocument
from app.schemas import ChatMessage, ChatRequest, ChatResponse
from app.services.telemetry_service import TelemetryLogger


def build_messages(messages: list[ChatMessage], system_prompt: str) -> list[ChatMessage]:
    user_visible_messages = [message for message in messages if message.role != "system"]
    return [ChatMessage(role="system", content=system_prompt), *user_visible_messages]


def build_refusal_response(
    *,
    request: ChatRequest,
    request_id: str,
    prompt_version: str,
    timestamp: str,
    request_started_at: float,
    rag_result: PostRetrievalResult,
    yes_no_maybe_task: bool,
    settings: Settings,
    telemetry_logger: TelemetryLogger,
) -> ChatResponse:
    refusal_content = low_evidence_refusal(
        request.messages,
        rag_result.citations,
        yes_no_maybe_task=yes_no_maybe_task,
    )
    latency_ms = int((perf_counter() - request_started_at) * 1000)
    citation_validation = validate_citations(refusal_content, rag_result.citations)
    answer_quality = evaluate_answer_quality(
        refusal_content,
        rag_result.source_documents,
        method=settings.answer_quality_method,
        model_name=settings.answer_quality_model_name,
        similarity_threshold=settings.answer_quality_similarity_threshold,
    )
    status = rag_result.retrieval.status if rag_result.retrieval else "skipped"
    response = ChatResponse(
        model=request.model,
        message=ChatMessage(role="assistant", content=refusal_content),
        done=True,
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
            "status": status,
            "model": request.model,
            "temperature": request.temperature,
            "prompt_version": prompt_version,
            "messages": [message.model_dump() for message in request.messages],
            "response": response.model_dump(),
        }
    )
    return response


def low_evidence_refusal(
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
    if prefer_english(messages):
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
        "Baza wiedzy nie zwrocila wystarczajaco mocnych zrodel dla tego pytania, "
        "wiec nie moge udzielic odpowiedzi opartej na wiarygodnych cytowanych danych. "
        "Skonsultuj decyzje medyczne z wykwalifikowanym lekarzem."
    )
    if citation_labels:
        content = (
            f"{content}\n\n"
            f"Znalezione zrodla oznaczono jako niewystarczajace dla bezpiecznej odpowiedzi: {citation_labels}"
        )
    return content


def low_evidence_warning(messages: list[ChatMessage], citations: list) -> str:
    citation_labels = " ".join(f"[{citation.id}]" for citation in citations)
    if prefer_english(messages):
        warning = "Warning: retrieved sources were marked as low-evidence, so this answer may be incomplete or uncertain."
        if citation_labels:
            warning = f"{warning} Sources: {citation_labels}"
        return warning
    warning = "Ostrzezenie: znalezione zrodla oznaczono jako low-evidence, wiec odpowiedz moze byc niepelna lub niepewna."
    if citation_labels:
        warning = f"{warning} Zrodla: {citation_labels}"
    return warning


def prefer_english(messages: list[ChatMessage]) -> bool:
    user_text = " ".join(message.content for message in messages if message.role == "user").lower()
    if not user_text:
        return False
    english_markers = {"answer", "what", "which", "how", "does", "do", "is", "are", "known", "compare", "risk", "used"}
    polish_markers = {"jak", "jakie", "czy", "jest", "stosuje", "leki", "chorobie", "ryzyko"}
    words = set(user_text.split())
    return bool(words & english_markers) and not bool(words & polish_markers)


def quality_gate_failed(citation_validation, answer_quality, settings: Settings) -> bool:
    if not settings.rag_answer_quality_gate_enabled:
        return False
    if not settings.rag_extractive_fallback_enabled:
        return False
    if not citation_validation.passed:
        return True
    if answer_quality.hallucination_rate is None:
        return False
    return answer_quality.hallucination_rate > settings.rag_answer_quality_max_hallucination_rate


async def repair_yes_no_maybe_answer(
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
                f"Retrieved source excerpts:\n{decision_source_block(source_documents)}\n\n"
                f"Draft answer for reference only:\n{draft_answer}\n\n"
                "Return JSON now."
            ),
        ),
    ]
    try:
        response = await llm_provider.chat(model=model, messages=decision_messages, temperature=0.0)
    except ProviderError:
        return enforce_yes_no_maybe_contract(draft_answer, source_documents)

    decision = parse_yes_no_maybe_decision(response.message.content)
    if decision is None:
        return enforce_yes_no_maybe_contract(draft_answer, source_documents)

    label, evidence = decision
    return enforce_yes_no_maybe_contract(
        f"Answer: {label}\nEvidence: {normalize_citation_format(evidence)}",
        source_documents,
    )


def decision_source_block(source_documents: list[RetrievedDocument]) -> str:
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


def parse_yes_no_maybe_decision(content: str) -> tuple[str, str] | None:
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


def extractive_fallback_answer(
    messages: list[ChatMessage],
    source_documents: list[RetrievedDocument],
) -> str:
    if not source_documents:
        return ""

    query_text = " ".join(message.content for message in messages if message.role == "user")
    query_terms = content_tokens(query_text)
    sentences = []
    for index, document in enumerate(source_documents[:2], start=1):
        sentence_limit = 2 if index == 1 else 1
        for selected_sentence in best_source_sentences(document.content, query_terms, limit=sentence_limit):
            sentences.append(append_source_citation(selected_sentence, f"S{index}"))

    if not sentences:
        return ""

    if prefer_english(messages):
        return (
            "Retrieved evidence summary: "
            + " ".join(sentences)
            + " Consult a qualified clinician for medical decisions."
        )
    return (
        "Podsumowanie danych ze zrodel: "
        + " ".join(sentences)
        + " Skonsultuj decyzje medyczne z wykwalifikowanym lekarzem."
    )


def best_source_sentences(content: str, query_terms: set[str], *, limit: int) -> list[str]:
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
        key=lambda sentence: len(content_tokens(sentence) & query_terms) / max(len(query_terms), 1),
        reverse=True,
    )
    return ranked[:limit]


def append_source_citation(sentence: str, citation_id: str) -> str:
    match = re.search(r"([.!?])$", sentence)
    if match:
        return f"{sentence[: match.start()].rstrip()} [{citation_id}]{match.group(1)}"
    return f"{sentence} [{citation_id}]."


def content_tokens(text: str) -> set[str]:
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
