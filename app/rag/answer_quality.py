from __future__ import annotations

from functools import lru_cache
import re

from app.rag.citation_validation import extract_citation_ids, requires_citation, strip_citation_tags
from app.rag.models import RetrievedDocument
from app.schemas import AnswerQuality


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[\w]+", re.IGNORECASE)
_STOPWORDS = {
    "and",
    "are",
    "bez",
    "dla",
    "jest",
    "lub",
    "nie",
    "oraz",
    "pod",
    "przez",
    "przy",
    "sie",
    "the",
    "with",
}


def evaluate_answer_quality(
    answer: str,
    source_documents: list[RetrievedDocument],
    *,
    method: str = "semantic_similarity",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.45,
    overlap_threshold: float = 0.35,
) -> AnswerQuality:
    source_text_by_id = {
        f"S{index + 1}": document.content
        for index, document in enumerate(source_documents)
    }
    all_context = " ".join(source_text_by_id.values())
    statements = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_PATTERN.split(answer.strip())
        if sentence.strip() and requires_citation(sentence)
    ]
    if not statements:
        return AnswerQuality(
            groundedness=None,
            hallucination_rate=None,
            unsupported_statements=[],
            evaluated_statements_count=0,
            average_similarity=None,
            method=method,
        )

    if method == "semantic_similarity":
        try:
            return _evaluate_semantic_similarity(
                statements=statements,
                source_text_by_id=source_text_by_id,
                all_context=all_context,
                model_name=model_name,
                similarity_threshold=similarity_threshold,
            )
        except Exception:
            return _evaluate_token_overlap(
                statements=statements,
                source_text_by_id=source_text_by_id,
                all_context=all_context,
                overlap_threshold=overlap_threshold,
                method="token_overlap_with_retrieved_context_fallback",
            )

    return _evaluate_token_overlap(
        statements=statements,
        source_text_by_id=source_text_by_id,
        all_context=all_context,
        overlap_threshold=overlap_threshold,
        method="token_overlap_with_retrieved_context",
    )


def _evaluate_semantic_similarity(
    *,
    statements: list[str],
    source_text_by_id: dict[str, str],
    all_context: str,
    model_name: str,
    similarity_threshold: float,
) -> AnswerQuality:
    model = _load_sentence_transformer(model_name)
    unsupported = []
    supported_count = 0
    similarities = []

    for statement in statements:
        cited_ids = extract_citation_ids(statement)
        statement_without_citations = _strip_citations(statement)
        candidate_texts = [
            source_text_by_id[source_id]
            for source_id in cited_ids
            if source_id in source_text_by_id
        ]
        if not candidate_texts and all_context:
            candidate_texts = [all_context]
        if not candidate_texts:
            unsupported.append(statement)
            similarities.append(0.0)
            continue

        similarity = _max_semantic_similarity(model, statement_without_citations, candidate_texts)
        similarities.append(similarity)
        if similarity >= similarity_threshold:
            supported_count += 1
        else:
            unsupported.append(statement)

    return AnswerQuality(
        groundedness=supported_count / len(statements),
        hallucination_rate=len(unsupported) / len(statements),
        unsupported_statements=unsupported,
        evaluated_statements_count=len(statements),
        average_similarity=sum(similarities) / len(similarities) if similarities else None,
        method=f"semantic_similarity:{model_name}",
    )


def _evaluate_token_overlap(
    *,
    statements: list[str],
    source_text_by_id: dict[str, str],
    all_context: str,
    overlap_threshold: float,
    method: str,
) -> AnswerQuality:
    unsupported = []
    supported_count = 0
    for statement in statements:
        cited_ids = extract_citation_ids(statement)
        statement_without_citations = _strip_citations(statement)
        context = " ".join(
            source_text_by_id[source_id]
            for source_id in cited_ids
            if source_id in source_text_by_id
        )
        if not context:
            context = all_context

        if _is_supported_by_token_overlap(
            statement_without_citations,
            context,
            overlap_threshold=overlap_threshold,
        ):
            supported_count += 1
        else:
            unsupported.append(statement)

    return AnswerQuality(
        groundedness=supported_count / len(statements),
        hallucination_rate=len(unsupported) / len(statements),
        unsupported_statements=unsupported,
        evaluated_statements_count=len(statements),
        average_similarity=None,
        method=method,
    )


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Semantic answer quality requires sentence-transformers.") from exc
    return SentenceTransformer(model_name)


def _max_semantic_similarity(model, statement: str, candidate_texts: list[str]) -> float:
    statement_embedding = model.encode(
        [statement],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_embeddings = model.encode(
        candidate_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = statement_embedding @ candidate_embeddings.T
    return float(similarities.max().item())


def _strip_citations(statement: str) -> str:
    return strip_citation_tags(statement)


def _is_supported_by_token_overlap(statement: str, context: str, *, overlap_threshold: float) -> bool:
    statement_tokens = _content_tokens(statement)
    if not statement_tokens:
        return True
    context_tokens = _content_tokens(context)
    if not context_tokens:
        return False
    overlap = len(statement_tokens & context_tokens) / len(statement_tokens)
    return overlap >= overlap_threshold


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in (value.lower() for value in _TOKEN_PATTERN.findall(text))
        if len(token) >= 4 and token not in _STOPWORDS
    }
