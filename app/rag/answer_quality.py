from __future__ import annotations

import re

from app.rag.models import RetrievedDocument
from app.schemas import AnswerQuality


_CITATION_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")
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
        if sentence.strip()
    ]
    if not statements:
        return AnswerQuality(
            groundedness=None,
            hallucination_rate=None,
            unsupported_statements=[],
            evaluated_statements_count=0,
            method="token_overlap_with_retrieved_context",
        )

    unsupported = []
    supported_count = 0
    for statement in statements:
        cited_ids = [f"S{match}" for match in _CITATION_PATTERN.findall(statement)]
        context = " ".join(
            source_text_by_id[source_id]
            for source_id in cited_ids
            if source_id in source_text_by_id
        )
        if not context:
            context = all_context

        if _is_supported(statement, context, overlap_threshold=overlap_threshold):
            supported_count += 1
        else:
            unsupported.append(statement)

    return AnswerQuality(
        groundedness=supported_count / len(statements),
        hallucination_rate=len(unsupported) / len(statements),
        unsupported_statements=unsupported,
        evaluated_statements_count=len(statements),
        method="token_overlap_with_retrieved_context",
    )


def _is_supported(statement: str, context: str, *, overlap_threshold: float) -> bool:
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
