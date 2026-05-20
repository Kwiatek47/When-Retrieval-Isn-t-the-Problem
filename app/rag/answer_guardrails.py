import re

from app.rag.citation_validation import (
    extract_citation_ids,
    normalize_citation_format,
    requires_citation,
    strip_citation_tags,
)
from app.rag.models import RetrievedDocument


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[\w]+", re.IGNORECASE)
_RECOMMENDATION_TERMS = re.compile(
    r"\b(recommended|recommendation|recommendations|guidelines?\s+recommend|standard\s+of\s+care|should\s+be\s+used)\b",
    re.IGNORECASE,
)
_GUIDELINE_TYPES = {"guideline", "practice guideline"}
_CITATION_REPAIR_MIN_OVERLAP = 0.25
_REPAIR_STOPWORDS = {
    "and",
    "are",
    "but",
    "dla",
    "for",
    "from",
    "has",
    "have",
    "into",
    "jest",
    "oraz",
    "that",
    "the",
    "this",
    "was",
    "were",
    "with",
}


def apply_answer_guardrails(answer: str, source_documents: list[RetrievedDocument]) -> str:
    """Apply deterministic safety/style guardrails to generated RAG answers."""

    normalized = normalize_citation_format(answer)
    source_by_id = {f"S{index + 1}": document for index, document in enumerate(source_documents)}
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized)
        if sentence.strip()
    ]

    guarded_sentences = []
    for sentence in sentences:
        if _is_orphan_citation(sentence):
            continue
        guarded_sentences.append(_neutralize_unsupported_recommendation(sentence, source_by_id))

    return " ".join(guarded_sentences).strip()


def repair_missing_citations(answer: str, source_documents: list[RetrievedDocument]) -> str:
    """Add a citation to uncited claims only when one retrieved source clearly supports the sentence."""

    if not answer.strip() or not source_documents:
        return answer

    source_tokens = {
        f"S{index + 1}": _content_tokens(f"{document.title} {document.content}")
        for index, document in enumerate(source_documents)
    }
    repaired_sentences = []
    for sentence in _sentences(answer):
        if extract_citation_ids(sentence) or not requires_citation(sentence):
            repaired_sentences.append(sentence)
            continue

        citation_id = _best_supporting_source(sentence, source_tokens)
        repaired_sentences.append(_append_citation(sentence, citation_id) if citation_id else sentence)

    return " ".join(repaired_sentences).strip()


def _is_orphan_citation(sentence: str) -> bool:
    return not strip_citation_tags(sentence).strip(" \t\r\n.,;:")


def _sentences(content: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", content).strip()
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized) if sentence.strip()]


def _best_supporting_source(sentence: str, source_tokens: dict[str, set[str]]) -> str | None:
    statement_tokens = _content_tokens(strip_citation_tags(sentence))
    if not statement_tokens:
        return None

    best_id = None
    best_overlap = 0.0
    for citation_id, tokens in source_tokens.items():
        if not tokens:
            continue
        overlap = len(statement_tokens & tokens) / len(statement_tokens)
        if overlap > best_overlap:
            best_id = citation_id
            best_overlap = overlap
    if best_overlap < _CITATION_REPAIR_MIN_OVERLAP:
        return None
    return best_id


def _append_citation(sentence: str, citation_id: str) -> str:
    match = re.search(r"([.!?])$", sentence)
    if match:
        return f"{sentence[: match.start()].rstrip()} [{citation_id}]{match.group(1)}"
    return f"{sentence} [{citation_id}]"


def _content_tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in (value.lower() for value in _TOKEN_PATTERN.findall(text))
        if len(token) >= 4 and token not in _REPAIR_STOPWORDS
    }


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _neutralize_unsupported_recommendation(
    sentence: str,
    source_by_id: dict[str, RetrievedDocument],
) -> str:
    if not _RECOMMENDATION_TERMS.search(sentence):
        return sentence
    if _has_guideline_citation(sentence, source_by_id):
        return sentence

    neutralized = sentence
    neutralized = re.sub(
        r"^(.+?)\s+(?:are|is)\s+recommended\s+for\s+(.+?)(\s+(?:\[[^\]]+\]\s*)+)([.!?])?$",
        r"Retrieved evidence discusses \1 for \2\3\4",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(
        r"^(.+?)\s+(?:are|is)\s+recommended\s+in\s+(.+?)(\s+(?:\[[^\]]+\]\s*)+)([.!?])?$",
        r"Retrieved evidence discusses \1 in \2\3\4",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(
        r"\b(?:are|is)\s+recommended\s+for\b",
        "are discussed for",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(
        r"\b(?:are|is)\s+recommended\s+in\b",
        "are discussed in",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(
        r"\bshould\s+be\s+used\b",
        "are discussed in retrieved evidence",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(
        r"\bguidelines?\s+recommend\b",
        "retrieved evidence discusses",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(
        r"\bstandard\s+of\s+care\b",
        "therapeutic option discussed in retrieved evidence",
        neutralized,
        flags=re.IGNORECASE,
    )
    neutralized = re.sub(r"\s+([.,;:])", r"\1", neutralized)
    return neutralized


def _has_guideline_citation(sentence: str, source_by_id: dict[str, RetrievedDocument]) -> bool:
    cited_ids = extract_citation_ids(sentence)
    return any(_is_guideline_source(source_by_id[citation_id]) for citation_id in cited_ids if citation_id in source_by_id)


def _is_guideline_source(document: RetrievedDocument) -> bool:
    publication_types = _publication_types(document.metadata.get("publicationTypes"))
    return bool(publication_types & _GUIDELINE_TYPES)


def _publication_types(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped.strip("[]")
        return {item.strip().strip("'\"").lower() for item in re.split(r"[;,]", stripped) if item.strip()}
    return {str(item).strip().lower() for item in value if str(item).strip()}
