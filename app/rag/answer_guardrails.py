import re

from app.rag.citation_validation import extract_citation_ids, normalize_citation_format, strip_citation_tags
from app.rag.models import RetrievedDocument


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_RECOMMENDATION_TERMS = re.compile(
    r"\b(recommended|recommendation|recommendations|guidelines?\s+recommend|standard\s+of\s+care|should\s+be\s+used)\b",
    re.IGNORECASE,
)
_GUIDELINE_TYPES = {"guideline", "practice guideline"}


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


def _is_orphan_citation(sentence: str) -> bool:
    return not strip_citation_tags(sentence).strip(" \t\r\n.,;:")


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
