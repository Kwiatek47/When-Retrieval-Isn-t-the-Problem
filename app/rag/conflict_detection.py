from __future__ import annotations

import re
from typing import Iterable

from app.rag.models import RetrievedDocument
from app.schemas import Citation, EvidenceConflictInfo, EvidenceConflictPair


_TOKEN_PATTERN = re.compile(r"[\w]+", re.IGNORECASE)
_STOPWORDS = {
    "and",
    "are",
    "but",
    "can",
    "for",
    "from",
    "into",
    "jest",
    "oraz",
    "pod",
    "przy",
    "the",
    "this",
    "with",
    "without",
}

_SUPPORT_PATTERNS = (
    re.compile(r"\brecommend(?:ed|s|ation)?\b", re.IGNORECASE),
    re.compile(r"\bshould be (?:used|offered|considered)\b", re.IGNORECASE),
    re.compile(r"\bbenefit(?:s|ed)?\b", re.IGNORECASE),
    re.compile(r"\breduc(?:e|ed|es|ing|tions?)\b", re.IGNORECASE),
    re.compile(r"\beffective\b", re.IGNORECASE),
    re.compile(r"\blower(?:ed|s)?\b", re.IGNORECASE),
)
_CAUTION_PATTERNS = (
    re.compile(r"\bnot recommended\b", re.IGNORECASE),
    re.compile(r"\bavoid(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bcontraindicat(?:ed|ion|ions)\b", re.IGNORECASE),
    re.compile(r"\bshould not\b", re.IGNORECASE),
    re.compile(r"\bincrease(?:d|s)? (?:risk|bleeding|mortality|harm|adverse)\b", re.IGNORECASE),
    re.compile(r"\bharm(?:ful|s)?\b", re.IGNORECASE),
    re.compile(r"\badverse event(?:s)?\b", re.IGNORECASE),
)


def detect_evidence_conflicts(
    documents: list[RetrievedDocument],
    citations: list[Citation],
    query: str,
) -> EvidenceConflictInfo:
    labels = [citation.id for citation in citations]
    evidence = [
        _Evidence(
            source_id=source_id,
            document=document,
            terms=_terms_for(document, query),
            polarity=_polarity_for(document.content),
            year=_year_for(document),
        )
        for source_id, document in zip(labels, documents, strict=False)
    ]

    pairs: list[EvidenceConflictPair] = []
    for index, left in enumerate(evidence):
        for right in evidence[index + 1 :]:
            if {left.polarity, right.polarity} != {"support", "caution"}:
                continue
            shared_terms = sorted(left.terms & right.terms)
            if not shared_terms:
                continue
            pairs.append(
                EvidenceConflictPair(
                    source_ids=[left.source_id, right.source_id],
                    shared_terms=shared_terms[:6],
                    reason=(
                        "Retrieved sources contain opposing recommendation or risk language "
                        "for overlapping clinical terms."
                    ),
                    newer_source_id=_newer_source_id(left, right),
                )
            )

    if not pairs:
        return EvidenceConflictInfo(detected=False)

    return EvidenceConflictInfo(
        detected=True,
        strategy="conflicting_evidence_flag",
        conflict_count=len(pairs),
        pairs=pairs,
        instruction=(
            "Conflicting evidence was detected. Present the competing source positions with citations, "
            "do not average them into one recommendation, and abstain from a specific clinical directive "
            "unless the retrieved sources clearly establish a priority such as newer or stronger evidence."
        ),
    )


class _Evidence:
    def __init__(self, *, source_id: str, document: RetrievedDocument, terms: set[str], polarity: str, year: int | None):
        self.source_id = source_id
        self.document = document
        self.terms = terms
        self.polarity = polarity
        self.year = year


def _terms_for(document: RetrievedDocument, query: str) -> set[str]:
    candidates = " ".join(
        [
            query,
            document.title,
            str(document.metadata.get("topic") or ""),
            str(document.metadata.get("meshTerms") or ""),
        ]
    )
    return {
        token
        for token in _tokens(candidates)
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _polarity_for(text: str) -> str:
    support = _matches_any(_SUPPORT_PATTERNS, text)
    caution = _matches_any(_CAUTION_PATTERNS, text)
    if support and not caution:
        return "support"
    if caution and not support:
        return "caution"
    if support and caution:
        return "mixed"
    return "neutral"


def _matches_any(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _year_for(document: RetrievedDocument) -> int | None:
    value = document.metadata.get("year")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _newer_source_id(left: _Evidence, right: _Evidence) -> str | None:
    if left.year is None or right.year is None or left.year == right.year:
        return None
    return left.source_id if left.year > right.year else right.source_id
