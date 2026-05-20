from __future__ import annotations

import re

from app.rag.citation_validation import extract_citation_ids
from app.rag.models import RetrievedDocument
from app.schemas import ChatMessage


_YES_NO_MAYBE_TASK_PATTERNS = (
    re.compile(r"\banswer\s+(?:with\s+)?yes\s*,?\s+no\s*,?\s+(?:or\s+)?maybe\b", re.IGNORECASE),
    re.compile(r"\byes\s*/\s*no\s*/\s*maybe\b", re.IGNORECASE),
    re.compile(r"\byes\s*,\s*no\s*,\s*or\s*maybe\b", re.IGNORECASE),
)
_LABEL_PATTERNS = (
    re.compile(r"^\s*(?:final\s+)?answer\s*[:\-]\s*(yes|no|maybe)\b", re.IGNORECASE),
    re.compile(r"^\s*(yes|no|maybe)\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?answer\s+(?:is|should\s+be)\s+(yes|no|maybe)\b", re.IGNORECASE),
)
_UNCERTAINTY_PATTERN = re.compile(
    r"\b("
    r"maybe|uncertain|uncertainty|inconclusive|insufficient|not\s+enough|cannot\s+(?:determine|conclude)|"
    r"does\s+not\s+definitively|not\s+definitively|mixed|limited|unclear|conflicting"
    r")\b",
    re.IGNORECASE,
)
_NEGATIVE_PATTERN = re.compile(
    r"\b("
    r"does\s+not|did\s+not|do\s+not|was\s+not|were\s+not|is\s+not|are\s+not|"
    r"no\s+(?:evidence|association|benefit|difference|improvement|reduction)|"
    r"not\s+(?:associated|significant|effective|beneficial|superior)"
    r")\b",
    re.IGNORECASE,
)
_POSITIVE_PATTERN = re.compile(
    r"\b("
    r"supports|suggests|indicates|demonstrates|shows|showed|associated\s+with|"
    r"significantly|benefit|beneficial|effective|improved|reduced|increased|"
    r"should|provides?|useful|accurate|reliable|sensitive|specific|predicts?|"
    r"seems?\s+to|can\s+be\s+used|can\s+help|may\s+be\s+(?:useful|specific|effective|beneficial)"
    r")\b",
    re.IGNORECASE,
)


def is_yes_no_maybe_task(messages: list[ChatMessage]) -> bool:
    user_text = " ".join(message.content for message in messages if message.role == "user")
    return is_yes_no_maybe_task_text(user_text)


def is_yes_no_maybe_task_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in _YES_NO_MAYBE_TASK_PATTERNS)


def extract_yes_no_maybe_label(answer: str) -> str | None:
    normalized = _strip_citations(answer).strip()
    for pattern in _LABEL_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(1).lower()
    if _UNCERTAINTY_PATTERN.search(normalized):
        return "maybe"
    return None


def enforce_yes_no_maybe_contract(answer: str, source_documents: list[RetrievedDocument]) -> str:
    label = extract_yes_no_maybe_label(answer) or _infer_label(answer)
    body = _evidence_body(answer, label=label)
    if not extract_citation_ids(body) and source_documents:
        body = _append_primary_citation(body)
    return f"Answer: {label}\nEvidence: {body}".strip()


def _infer_label(answer: str) -> str:
    if _UNCERTAINTY_PATTERN.search(answer):
        return "maybe"
    if _NEGATIVE_PATTERN.search(answer):
        return "no"
    if _POSITIVE_PATTERN.search(answer):
        return "yes"
    return "maybe"


def _evidence_body(answer: str, *, label: str) -> str:
    content = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
    content = re.sub(r"^\s*(?:final\s+)?answer\s*[:\-]\s*(?:yes|no|maybe)\b\s*[,.;:\-]*\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^\s*(?:yes|no|maybe)\b\s*[,.;:\-]*\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^\s*evidence\s*[:\-]\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s+", " ", content).strip()
    if content:
        return content
    if label == "maybe":
        return "The retrieved evidence is insufficient to support a stronger yes or no conclusion."
    return "The retrieved evidence supports this classification."


def _append_primary_citation(body: str) -> str:
    body = body.rstrip()
    match = re.search(r"([.!?])$", body)
    if match:
        return f"{body[: match.start()].rstrip()} [S1]{match.group(1)}"
    return f"{body} [S1]."


def _strip_citations(answer: str) -> str:
    return re.sub(r"\[[Ss][1-9][0-9]*\]", "", answer)
