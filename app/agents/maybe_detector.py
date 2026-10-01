"""Standalone `maybe` detector for PubMedQA (category 1: answer splits).

Half of the `maybe` cases that every debate configuration misses share one shape:
the answer exists but is not single-valued — it differs across subgroups, or
across the parts of a compound question, while the question is posed as if it had
one answer. See docs/agents/maybe-detector-spec.md.

This is deliberately NOT a debate persona. A persona's vote gets averaged into a
panel and, measured on balanced90, the dedicated `uncertainty_advocate` sat at the
base rate however its prompt was written. A detector is scored on its own
precision/recall and composed with a binary answer afterwards.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.backends import InferenceBackend
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

SplitKind = Literal["subgroup", "compound_question", "outcome_conflict", "none"]


class AnswerSplitVerdict(BaseModel):
    """Whether the abstract gives a different answer for different slices."""

    split_detected: bool = False
    split_kind: SplitKind = "none"
    # Verbatim spans that carry the conflicting findings; empty when none found.
    conflicting_findings: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""

    @property
    def is_usable(self) -> bool:
        """A split claim without quoted findings is unsupported and ignored."""
        return self.split_detected and bool(self.conflicting_findings)


ANSWER_SPLIT_PROMPT = """
You check ONE thing about a research abstract: does it give the SAME answer
everywhere, or does the answer change depending on which slice you look at?

RESEARCH QUESTION:
{question}

ABSTRACT:
{abstract}

A "split" means the abstract reports findings that point in DIFFERENT directions,
or that differ in significance, across:
- subgroup: the effect holds in one population/lesion type/disease subtype and not
  in another (e.g. significant in ulcerative colitis, only in colonic Crohn's).
- compound_question: the question asks two things joined by "and", and the abstract
  answers them differently (e.g. parents recalled the weight but did not understand it).
- outcome_conflict: different endpoints or markers disagree (e.g. one marker elevated
  while another shows nothing; primary endpoint null while secondary endpoints move).

NOT a split, do not report these:
- one single finding stated with academic caution ("suggests", "may")
- limitations: small sample, retrospective design, short follow-up, single centre
- a call for further research
- several findings that all point the SAME way

Quote the conflicting findings verbatim from the abstract. If you cannot quote at
least two findings that actually disagree, there is no split — say so.

Output ONLY a JSON object:
{{
  "split_detected": true | false,
  "split_kind": "subgroup" | "compound_question" | "outcome_conflict" | "none",
  "conflicting_findings": ["verbatim quote 1", "verbatim quote 2"],
  "confidence": 0.0,
  "rationale": "one sentence naming what disagrees with what"
}}
""".strip()


def _extract_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    data = json.loads(match.group(0) if match else text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    kind = str(data.get("split_kind") or "none").strip().lower()
    if kind not in {"subgroup", "compound_question", "outcome_conflict", "none"}:
        kind = "none"
    findings = data.get("conflicting_findings")
    if not isinstance(findings, list):
        findings = []
    findings = [str(f).strip() for f in findings if str(f).strip()]
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    detected = bool(data.get("split_detected"))
    # A "split" with nothing quoted is the failure mode this detector exists to
    # avoid: an unsupported assertion of uncertainty. Downgrade it.
    if detected and len(findings) < 2:
        detected = False
        kind = "none"
    return {
        "split_detected": detected,
        "split_kind": kind if detected else "none",
        "conflicting_findings": findings,
        "confidence": min(1.0, max(0.0, confidence)),
        "rationale": str(data.get("rationale") or "").strip(),
    }


async def detect_answer_split(
    backend: InferenceBackend,
    *,
    question: str,
    abstract: str,
    temperature: float = 0.0,
    num_predict: int | None = 600,
) -> AnswerSplitVerdict:
    """Ask whether the abstract's answer splits across subgroups / question parts.

    Returns a non-detecting verdict rather than raising when the model emits
    unparseable output, so a detector failure never fabricates uncertainty.
    """
    prompt = ANSWER_SPLIT_PROMPT.format(
        question=(question or "").strip(),
        abstract=(abstract or "").strip(),
    )
    messages = [
        ChatMessage(role="system", content="Return only valid JSON."),
        ChatMessage(role="user", content=prompt),
    ]
    raw = await backend.complete(
        messages, temperature=temperature, num_predict=num_predict
    )
    try:
        return AnswerSplitVerdict.model_validate(_normalize(_extract_json(raw)))
    except Exception:
        logger.warning("Failed to parse AnswerSplitVerdict; reporting no split.", exc_info=True)
        return AnswerSplitVerdict()


# Split kinds that earn an override, measured on balanced90 with qwen2.5:14b:
#   subgroup           25 fires, precision 0.600
#   compound_question   1 fire,  precision 1.000
#   outcome_conflict   20 fires, precision 0.150  <- excluded
# `outcome_conflict` fires on ordinary multi-endpoint studies that do have a clear
# answer, so it lands below the 0.333 base rate. Acting on all three kinds scores
# 0.556 against a 0.656 baseline; acting on the two below scores 0.689.
# It is still detected and recorded — just not acted on.
ACTIONABLE_SPLIT_KINDS: frozenset[str] = frozenset({"subgroup", "compound_question"})


def apply_split_detector(
    binary_label: str | None,
    verdict: AnswerSplitVerdict,
    *,
    min_confidence: float = 0.0,
    actionable_kinds: frozenset[str] = ACTIONABLE_SPLIT_KINDS,
) -> tuple[str | None, bool]:
    """Compose the detector with a binary yes/no answer.

    Returns ``(label, overridden)``. The detector only ever turns a yes/no into a
    ``maybe``; it never rewrites the direction of a binary answer, because it has
    no view on direction.

    Note on ``min_confidence``: measured on balanced90, the model returned
    ``confidence=1.0`` on all 46 fires, so the field currently carries no signal
    and this knob is inert. It is kept for models that do calibrate.
    """
    if binary_label not in {"yes", "no"}:
        return binary_label, False
    if not verdict.is_usable or verdict.confidence < min_confidence:
        return binary_label, False
    if verdict.split_kind not in actionable_kinds:
        return binary_label, False
    return "maybe", True
