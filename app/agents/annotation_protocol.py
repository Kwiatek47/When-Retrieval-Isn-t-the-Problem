"""Replay of PubMedQA's annotation protocol with two LLM annotators.

PQA-L was labelled by two people (Jin et al. 2019, Algorithm 1): annotator 1 read the question,
the abstract and the authors' conclusion; annotator 2 read only the question and the abstract.
When their labels matched, that was the label. When they did not, the same two people discussed
until they agreed, and questions they could not settle were removed from the dataset.

``AnnotationProtocol`` plays that procedure with two agents on one inference backend:

  round 0   both agents label independently, each from its own text;
  agreement the shared label is final (status ``agreed``);
  dispute   up to ``max_rounds`` discussion rounds; in each, both agents see their own last
            answer and the colleague's, and are told what the colleague could see; the first
            round in which the labels match ends the discussion (status ``negotiated``);
  no deal   after ``max_rounds`` the question is removed (status ``removed``), as in PQA-L.

Prompts are passed in as specs with ``messages(**fields)`` and ``schema`` (see
``scripts/agents/probe_prompts.py``), so every run records exactly which prompt text it used.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Protocol

from app.schemas import ChatMessage

LABELS = ("yes", "no", "maybe")
WITH_CONCLUSION = "with_conclusion"
CONTEXT_ONLY = "context_only"
ANNOTATORS = (WITH_CONCLUSION, CONTEXT_ONLY)

# What each annotator is told about the colleague's information.
PARTNER_VIEW = {
    WITH_CONCLUSION: "the same abstract without the authors' conclusion, which you can see",
    CONTEXT_ONLY: "the same abstract and also the authors' conclusion, which you have not seen",
}


class PromptSpecLike(Protocol):
    schema: dict

    def messages(self, **values: str) -> list[dict]: ...


class Backend(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...


@dataclass
class Turn:
    annotator: str
    round: int
    label: str | None
    confidence: int | None
    rationale: str
    error: str = ""


@dataclass
class ProtocolResult:
    pmid: str
    status: str  # agreed | negotiated | removed | failed
    final: str | None
    initial: dict[str, str | None]
    rounds_used: int
    turns: list[Turn] = field(default_factory=list)

    @property
    def final_follows(self) -> str:
        """Whose initial label the final label is: both, one annotator, or neither."""
        a, b = self.initial[WITH_CONCLUSION], self.initial[CONTEXT_ONLY]
        if self.final is None:
            return "none"
        if a == b == self.final:
            return "both"
        if self.final == a:
            return WITH_CONCLUSION
        if self.final == b:
            return CONTEXT_ONLY
        return "neither"

    def to_json(self) -> dict:
        record = asdict(self)
        record["final_follows"] = self.final_follows
        return record


def parse_answer(content: str) -> tuple[str | None, int | None, str]:
    """Label, confidence and rationale from a JSON reply; label ``None`` if unusable."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, None, ""
    if not isinstance(data, dict) or data.get("label") not in LABELS:
        return None, None, ""
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        confidence = None
    rationale = data.get("rationale", "")
    return data["label"], confidence, rationale if isinstance(rationale, str) else ""


class AnnotationProtocol:
    def __init__(
        self,
        backend: Backend,
        *,
        label_spec: PromptSpecLike,
        negotiate_spec: PromptSpecLike,
        max_rounds: int = 3,
        num_predict: int = 600,
        attempts: int = 2,
    ) -> None:
        self.backend = backend
        self.label_spec = label_spec
        self.negotiate_spec = negotiate_spec
        self.max_rounds = max_rounds
        self.num_predict = num_predict
        self.attempts = attempts

    async def _ask(self, spec: PromptSpecLike, annotator: str, round_number: int, **fields: str) -> Turn:
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in spec.messages(**fields)]
        error = ""
        for _ in range(self.attempts):
            try:
                content = await self.backend.complete(
                    messages, temperature=0.0, num_predict=self.num_predict, response_format=spec.schema
                )
            except Exception as exc:  # backend already retried; record and try once more
                error = f"{type(exc).__name__}: {exc}"
                continue
            label, confidence, rationale = parse_answer(content)
            if label is not None:
                return Turn(annotator, round_number, label, confidence, rationale)
            error = "unparseable reply"
        return Turn(annotator, round_number, None, None, "", error)

    async def run(self, pmid: str, question: str, evidence: dict[str, str]) -> ProtocolResult:
        """``evidence`` maps each annotator to the text it may see."""
        first = await asyncio.gather(
            *(self._ask(self.label_spec, a, 0, question=question, evidence=evidence[a]) for a in ANNOTATORS)
        )
        turns = list(first)
        current = {t.annotator: t for t in first}
        initial = {a: current[a].label for a in ANNOTATORS}
        if any(label is None for label in initial.values()):
            return ProtocolResult(pmid, "failed", None, initial, 0, turns)
        if initial[WITH_CONCLUSION] == initial[CONTEXT_ONLY]:
            return ProtocolResult(pmid, "agreed", initial[WITH_CONCLUSION], initial, 0, turns)

        for round_number in range(1, self.max_rounds + 1):
            replies = await asyncio.gather(
                *(
                    self._ask(
                        self.negotiate_spec,
                        a,
                        round_number,
                        question=question,
                        evidence=evidence[a],
                        partner_view=PARTNER_VIEW[a],
                        round=str(round_number),
                        own_label=current[a].label or "",
                        own_rationale=current[a].rationale,
                        partner_label=current[b].label or "",
                        partner_rationale=current[b].rationale,
                    )
                    for a, b in ((WITH_CONCLUSION, CONTEXT_ONLY), (CONTEXT_ONLY, WITH_CONCLUSION))
                )
            )
            turns.extend(replies)
            if any(t.label is None for t in replies):
                return ProtocolResult(pmid, "failed", None, initial, round_number, turns)
            current = {t.annotator: t for t in replies}
            if current[WITH_CONCLUSION].label == current[CONTEXT_ONLY].label:
                return ProtocolResult(pmid, "negotiated", current[WITH_CONCLUSION].label, initial, round_number, turns)
        return ProtocolResult(pmid, "removed", None, initial, self.max_rounds, turns)
