"""Replay of PubMedQA's annotation protocol with two LLM annotators.

PQA-L was labelled by two people (Jin et al. 2019, Algorithm 1): annotator 1 read the question,
the abstract and the authors' conclusion; annotator 2 read only the question and the abstract.
When their labels matched, that was the label. When they did not, the same two people discussed
until they agreed, and questions they could not settle were removed from the dataset.

``AnnotationProtocol`` plays that procedure with two agents, on one inference backend or on one
backend per annotator (``backends``), so the two seats can be different models:

  round 0   both agents label independently, each from its own text;
  agreement the shared label is final (status ``agreed``);
  dispute   up to ``max_rounds`` discussion rounds; in each, both agents see their own last
            answer and the colleague's, and are told what the colleague could see; the first
            round in which the labels match ends the discussion (status ``negotiated``);
  no deal   after ``max_rounds`` the question is removed (status ``removed``), as in PQA-L.

With ``justify_spec``, each side first quotes the passage behind its label once a dispute starts,
and the quotes are passed to the discussion prompt (``own_quote`` / ``partner_quote``); a
discussion reply that carries a ``quote`` replaces that side's quote. ``negotiate`` replays the
discussion from given round-0 turns, so a new discussion prompt can be compared on the same
disputes.

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
    quote: str = ""
    kind: str = "label"  # label | justify


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


def parse_quote(content: str) -> str | None:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    quote = data.get("quote") if isinstance(data, dict) else None
    return quote if isinstance(quote, str) else None


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
        backend: Backend | None = None,
        *,
        backends: dict[str, Backend] | None = None,
        label_spec: PromptSpecLike,
        negotiate_spec: PromptSpecLike,
        justify_spec: PromptSpecLike | None = None,
        max_rounds: int = 3,
        num_predict: int = 600,
        attempts: int = 2,
    ) -> None:
        if (backend is None) == (backends is None):
            raise ValueError("pass either one backend for both annotators or backends for each")
        self.backends = backends if backends is not None else {a: backend for a in ANNOTATORS}
        if set(self.backends) != set(ANNOTATORS):
            raise ValueError(f"backends must cover exactly {ANNOTATORS}")
        self.label_spec = label_spec
        self.negotiate_spec = negotiate_spec
        self.justify_spec = justify_spec
        self.max_rounds = max_rounds
        self.num_predict = num_predict
        self.attempts = attempts

    async def _complete(self, spec: PromptSpecLike, annotator: str, fields: dict, parse) -> tuple[Any, str]:
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in spec.messages(**fields)]
        error = ""
        for _ in range(self.attempts):
            try:
                content = await self.backends[annotator].complete(
                    messages, temperature=0.0, num_predict=self.num_predict, response_format=spec.schema
                )
            except Exception as exc:  # backend already retried; record and try once more
                error = f"{type(exc).__name__}: {exc}"
                continue
            parsed = parse(content)
            if parsed is not None:
                return parsed, content
            error = "unparseable reply"
        return None, error

    async def _ask(self, spec: PromptSpecLike, annotator: str, round_number: int, **fields: str) -> Turn:
        def parse(content: str):
            answer = parse_answer(content)
            return (answer, parse_quote(content) or "") if answer[0] is not None else None

        parsed, info = await self._complete(spec, annotator, fields, parse)
        if parsed is None:
            return Turn(annotator, round_number, None, None, "", info)
        (label, confidence, rationale), quote = parsed
        return Turn(annotator, round_number, label, confidence, rationale, quote=quote)

    async def _justify(self, annotator: str, question: str, evidence: str, turn: Turn) -> Turn:
        quote, info = await self._complete(
            self.justify_spec, annotator,
            {"question": question, "evidence": evidence, "own_label": turn.label or "", "own_rationale": turn.rationale},
            parse_quote,
        )
        return Turn(annotator, 0, turn.label, turn.confidence, turn.rationale,
                    error="" if quote is not None else info, quote=quote or "", kind="justify")

    async def run(self, pmid: str, question: str, evidence: dict[str, str]) -> ProtocolResult:
        """``evidence`` maps each annotator to the text it may see."""
        first = await asyncio.gather(
            *(self._ask(self.label_spec, a, 0, question=question, evidence=evidence[a]) for a in ANNOTATORS)
        )
        return await self.negotiate(pmid, question, evidence, list(first))

    async def negotiate(self, pmid: str, question: str, evidence: dict[str, str], first: list[Turn]) -> ProtocolResult:
        """Settle a question from its round-0 turns (one per annotator)."""
        turns = list(first)
        current = {t.annotator: t for t in first}
        initial = {a: current[a].label for a in ANNOTATORS}
        if any(label is None for label in initial.values()):
            return ProtocolResult(pmid, "failed", None, initial, 0, turns)
        if initial[WITH_CONCLUSION] == initial[CONTEXT_ONLY]:
            return ProtocolResult(pmid, "agreed", initial[WITH_CONCLUSION], initial, 0, turns)

        quotes = {a: "" for a in ANNOTATORS}
        if self.justify_spec is not None:
            justified = await asyncio.gather(
                *(self._justify(a, question, evidence[a], current[a]) for a in ANNOTATORS)
            )
            turns.extend(justified)
            if any(t.error for t in justified):
                return ProtocolResult(pmid, "failed", None, initial, 0, turns)
            quotes = {t.annotator: t.quote for t in justified}

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
                        own_quote=quotes[a],
                        partner_quote=quotes[b],
                    )
                    for a, b in ((WITH_CONCLUSION, CONTEXT_ONLY), (CONTEXT_ONLY, WITH_CONCLUSION))
                )
            )
            turns.extend(replies)
            if any(t.label is None for t in replies):
                return ProtocolResult(pmid, "failed", None, initial, round_number, turns)
            current = {t.annotator: t for t in replies}
            quotes = {a: current[a].quote or quotes[a] for a in ANNOTATORS}
            if current[WITH_CONCLUSION].label == current[CONTEXT_ONLY].label:
                return ProtocolResult(pmid, "negotiated", current[WITH_CONCLUSION].label, initial, round_number, turns)
        return ProtocolResult(pmid, "removed", None, initial, self.max_rounds, turns)
